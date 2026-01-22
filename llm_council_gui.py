# llm_council_gui.py
import asyncio
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from llm_council.app import plan_council, run_council, RunCallbacks
from llm_council.role_planner import apply_user_override
from llm_council.persistence import save_state, load_state, default_run_path
from llm_council.runner import build_client_by_provider


def _model_for_provider_from_plan(plan, provider: str) -> str | None:
    if not plan:
        return None
    roster = [plan.judge] + list(plan.solvers)
    for m in roster:
        if m.provider == provider:
            return m.model
    return None


def _as_dict_maybe(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):  # pydantic
        try:
            return x.model_dump()
        except Exception:
            return None
    return None


def _format_plan_block(plan) -> str:
    lines = []
    lines.append("=== Default plan ===")
    lines.append(f"Judge:  {plan.judge.provider}  ({plan.judge.model})")
    lines.append("Solvers:")
    for s in plan.solvers:
        lines.append(f"  - {s.provider} ({s.model})")
    return "\n".join(lines)


def _format_opinions_block(plan) -> str:
    lines = []
    lines.append("=== Role opinions ===")
    for prov, res in plan.opinions.items():
        if res.parsed:
            self_role = res.parsed.self.preferred_role
            self_conf = res.parsed.self.confidence
            rec = res.parsed.recommended_judge
            lines.append(
                f"- {prov:10} self={self_role} conf={self_conf:.2f} | recommends={rec.provider} conf={rec.confidence:.2f}"
            )
        else:
            lines.append(f"- {prov:10} (unparsed) err={res.parse_error}")

    lines.append("")
    lines.append("=== Judge score breakdown (components) ===")
    for prov, b in plan.score_breakdown.items():
        lines.append(
            f"- {prov:10} prior={b.get('prior',0):.2f} self={b.get('self',0):.2f} nominations={b.get('nominations',0):.2f}"
        )
    return "\n".join(lines)


def _format_judge_block(state) -> str:
    j = state.judge
    if not j:
        return "=== Judge decision ===\n(no judge result)"

    parsed_ok = (j.parsed is not None) and (j.parse_error is None)
    parsed_d = _as_dict_maybe(j.parsed) if parsed_ok else None

    # Determine whether fallback was used
    used_fallback = True
    judge_winner = None
    if parsed_d:
        judge_winner = parsed_d.get("winner_provider")
        used_fallback = (judge_winner != state.winner_provider)

    lines = []
    lines.append("=== Judge decision ===")
    lines.append(f"Judge:   {j.provider} ({j.model})")
    lines.append(f"Attempts: {j.attempts}")
    lines.append(f"Parsed JSON: {'YES' if parsed_ok else 'NO'}")
    if j.parse_error:
        lines.append(f"Parse error: {j.parse_error}")
    lines.append(f"Used fallback: {'YES' if used_fallback else 'NO'}")

    # Decision details
    if parsed_d:
        lines.append(f"Picked winner: {parsed_d.get('winner_provider')}")
        rationale = parsed_d.get("rationale")
        if rationale:
            lines.append("")
            lines.append("Rationale:")
            lines.append(rationale)

        ranking = parsed_d.get("ranking") or []
        if isinstance(ranking, list) and ranking:
            lines.append("")
            lines.append("Ranking:")
            for item in ranking:
                if isinstance(item, dict):
                    p = item.get("provider")
                    sc = item.get("score")
                    rsn = item.get("reason")
                    lines.append(f"- {p}: score={sc} reason={rsn}")
    else:
        lines.append("")
        lines.append("Judge output was not parseable; winner may come from fallback scoring.")
    return "\n".join(lines)


def _format_winner_display(state) -> str:
    wp = state.winner_provider or "(none)"
    wm = _model_for_provider_from_plan(state.plan, state.winner_provider) if state.winner_provider else None
    header = f"Winner: {wp}" + (f" ({wm})" if wm else "")
    text = state.winner_text or ""
    return header + "\n\n" + text


class CouncilGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LLM Council GUI")
        self.root.geometry("1150x750")

        self.msgq: queue.Queue[tuple[str, str]] = queue.Queue()

        # State
        self.clients = None
        self.roster = None
        self.client_by_provider = None
        self.plan = None
        self.last_state = None
        self._worker_thread = None

        self._build_ui()
        self._init_clients()
        self._poll_queue()

    # ---------- UI ----------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="both", expand=True)

        # Problem input
        problem_frame = ttk.LabelFrame(top, text="Problem Statement", padding=8)
        problem_frame.pack(fill="x")

        self.problem_text = tk.Text(problem_frame, height=6, wrap="word")
        self.problem_text.pack(fill="x", expand=True)

        # Controls row
        controls = ttk.Frame(top, padding=(0, 8))
        controls.pack(fill="x")

        self.btn_plan = ttk.Button(controls, text="Plan roles", command=self.on_plan_roles)
        self.btn_plan.pack(side="left")

        self.btn_run = ttk.Button(controls, text="Run council", command=self.on_run_council)
        self.btn_run.pack(side="left", padx=(8, 0))

        self.btn_load = ttk.Button(controls, text="Load run JSON", command=self.on_load_run)
        self.btn_load.pack(side="left", padx=(8, 0))

        self.btn_save = ttk.Button(controls, text="Save last run JSON", command=self.on_save_last_run)
        self.btn_save.pack(side="left", padx=(8, 0))

        # Options
        opts = ttk.Frame(controls)
        opts.pack(side="right")

        self.var_no_reviews = tk.BooleanVar(value=False)
        self.var_no_revise = tk.BooleanVar(value=False)

        ttk.Checkbutton(opts, text="Skip reviews", variable=self.var_no_reviews).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(opts, text="Skip revise", variable=self.var_no_revise).pack(side="left", padx=(0, 10))

        # Judge override (simple)
        judge_frame = ttk.LabelFrame(top, text="Plan + Judge Override", padding=8)
        judge_frame.pack(fill="x")

        row = ttk.Frame(judge_frame)
        row.pack(fill="x")

        ttk.Label(row, text="Current plan:").pack(side="left")
        self.plan_label = ttk.Label(row, text="(not planned yet)")
        self.plan_label.pack(side="left", padx=(8, 0))

        ttk.Label(row, text="Override judge:").pack(side="left", padx=(30, 0))
        self.judge_var = tk.StringVar(value="")
        self.judge_combo = ttk.Combobox(row, textvariable=self.judge_var, state="readonly", width=18, values=[])
        self.judge_combo.pack(side="left", padx=(8, 0))
        self.judge_combo.set("")

        # Status bar
        status_frame = ttk.Frame(top, padding=(0, 8))
        status_frame.pack(fill="x")

        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.status_label = ttk.Label(status_frame, text="Idle")
        self.status_label.pack(side="left", padx=(8, 0))

        ttk.Label(status_frame, text="Winner:").pack(side="left", padx=(30, 0))
        self.winner_label = ttk.Label(status_frame, text="(none)")
        self.winner_label.pack(side="left", padx=(8, 0))

        # Split view: logs + winner text (bigger now, since no extra panel)
        paned = ttk.Panedwindow(top, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        logs_frame = ttk.LabelFrame(left, text="Live Log", padding=6)
        logs_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(logs_frame, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        winner_frame = ttk.LabelFrame(right, text="Winner Text / Loaded Run", padding=6)
        winner_frame.pack(fill="both", expand=True)

        self.winner_text = tk.Text(winner_frame, wrap="word")
        self.winner_text.pack(fill="both", expand=True)

    # ---------- Logging / queue ----------

    def _log(self, s: str):
        self.log_text.insert("end", s + "\n")
        self.log_text.see("end")

    def _set_status(self, s: str):
        self.status_label.configure(text=s)

    def _set_winner(self, s: str):
        self.winner_label.configure(text=s)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self._set_status(payload)
                elif kind == "winner":
                    self._set_winner(payload)
                elif kind == "winner_text":
                    self.winner_text.delete("1.0", "end")
                    self.winner_text.insert("end", payload)
                elif kind == "plan_label":
                    self.plan_label.configure(text=payload)
                elif kind == "enable":
                    self._set_controls_enabled(True)
                elif kind == "disable":
                    self._set_controls_enabled(False)
        except queue.Empty:
            pass

        self.root.after(50, self._poll_queue)

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in [self.btn_plan, self.btn_run, self.btn_load, self.btn_save]:
            b.configure(state=state)
        self.judge_combo.configure(state=("readonly" if enabled else "disabled"))

    def _run_async_in_thread(self, coro):
        self.msgq.put(("disable", ""))

        def worker():
            try:
                asyncio.run(coro)
            except Exception as e:
                self.msgq.put(("log", f"[ERROR] {e}"))
            finally:
                self.msgq.put(("enable", ""))

        t = threading.Thread(target=worker, daemon=True)
        self._worker_thread = t
        t.start()

    # ---------- Init clients ----------

    def _init_clients(self):
        try:
            clients, roster, mapping = build_client_by_provider()
            self.clients = clients
            self.roster = roster
            self.client_by_provider = mapping

            if len(roster) < 2:
                messagebox.showwarning(
                    "Not enough providers",
                    "Need at least 2 configured providers (API keys) to run a council.",
                )

            # Log provider + model (requested)
            joined = ", ".join([f"{m.provider}({m.model})" for m in roster])
            self._log(f"Loaded providers: {joined}")
        except Exception as e:
            messagebox.showerror("Init error", str(e))

    # ---------- Plan / Run / Load / Save ----------

    def on_plan_roles(self):
        problem = self.problem_text.get("1.0", "end").strip()
        if not problem:
            messagebox.showwarning("Missing problem", "Please enter a problem statement.")
            return
        if not self.clients or not self.roster:
            messagebox.showwarning("Not ready", "Clients/roster not initialized.")
            return

        async def go():
            self.msgq.put(("status", "Planning roles..."))
            self.msgq.put(("log", "\n=== PHASE: role_planning ==="))

            plan = await plan_council(problem_statement=problem, clients=self.clients, roster=self.roster)
            self.plan = plan

            self.msgq.put(("log", "=== DONE:  role_planning ==="))
            self.msgq.put(("status", "Plan ready."))

            # Update plan label + override dropdown
            plan_label = f"Judge={plan.judge.provider} | Solvers={[s.provider for s in plan.solvers]}"
            self.msgq.put(("plan_label", plan_label))

            providers = [plan.judge.provider] + [s.provider for s in plan.solvers]
            self.root.after(0, lambda: self._set_judge_options(providers))

            # Put plan + opinions into LIVE LOG (requested)
            self.msgq.put(("log", "\n" + _format_plan_block(plan)))
            self.msgq.put(("log", "\n" + _format_opinions_block(plan)))

        self._run_async_in_thread(go())

    def _set_judge_options(self, providers: list[str]):
        self.judge_combo["values"] = [""] + providers  # empty = no override
        self.judge_combo.set("")

    def on_run_council(self):
        problem = self.problem_text.get("1.0", "end").strip()
        if not problem:
            messagebox.showwarning("Missing problem", "Please enter a problem statement.")
            return
        if not self.plan:
            messagebox.showwarning("No plan", "Click 'Plan roles' first.")
            return

        # Apply override locally (no extra role planning)
        override = (self.judge_var.get() or "").strip()
        plan = self.plan
        if override:
            try:
                plan = apply_user_override(plan, judge_provider=override)
                self.msgq.put(("log", f"\n[override] Using judge={plan.judge.provider} ({plan.judge.model})"))
                self.msgq.put(("plan_label", f"Judge={plan.judge.provider} | Solvers={[s.provider for s in plan.solvers]}"))
            except Exception as e:
                messagebox.showerror("Invalid override", str(e))
                return

        do_reviews = not self.var_no_reviews.get()
        do_revise = not self.var_no_revise.get()

        callbacks = RunCallbacks(
            on_phase_start=lambda name: (
                self.msgq.put(("log", f"\n=== PHASE: {name} ===")),
                self.msgq.put(("status", f"Running: {name}")),
            ),
            on_phase_end=lambda name: self.msgq.put(("log", f"=== DONE:  {name} ===")),
            on_draft=lambda d: self.msgq.put(
                ("log", f"[draft] {d.provider:10} {'OK' if not d.error else 'ERR'}  chars={len(d.text or '')}")
            ),
            on_review=lambda r: self.msgq.put(
                (
                    "log",
                    f"[review] {r.reviewer_provider:10}->{r.target_provider:10} "
                    f"{'OK' if (r.parsed and not r.parse_error) else 'BAD_JSON'} attempts={r.attempts}",
                )
            ),
            on_revision=lambda r: self.msgq.put(
                ("log", f"[revise] {r.provider:10} {'OK' if not r.error else 'ERR'}  chars={len(r.text or '')}")
            ),
        )

        async def go():
            self.msgq.put(("status", "Running council..."))
            self.msgq.put(("winner", "(running)"))

            state = await run_council(
                problem_statement=problem,
                plan=plan,
                client_by_provider=self.client_by_provider,
                callbacks=callbacks,
                do_reviews=do_reviews,
                do_revise=do_revise,
            )

            self.last_state = state

            # Judge details into live log (requested)
            self.msgq.put(("log", "\n" + _format_judge_block(state)))

            self.msgq.put(("status", "Done."))
            self.msgq.put(("winner", state.winner_provider or "(none)"))

            # Winner text should start with model/provider (requested)
            self.msgq.put(("winner_text", _format_winner_display(state)))

            # Auto-save to default path
            out_path = default_run_path()
            save_state(state, out_path)
            self.msgq.put(("log", f"\n[SAVED] {out_path}"))

        self._run_async_in_thread(go())

    def on_save_last_run(self):
        if not self.last_state:
            messagebox.showinfo("No run", "No run in memory yet. Run the council or load a run JSON first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save run JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="runs",
        )
        if not path:
            return

        try:
            save_state(self.last_state, Path(path))
            self._log(f"[SAVED] {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def on_load_run(self):
        path = filedialog.askopenfilename(
            title="Load run JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="runs",
        )
        if not path:
            return

        try:
            state = load_state(Path(path))
            self.last_state = state

            self.msgq.put(("status", f"Loaded: {Path(path).name}"))
            self.msgq.put(("winner", state.winner_provider or "(none)"))

            # Winner pane with provider+model header (requested)
            self.msgq.put(("winner_text", _format_winner_display(state)))

            self._log(f"\n[LOADED] {path}")

            if state.plan:
                plan_label = f"Judge={state.plan.judge.provider} | Solvers={[s.provider for s in state.plan.solvers]}"
                self.msgq.put(("plan_label", plan_label))

            # Also show judge decision info in log when loading
            self._log("\n" + _format_judge_block(state))

        except Exception as e:
            messagebox.showerror("Load error", str(e))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    app = CouncilGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
