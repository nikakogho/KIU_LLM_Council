from pathlib import Path

source_dir = Path(__file__).parent.parent / "llm_council"
output_file = Path(__file__).parent / "llm_council_src.txt"

py_files = sorted(source_dir.rglob("*.py"))

with open(output_file, "w", encoding="utf-8") as f:
    for py_file in py_files:
        relative_path = py_file.relative_to(source_dir)
        f.write(f"\n{'='*80}\n")
        f.write(f"FILE: {relative_path}\n")
        f.write(f"{'='*80}\n\n")
        
        with open(py_file, "r", encoding="utf-8") as src:
            f.write(src.read())
        
        f.write("\n\n")

print(f"Aggregated {len(py_files)} Python files to {output_file}")