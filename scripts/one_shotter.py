"""
one_shotter.py

Queries each question in problem_dataset.json to all available LLM models in parallel,
and saves results as one_shot_results.json.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_council.runner import build_client_by_provider


async def query_all_models_for_question(
    question: str, client_by_provider, roster
) -> list[dict[str, Any]]:
    """
    Query all models with a single question in parallel.
    Returns a list of dicts with 'provider' and 'answer' fields.
    """
    # Create tasks for all clients
    tasks = []
    providers = []
    
    for model_info in roster:
        provider = model_info.provider
        client = client_by_provider[provider]
        tasks.append(client.generate(question, system_prompt="Answer in under 200 words."))
        providers.append(provider)
    
    # Run all in parallel
    replies = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    answers = []
    for provider, reply in zip(providers, replies):
        if isinstance(reply, Exception):
            answer_text = f"ERROR: {str(reply)}"
        elif hasattr(reply, 'error') and reply.error:
            answer_text = f"ERROR: {reply.error}"
        else:
            answer_text = reply.text if hasattr(reply, 'text') else str(reply)
        
        answers.append({
            "provider": provider,
            "answer": answer_text
        })
    
    return answers


async def main():
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets/problem_dataset.json"
    
    if not dataset_path.exists():
        print(f"Error: Could not find {dataset_path}")
        return 1
    
    with open(dataset_path, "r", encoding='utf-8') as f:
        problems = json.load(f)
    
    print(f"Loaded {len(problems)} problems from {dataset_path}")
    
    # Build clients
    clients, roster, client_by_provider = build_client_by_provider()
    
    if not clients:
        print("Error: No LLM clients configured. Check your API keys in settings.")
        return 1
    
    print(f"Using {len(roster)} LLM providers:")
    for model_info in roster:
        print(f"  - {model_info.provider} ({model_info.model})")
    
    # Query all models for all problems
    print(f"\nQuerying {len(problems)} problems x {len(roster)} providers...")
    
    results = []
    for idx, problem in enumerate(problems, 1):
        print(f"  [{idx}/{len(problems)}] Querying models...", end="", flush=True)
        
        question = problem.get("question", "")
        correct_answer = problem.get("answer", "")
        
        # Query all models for this question in parallel
        answers_from_clients = await query_all_models_for_question(
            question, client_by_provider, roster
        )
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "answers_from_clients": answers_from_clients
        })
        
        print(" Done")
    
    # Save results
    output_path = Path(__file__).parent.parent / "one_shot_results.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Total entries: {len(results)}")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
