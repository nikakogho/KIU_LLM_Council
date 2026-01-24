#!/usr/bin/env python
"""
drafts_only_results_distiller.py

Reads council_drafts_only_results.json and extracts question, correct_answer, and winner_text
into a simpler JSON format.
"""

import json
from pathlib import Path


def main():
    # Paths
    results_dir = Path(__file__).parent.parent / "results"
    input_file = results_dir / "council_drafts_only_results.json"
    output_file = results_dir / "council_drafts_only_results_distilled.json"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return 1
    
    # Load results
    with open(input_file, encoding='utf-8') as f:
        drafts_only_results = json.load(f)
    
    print(f"Loaded {len(drafts_only_results)} results from {input_file}")
    
    # Extract key fields
    distilled = []
    for item in drafts_only_results:
        distilled.append({
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
            "selected_answer": item.get("winner_text", "")
        })
    
    # Save distilled results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(distilled, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(distilled)} distilled results to {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
