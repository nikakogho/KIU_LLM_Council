#!/usr/bin/env python
"""
accuracy_plot.py

Generates a bar chart comparing accuracy across:
- One-shot mode (individual LLM responses)
- Council drafts-only mode
- Council full mode (with review and revision)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_one_shot_scores():
    """Load one-shot results and calculate scores per provider."""
    results_dir = Path(__file__).parent.parent / "results"
    one_shot_file = results_dir / "one_shot_results_correctness.json"
    
    if not one_shot_file.exists():
        print(f"Warning: {one_shot_file} not found")
        return {}
    
    with open(one_shot_file, encoding='utf-8') as f:
        data = json.load(f)
    
    scores = {
        'gpt-5-nano': 0,
        'grok-3-mini': 0,
        'claude-haiku-4-5': 0,
        'gemini-2.5-flash-lite': 0
    }
    
    # Map providers to model names
    provider_to_model = {
        'openai': 'gpt-5-nano',
        'xai': 'grok-3-mini',
        'anthropic': 'claude-haiku-4-5',
        'gemini': 'gemini-2.5-flash-lite'
    }
    
    for entry in data:
        for provider, model in provider_to_model.items():
            if entry.get(f'{provider}_correct', False):
                scores[model] += 1
    
    return scores


def check_answer_match(selected_text, correct_answer):
    """
    Simple heuristic to check if selected answer matches correct answer.
    Extracts numbers and checks if they appear in the answer.
    """
    if not selected_text or not correct_answer:
        return False
    
    selected_lower = selected_text.lower()
    correct_lower = correct_answer.lower()
    
    # Direct substring match
    if correct_lower in selected_lower:
        return True
    
    # Try to extract key numbers/fractions from correct answer
    # and check if they appear in selected text
    import re
    
    # Extract numbers (including decimals and fractions)
    numbers = re.findall(r'\d+(?:\.\d+)?(?:/\d+)?', correct_lower)
    if numbers:
        # Check if any key number appears in selected text
        for num in numbers[-3:]:  # Check last few numbers (likely the answer)
            if num in selected_lower:
                return True
    
    return False


def load_council_scores(results_file):
    """Load council results and calculate correctness score."""
    results_dir = Path(__file__).parent.parent / "results"
    council_file = results_dir / results_file
    
    if not council_file.exists():
        print(f"Warning: {council_file} not found")
        return 0
    
    with open(council_file, encoding='utf-8') as f:
        data = json.load(f)
    
    correct_count = 0
    for entry in data:
        selected_answer = entry.get('selected_answer', '')
        correct_answer = entry.get('correct_answer', '')
        
        if check_answer_match(selected_answer, correct_answer):
            correct_count += 1
    
    return correct_count


def main():
    print("Calculating scores...")
    
    # Load one-shot scores
    one_shot_scores = load_one_shot_scores()
    print(f"One-shot scores: {one_shot_scores}")
    
    # Load council scores
    drafts_only_score = load_council_scores('council_drafts_only_results_distilled.json')
    print(f"Council (drafts-only) score: {drafts_only_score}/25")
    
    full_council_score = load_council_scores('council_full_results_distilled.json')
    print(f"Council (full) score: {full_council_score}/25")
    
    # Prepare data for plotting
    models = list(one_shot_scores.keys())
    one_shot_vals = list(one_shot_scores.values())
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Plot one-shot results
    bars1 = ax.bar(x, one_shot_vals, width, label='One-shot', color='#3498db', alpha=0.8)
    
    # Add council results as horizontal lines or separate bars
    ax.axhline(y=drafts_only_score, color='#e74c3c', linestyle='--', linewidth=2, label=f'Council (Drafts-only): {drafts_only_score}/25', alpha=0.7)
    ax.axhline(y=full_council_score, color='#2ecc71', linestyle='--', linewidth=2, label=f'Council (Full): {full_council_score}/25', alpha=0.7)
    
    # Customize axes
    ax.set_ylabel('Correct Answers (out of 25)', fontsize=12, fontweight='bold')
    ax.set_title('LLM Accuracy Comparison: One-shot vs Council Modes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 26)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "results" / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {output_path}")
    
    # Also create a summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY (out of 25)")
    print("="*60)
    for model, score in one_shot_scores.items():
        pct = (score / 25) * 100
        print(f"{model:25} {score:2}/25  ({pct:5.1f}%)")
    print("-"*60)
    pct_drafts = (drafts_only_score / 25) * 100
    print(f"{'Council (drafts-only)':25} {drafts_only_score:2}/25  ({pct_drafts:5.1f}%)")
    pct_full = (full_council_score / 25) * 100
    print(f"{'Council (full review)':25} {full_council_score:2}/25  ({pct_full:5.1f}%)")
    print("="*60)
    
    plt.show()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
