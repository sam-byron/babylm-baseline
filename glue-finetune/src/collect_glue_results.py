#!/usr/bin/env python3
"""
collect_glue_results.py

Script to traverse fine-tuned-ltg-bert-glue directory and average eval_accuracy 
for specified GLUE tasks.

Usage:
    python collect_glue_results.py task1 task2 task3 ...
    python collect_glue_results.py cola sst2 mnli
    python collect_glue_results.py --all  # Process all available tasks

The script will:
1. Find the fine-tuned-ltg-bert-glue directory
2. Read eval_results.json from each specified task directory
3. Extract eval_accuracy values
4. Calculate and display the average accuracy
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def find_glue_results_dir() -> Optional[Path]:
    """Find the fine-tuned-ltg-bert-glue directory."""
    # Start from current script location and search up
    current_dir = Path(__file__).parent.parent
    
    # Common locations to search
    search_paths = [
        current_dir / "ltgbert-glue-finetune" / "fine-tuned-ltg-bert-glue",
        current_dir / "fine-tuned-ltg-bert-glue",
        Path.cwd() / "ltgbert-glue-finetune" / "fine-tuned-ltg-bert-glue",
        Path.cwd() / "fine-tuned-ltg-bert-glue"
    ]
    
    for path in search_paths:
        if path.exists() and path.is_dir():
            return path
    
    return None


def get_available_tasks(glue_dir: Path) -> List[str]:
    """Get list of available GLUE tasks in the directory."""
    tasks = []
    for item in glue_dir.iterdir():
        if item.is_dir() and (item / "eval_results.json").exists():
            tasks.append(item.name)
    return sorted(tasks)


def load_eval_results(task_dir: Path) -> Optional[Dict]:
    """Load eval_results.json from a task directory."""
    eval_results_path = task_dir / "eval_results.json"
    
    if not eval_results_path.exists():
        print(f"⚠️  Warning: eval_results.json not found in {task_dir}")
        return None
    
    try:
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error reading {eval_results_path}: {e}")
        return None


def extract_accuracy(results: Dict) -> Optional[float]:
    """Extract eval_accuracy from results dictionary."""
    if 'eval_accuracy' in results:
        return float(results['eval_accuracy'])
    elif 'accuracy' in results:
        return float(results['accuracy'])
    else:
        # Look for other accuracy-like metrics
        for key in results.keys():
            if 'accuracy' in key.lower():
                return float(results[key])
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect and average GLUE evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python collect_glue_results.py cola sst2 mnli
    python collect_glue_results.py --all
    python collect_glue_results.py boolq cola mnli mrpc qnli qqp rte sst2 wsc
        """
    )
    parser.add_argument(
        'tasks', 
        nargs='*', 
        help='GLUE tasks to process (e.g., cola sst2 mnli)'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Process all available tasks'
    )
    parser.add_argument(
        '--glue-dir', 
        type=Path,
        help='Path to fine-tuned-ltg-bert-glue directory (auto-detected if not provided)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed results for each task'
    )
    
    args = parser.parse_args()
    
    # Find GLUE results directory
    if args.glue_dir:
        glue_dir = args.glue_dir
    else:
        glue_dir = find_glue_results_dir()
    
    if not glue_dir or not glue_dir.exists():
        print("❌ Error: Could not find fine-tuned-ltg-bert-glue directory")
        print("   Please specify the path with --glue-dir")
        sys.exit(1)
    
    print(f"📁 Using GLUE results directory: {glue_dir}")
    
    # Get available tasks
    available_tasks = get_available_tasks(glue_dir)
    
    if not available_tasks:
        print("❌ Error: No tasks with eval_results.json found")
        sys.exit(1)
    
    print(f"📋 Available tasks: {', '.join(available_tasks)}")
    
    # Determine which tasks to process
    if args.all:
        tasks_to_process = available_tasks
    elif args.tasks:
        tasks_to_process = args.tasks
        # Validate that requested tasks exist
        missing_tasks = [task for task in tasks_to_process if task not in available_tasks]
        if missing_tasks:
            print(f"⚠️  Warning: These tasks were not found: {', '.join(missing_tasks)}")
            tasks_to_process = [task for task in tasks_to_process if task in available_tasks]
    else:
        print("❌ Error: No tasks specified. Use task names or --all")
        print(f"   Available tasks: {', '.join(available_tasks)}")
        sys.exit(1)
    
    if not tasks_to_process:
        print("❌ Error: No valid tasks to process")
        sys.exit(1)
    
    print(f"🎯 Processing tasks: {', '.join(tasks_to_process)}")
    print("=" * 60)
    
    # Collect results
    task_results = {}
    accuracies = []
    
    for task in tasks_to_process:
        task_dir = glue_dir / task
        print(f"📊 Processing {task}...")
        
        # Load results
        results = load_eval_results(task_dir)
        if results is None:
            print(f"   ❌ Failed to load results for {task}")
            continue
        
        # Extract accuracy
        accuracy = extract_accuracy(results)
        if accuracy is None:
            print(f"   ❌ No accuracy metric found for {task}")
            if args.verbose:
                print(f"      Available metrics: {list(results.keys())}")
            continue
        
        task_results[task] = {
            'accuracy': accuracy,
            'all_metrics': results
        }
        accuracies.append(accuracy)
        
        print(f"   ✅ eval_accuracy: {accuracy:.4f}")
        
        if args.verbose:
            # Show other metrics too
            other_metrics = {k: v for k, v in results.items() 
                            if k != 'eval_accuracy' and isinstance(v, (int, float))}
            if other_metrics:
                print(f"      Other metrics: {other_metrics}")
    
    # Calculate and display average
    print("=" * 60)
    
    if not accuracies:
        print("❌ No valid accuracy results found")
        sys.exit(1)
    
    average_accuracy = sum(accuracies) / len(accuracies)
    
    print("📈 RESULTS SUMMARY")
    print("-" * 30)
    print(f"Tasks processed: {len(accuracies)}")
    print(f"Average eval_accuracy: {average_accuracy:.4f}")
    print(f"Min accuracy: {min(accuracies):.4f}")
    print(f"Max accuracy: {max(accuracies):.4f}")
    
    if args.verbose:
        print("\n📋 Individual Results:")
        for task, result in task_results.items():
            print(f"  {task}: {result['accuracy']:.4f}")
    
    print("=" * 60)
    print(f"🎯 Final Average Accuracy: {average_accuracy:.4f}")


if __name__ == "__main__":
    main()