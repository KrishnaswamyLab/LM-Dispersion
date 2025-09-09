#!/usr/bin/env python3
"""
TODO need to review and test this
Script to evaluate saved checkpoints on additional lm-eval tasks.
This script finds all saved checkpoints and evaluates them on the new tasks
that were added after training was completed.
"""

import os
import json
import argparse
import tempfile
import glob
from pathlib import Path
from lm_eval import simple_evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM


def log(s, filepath=None, to_console=True):
    """Log a string to either file or console"""
    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
                o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')


def find_all_checkpoints(results_dir):
    """Find all checkpoint directories in the results directory"""
    checkpoints = []
    
    # Pattern to match checkpoint directories
    patterns = [
        "**/eval_ckpt_*_step*",  # Interval and end checkpoints
        "**/eval_ckpt_begin_step*",  # Begin checkpoints
    ]
    
    for pattern in patterns:
        found = glob.glob(os.path.join(results_dir, pattern), recursive=True)
        checkpoints.extend(found)
    
    # Also include final model directories (those with model.safetensors directly)
    final_models = []
    for root, dirs, files in os.walk(results_dir):
        if 'model.safetensors' in files and 'config.json' in files:
            # Check if this is not already a checkpoint directory
            if not any(ckpt_part in os.path.basename(root) for ckpt_part in ['eval_ckpt_', 'step']):
                final_models.append(root)
    
    checkpoints.extend(final_models)
    
    return sorted(list(set(checkpoints)))


def extract_checkpoint_info(checkpoint_path):
    """Extract information about the checkpoint from its path"""
    path_parts = checkpoint_path.split('/')
    
    # Find the run directory (contains the hyperparameters)
    run_dir = None
    for i, part in enumerate(path_parts):
        if part.startswith('Llama-3.2-1B_'):
            run_dir = part
            break
    
    # Extract step information
    step_info = "final"
    if 'step' in os.path.basename(checkpoint_path):
        step_part = os.path.basename(checkpoint_path)
        if 'step' in step_part:
            step_info = step_part.split('step')[-1]
    
    # Extract stage information
    stage_info = "final"
    if 'begin' in checkpoint_path:
        stage_info = "begin"
    elif 'interval' in checkpoint_path:
        stage_info = "interval"
    elif 'end' in checkpoint_path:
        stage_info = "end"
    
    return {
        'run_dir': run_dir,
        'step': step_info,
        'stage': stage_info,
        'full_path': checkpoint_path
    }


def evaluate_checkpoint(checkpoint_path, tasks, num_fewshot=1, max_eval_samples=200, 
                       device="auto", log_path=None):
    """Evaluate a single checkpoint on the specified tasks"""
    
    log(f"Evaluating checkpoint: {checkpoint_path}", filepath=log_path)
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        
        # Determine precision
        if hasattr(model.config, 'torch_dtype'):
            if model.config.torch_dtype == 'bfloat16':
                dtype = "bfloat16"
            elif model.config.torch_dtype == 'float16':
                dtype = "float16"
            else:
                dtype = "float32"
        else:
            dtype = "float32"
        
        # Use a temporary directory for lm-eval
        with tempfile.TemporaryDirectory() as tmp:
            # Save model to temp directory for lm-eval
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            
            model_args = f"pretrained={tmp},dtype={dtype}"
            
            # Run evaluation
            results = simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size="auto",
                device=device,
                limit=max_eval_samples,
                log_samples=False,
            )
            
            return results
            
    except Exception as e:
        log(f"Error evaluating checkpoint {checkpoint_path}: {e}", filepath=log_path)
        return None


def save_results(results, checkpoint_info, output_dir, log_path=None):
    """Save evaluation results to JSON file"""
    if results is None or "results" not in results:
        log(f"No results to save for {checkpoint_info['full_path']}", filepath=log_path)
        return None
    
    # Create output filename
    run_name = checkpoint_info['run_dir']
    stage = checkpoint_info['stage']
    step = checkpoint_info['step']
    
    filename = f"additional_tasks_{stage}_{step}.json"
    output_path = os.path.join(output_dir, run_name, filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({"results": results["results"]}, f, indent=2)
    
    log(f"Results saved to: {output_path}", filepath=log_path)
    
    # Log key metrics
    for task, metrics in results["results"].items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    log(f"  {task}.{metric_name}: {value:.4f}", filepath=log_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on additional tasks")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing the saved model results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save additional evaluation results (defaults to results_dir)")
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=["hellaswag", "arc_challenge", "winogrande", "piqa", "truthfulqa_mc2", "gsm8k"],
                       help="Tasks to evaluate on")
    parser.add_argument("--num_fewshot", type=int, default=1,
                       help="Number of few-shot examples")
    parser.add_argument("--max_eval_samples", type=int, default=200,
                       help="Maximum number of evaluation samples per task")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run evaluation on")
    parser.add_argument("--checkpoint_pattern", type=str, default=None,
                       help="Pattern to filter checkpoints (e.g., '*None*' for baseline runs)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Just list checkpoints without evaluating")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    log_path = os.path.join(args.output_dir, "additional_eval_log.txt")
    
    # Find all checkpoints
    log("Finding checkpoints...", filepath=log_path)
    checkpoints = find_all_checkpoints(args.results_dir)
    
    if args.checkpoint_pattern:
        checkpoints = [ckpt for ckpt in checkpoints if args.checkpoint_pattern in ckpt]
    
    log(f"Found {len(checkpoints)} checkpoints", filepath=log_path)
    
    if args.dry_run:
        log("Dry run - listing checkpoints:", filepath=log_path)
        for ckpt in checkpoints:
            info = extract_checkpoint_info(ckpt)
            log(f"  {info['run_dir']} | {info['stage']} | step {info['step']}", filepath=log_path)
        return
    
    # Evaluate each checkpoint
    log(f"Starting evaluation on tasks: {args.tasks}", filepath=log_path)
    
    for i, checkpoint_path in enumerate(checkpoints):
        checkpoint_info = extract_checkpoint_info(checkpoint_path)
        
        log(f"\n[{i+1}/{len(checkpoints)}] Processing: {checkpoint_info['run_dir']} | {checkpoint_info['stage']} | step {checkpoint_info['step']}", filepath=log_path)
        
        # Check if results already exist
        output_filename = f"additional_tasks_{checkpoint_info['stage']}_{checkpoint_info['step']}.json"
        output_path = os.path.join(args.output_dir, checkpoint_info['run_dir'], output_filename)
        
        if os.path.exists(output_path):
            log(f"Results already exist, skipping: {output_path}", filepath=log_path)
            continue
        
        # Evaluate checkpoint
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            max_eval_samples=args.max_eval_samples,
            device=args.device,
            log_path=log_path
        )
        
        # Save results
        if results:
            save_results(results, checkpoint_info, args.output_dir, log_path)
        
        log(f"Completed checkpoint {i+1}/{len(checkpoints)}", filepath=log_path)
    
    log("\nAll evaluations completed!", filepath=log_path)


if __name__ == "__main__":
    main()
