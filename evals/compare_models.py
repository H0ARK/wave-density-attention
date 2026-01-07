"""
Model Comparison Script - Compare multiple Wave-Density checkpoints

Usage:
    python compare_models.py --checkpoints *.pt --output comparison_results
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import csv

from evaluate_model import ModelEvaluator


def compare_checkpoints(checkpoint_paths: List[str], output_dir: str = "./comparison_results", device: str = "cpu"):
    """Compare multiple model checkpoints."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    print("="*80)
    print("WAVE-DENSITY MODEL COMPARISON")
    print("="*80)
    print(f"Comparing {len(checkpoint_paths)} checkpoints...")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n[{i+1}/{len(checkpoint_paths)}] Evaluating: {ckpt_path}")
        print("-"*80)
        
        try:
            evaluator = ModelEvaluator(ckpt_path, device=device)
            
            # Run lightweight evaluation
            stats = evaluator.get_model_stats()
            bench = evaluator.benchmark_speed(batch_size=16, num_iterations=30)
            
            # Evaluate on both C4 and UltraChat if applicable
            ppl_c4 = evaluator.evaluate_perplexity(dataset_name="c4", num_samples=300)
            ppl = ppl_c4  # Default to C4
            
            if evaluator.data_cfg.get('dataset') == 'ultrachat':
                ppl_ultrachat = evaluator.evaluate_perplexity(dataset_name="ultrachat", num_samples=200)
                ppl = ppl_ultrachat  # Use training dataset for main metric
            
            wave_analysis = evaluator.analyze_wave_parameters()
            
            result = {
                'checkpoint': Path(ckpt_path).name,
                'step': evaluator.step,
                'total_params_M': stats['total_parameters_millions'],
                'wave_params_pct': stats['wave_param_percentage'],
                'memory_fp16_gb': stats['parameter_memory_gb_fp16'],
                'throughput_tokens_sec': bench['throughput_tokens_per_sec'],
                'gen_tokens_sec': bench['generation_tokens_per_sec'],
                'latency_ms_per_token': bench['generation_latency_ms_per_token'],
                'perplexity': ppl['perplexity'],
                'loss': ppl['average_loss'],
                'bits_per_byte': ppl['bits_per_byte'],
                'wave_freq_mean': wave_analysis['global']['freq_mean'],
                'wave_amp_mean': wave_analysis['global']['amp_mean'],
                'embed_dim': stats['architecture']['embed_dim'],
                'num_layers': stats['architecture']['num_layers'],
                'num_heads': stats['architecture']['num_heads'],
            }
            
            all_results.append(result)
            print(f"✓ Completed: PPL={ppl['perplexity']:.2f}, Params={stats['total_parameters_millions']:.1f}M")
            
        except Exception as e:
            print(f"✗ Failed to evaluate {ckpt_path}: {e}")
            continue
    
    if not all_results:
        print("\n❌ No checkpoints successfully evaluated!")
        return
    
    # Sort by training step
    all_results = sorted(all_results, key=lambda x: x['step'])
    
    # Save JSON
    json_path = output_path / "comparison.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")
    
    # Save CSV
    csv_path = output_path / "comparison.csv"
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Create markdown table
    md_path = output_path / "comparison_table.md"
    with open(md_path, 'w') as f:
        f.write("# Wave-Density Model Comparison\n\n")
        
        # Main comparison table
        f.write("## Performance Comparison\n\n")
        f.write("| Checkpoint | Step | Params (M) | Perplexity | Loss | Throughput (tok/s) | Gen Speed (tok/s) |\n")
        f.write("|------------|------|------------|------------|------|-------------------|------------------|\n")
        
        for r in all_results:
            f.write(f"| {r['checkpoint'][:30]} | {r['step']:,} | {r['total_params_M']:.1f} | "
                   f"{r['perplexity']:.2f} | {r['loss']:.4f} | {r['throughput_tokens_sec']:.0f} | "
                   f"{r['gen_tokens_sec']:.1f} |\n")
        
        # Detailed architecture table
        f.write("\n## Architecture Details\n\n")
        f.write("| Checkpoint | Embed Dim | Layers | Heads | Wave Params % | Memory (FP16 GB) |\n")
        f.write("|------------|-----------|--------|-------|---------------|------------------|\n")
        
        for r in all_results:
            f.write(f"| {r['checkpoint'][:30]} | {r['embed_dim']} | {r['num_layers']} | "
                   f"{r['num_heads']} | {r['wave_params_pct']:.3f}% | {r['memory_fp16_gb']:.3f} |\n")
        
        # Wave parameter statistics
        f.write("\n## Wave Parameter Statistics\n\n")
        f.write("| Checkpoint | Frequency (mean) | Amplitude (mean) | Bits/Byte |\n")
        f.write("|------------|------------------|------------------|------------|\n")
        
        for r in all_results:
            f.write(f"| {r['checkpoint'][:30]} | {r['wave_freq_mean']:.3f} | "
                   f"{r['wave_amp_mean']:.3f} | {r['bits_per_byte']:.4f} |\n")
        
        # Best model summary
        f.write("\n## Best Models\n\n")
        
        best_ppl = min(all_results, key=lambda x: x['perplexity'])
        best_speed = max(all_results, key=lambda x: x['gen_tokens_sec'])
        best_throughput = max(all_results, key=lambda x: x['throughput_tokens_sec'])
        
        f.write(f"- **Best Perplexity:** `{best_ppl['checkpoint']}` (PPL: {best_ppl['perplexity']:.2f}, Step: {best_ppl['step']:,})\n")
        f.write(f"- **Fastest Generation:** `{best_speed['checkpoint']}` ({best_speed['gen_tokens_sec']:.1f} tokens/sec)\n")
        f.write(f"- **Highest Throughput:** `{best_throughput['checkpoint']}` ({best_throughput['throughput_tokens_sec']:.0f} tokens/sec)\n")
        
    print(f"✓ Saved Markdown: {md_path}")
    
    # Create LaTeX table for paper
    latex_path = output_path / "comparison_latex.tex"
    with open(latex_path, 'w') as f:
        f.write("% Wave-Density Model Comparison - LaTeX Table\n")
        f.write("% Copy this into your paper\n\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Wave-Density Language Model Comparison}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write("Model & Params (M) & PPL $\\downarrow$ & Loss & Speed (tok/s) & Memory (GB) \\\\\n")
        f.write("\\hline\n")
        
        for r in all_results:
            name = r['checkpoint'].replace('_', '\\_').replace('.pt', '')
            f.write(f"{name} & {r['total_params_M']:.1f} & {r['perplexity']:.2f} & "
                   f"{r['loss']:.4f} & {r['gen_tokens_sec']:.1f} & {r['memory_fp16_gb']:.2f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX: {latex_path}")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
    print(f"\nFiles saved to: {output_dir}/")
    print(f"  - comparison.json (full data)")
    print(f"  - comparison.csv (spreadsheet)")
    print(f"  - comparison_table.md (markdown tables)")
    print(f"  - comparison_latex.tex (ready for paper)")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare Wave-Density model checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to checkpoint files to compare")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--output", type=str, default="./comparison_results",
                        help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    compare_checkpoints(args.checkpoints, output_dir=args.output, device=args.device)


if __name__ == "__main__":
    main()
