"""
Comprehensive Model Evaluation Script for Wave-Density Language Model
Generates all metrics and data needed for paper publication.

Usage:
    python evaluate_model.py --checkpoint wda-130m-mom.pt
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any
import csv

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset

from wave_dencity import WaveCharLM, generate_text, sample_batch, build_streaming_dataset, build_streaming_ultrachat_dataset


class ModelEvaluator:
    """Comprehensive evaluation suite for Wave-Density models."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.results = {}
        
        # Load checkpoint and model
        print(f"Loading checkpoint: {checkpoint_path}")
        self.ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Load tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Extract model config
        model_cfg = self.ckpt.get('model_cfg', {})
        
        # Build model
        self.model = WaveCharLM(
            vocab_size=len(self.tokenizer),
            seq_len=256,
            embed_dim=model_cfg.get("embed_dim", 768),
            num_layers=model_cfg.get("num_layers", 8),
            num_heads=model_cfg.get("num_heads", 4),
            num_masks=model_cfg.get("num_masks", 16),
            num_waves_per_mask=model_cfg.get("num_waves_per_mask", 8),
            topk_masks=model_cfg.get("topk_masks", 8),
            attn_alpha=model_cfg.get("attn_alpha", 3.0),
            content_rank=model_cfg.get("content_rank", 8),
            content_mix=model_cfg.get("content_mix", 0.15),
            learned_content=model_cfg.get("learned_content", True),
            use_sin_waves=model_cfg.get("use_sin_waves", True),
            ffn_mult=model_cfg.get("ffn_mult", 4),
            tie_embeddings=bool(model_cfg.get("tie_embeddings", False)),
        ).to(device)
        
        # Handle torch.compile checkpoints (remove _orig_mod. prefix)
        state_dict = self.ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # Checkpoints may include extra buffers/params from earlier iterations
        # (e.g. registered buffers or output head bias). For evaluation we can
        # safely ignore unexpected keys.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Note: checkpoint load strict=False (missing={len(missing)}, unexpected={len(unexpected)})")
        self.model.eval()
        
        # Extract training metadata
        self.step = self.ckpt.get('step', 0)
        self.best_val_loss = self.ckpt.get('best_val_loss', float('inf'))
        self.model_cfg = model_cfg
        self.data_cfg = self.ckpt.get('data_cfg', {})
        self.train_cfg = self.ckpt.get('train_cfg', {})
        
        print(f"✓ Loaded model at step {self.step}")
        print(f"✓ Best validation loss: {self.best_val_loss:.4f}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Compute model statistics (params, memory, architecture)."""
        print("\n" + "="*60)
        print("📊 Computing Model Statistics")
        print("="*60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Wave-specific parameters
        wave_params = 0
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['freqs', 'amps', 'phases']):
                wave_params += param.numel()
        
        # Memory footprint (approximate)
        param_memory = total_params * 4 / 1e9  # FP32 in GB
        
        stats = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "wave_parameters": wave_params,
            "wave_param_percentage": 100 * wave_params / total_params,
            "total_parameters_millions": total_params / 1e6,
            "parameter_memory_gb_fp32": param_memory,
            "parameter_memory_gb_fp16": param_memory / 2,
            "architecture": {
                "embed_dim": self.model.embed_dim,
                "num_layers": len(self.model.blocks),
                "num_heads": self.model_cfg.get("num_heads", 4),
                "seq_len": self.model.seq_len,
                "vocab_size": self.model.vocab_size,
                "num_masks": self.model_cfg.get("num_masks", 16),
                "num_waves_per_mask": self.model_cfg.get("num_waves_per_mask", 8),
                "topk_masks": self.model_cfg.get("topk_masks", 8),
                "content_mix": self.model_cfg.get("content_mix", 0.15),
            },
            "training_step": self.step,
            "best_validation_loss": self.best_val_loss,
        }
        
        print(f"Total Parameters: {stats['total_parameters_millions']:.2f}M")
        print(f"Wave Parameters: {stats['wave_parameters']:,} ({stats['wave_param_percentage']:.2f}%)")
        print(f"Memory (FP32): {stats['parameter_memory_gb_fp32']:.3f} GB")
        print(f"Memory (FP16): {stats['parameter_memory_gb_fp16']:.3f} GB")
        
        self.results['model_stats'] = stats
        return stats
    
    def benchmark_speed(self, batch_size: int = 32, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark inference speed and throughput."""
        print("\n" + "="*60)
        print("⚡ Benchmarking Inference Speed")
        print("="*60)
        
        seq_len = self.model.seq_len
        dummy_input = torch.randint(0, self.model.vocab_size, (batch_size, seq_len), device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        
        # Benchmark
        timings = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                timings.append(end - start)
        
        avg_time = sum(timings) / len(timings)
        std_time = (sum((t - avg_time) ** 2 for t in timings) / len(timings)) ** 0.5
        tokens_per_sec = (batch_size * seq_len) / avg_time
        
        # Generation speed (single sample)
        gen_tokens = 100
        dummy_input_single = torch.randint(0, self.model.vocab_size, (1, seq_len), device=self.device)
        
        gen_start = time.perf_counter()
        with torch.no_grad():
            idx = dummy_input_single.clone()
            for _ in range(gen_tokens):
                logits = self.model(idx)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat([idx[:, 1:], next_token], dim=1)
        
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        gen_end = time.perf_counter()
        
        gen_time = gen_end - gen_start
        gen_tokens_per_sec = gen_tokens / gen_time
        
        benchmark = {
            "batch_inference_ms": avg_time * 1000,
            "batch_inference_std_ms": std_time * 1000,
            "throughput_tokens_per_sec": tokens_per_sec,
            "throughput_million_tokens_per_sec": tokens_per_sec / 1e6,
            "generation_tokens_per_sec": gen_tokens_per_sec,
            "generation_latency_ms_per_token": (gen_time / gen_tokens) * 1000,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "device": self.device,
        }
        
        print(f"Batch Inference: {benchmark['batch_inference_ms']:.2f} ± {benchmark['batch_inference_std_ms']:.2f} ms")
        print(f"Throughput: {benchmark['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"Generation: {benchmark['generation_tokens_per_sec']:.1f} tokens/sec")
        print(f"Latency: {benchmark['generation_latency_ms_per_token']:.1f} ms/token")
        
        self.results['benchmark'] = benchmark
        return benchmark
    
    def evaluate_perplexity(self, dataset_name: str = "c4", num_samples: int = 1000, batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate perplexity on validation dataset."""
        print("\n" + "="*60)
        print(f"📈 Evaluating Perplexity on {dataset_name}")
        print("="*60)
        
        seq_len = self.model.seq_len
        
        # Load validation data
        if dataset_name == "c4":
            stream = build_streaming_dataset(self.tokenizer, seq_len=seq_len)
        elif dataset_name == "ultrachat":
            stream = build_streaming_ultrachat_dataset(
                self.tokenizer,
                split="test_sft",
                dataset_name="HuggingFaceH4/ultrachat_200k",
                assistant_only_loss=True,
                include_assistant_prefix_in_loss=False,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        losses = []
        total_tokens = 0
        
        num_batches = num_samples // batch_size
        assistant_only = (dataset_name == "ultrachat")
        
        with torch.no_grad():
            for i in range(num_batches):
                x, y = sample_batch(stream, batch_size, seq_len, self.device, assistant_only_loss=assistant_only)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                losses.append(loss.item())
                total_tokens += batch_size * seq_len
                
                if (i + 1) % 10 == 0:
                    print(f"  Batch {i+1}/{num_batches}: loss={loss.item():.4f}")
        
        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(avg_loss, 10.0))  # Cap for stability
        
        results = {
            "dataset": dataset_name,
            "num_samples": num_samples,
            "num_batches": num_batches,
            "total_tokens": total_tokens,
            "average_loss": avg_loss,
            "perplexity": perplexity,
            "bits_per_byte": avg_loss / math.log(2),
        }
        
        print(f"Average Loss: {results['average_loss']:.4f}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Bits per Byte: {results['bits_per_byte']:.4f}")
        
        self.results[f'perplexity_{dataset_name}'] = results
        return results
    
    def generate_samples(self, prompts: List[str] = None, num_tokens: int = 150, temperature: float = 0.8) -> Dict[str, Any]:
        """Generate text samples for qualitative evaluation."""
        print("\n" + "="*60)
        print("✍️  Generating Text Samples")
        print("="*60)
        
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "In recent studies, scientists discovered",
                "The most important aspect of machine learning",
                "Climate change poses significant challenges because",
                "The human brain works by",
            ]
        
        samples = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: {prompt}")
            print("-" * 60)
            
            generated = generate_text(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_tokens,
                temp=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                device=self.device,
            )
            
            full_text = prompt + generated
            print(full_text[:300] + "..." if len(full_text) > 300 else full_text)
            
            samples.append({
                "prompt": prompt,
                "generated": generated,
                "full_text": full_text,
                "length": len(self.tokenizer.encode(generated)),
            })
        
        self.results['generation_samples'] = samples
        return samples
    
    def analyze_wave_parameters(self) -> Dict[str, Any]:
        """Analyze learned wave parameters statistics."""
        print("\n" + "="*60)
        print("🌊 Analyzing Wave Parameters")
        print("="*60)
        
        wave_stats = {
            "per_layer": [],
            "global": {
                "freq_mean": 0.0,
                "freq_std": 0.0,
                "amp_mean": 0.0,
                "amp_std": 0.0,
                "phase_mean": 0.0,
                "phase_std": 0.0,
            }
        }
        
        all_freqs = []
        all_amps = []
        all_phases = []
        
        for layer_idx, block in enumerate(self.model.blocks):
            layer_freqs = []
            layer_amps = []
            layer_phases = []
            
            for head_idx in range(len(block.freqs)):
                freqs = block.freqs[head_idx].detach().cpu()
                amps = block.amps[head_idx].detach().cpu()
                phases = block.phases[head_idx].detach().cpu()
                
                layer_freqs.append(freqs)
                layer_amps.append(amps)
                layer_phases.append(phases)
                
                all_freqs.append(freqs)
                all_amps.append(amps)
                all_phases.append(phases)
            
            layer_freqs = torch.cat([f.flatten() for f in layer_freqs])
            layer_amps = torch.cat([a.flatten() for a in layer_amps])
            layer_phases = torch.cat([p.flatten() for p in layer_phases])
            
            wave_stats["per_layer"].append({
                "layer": layer_idx,
                "freq_mean": float(layer_freqs.mean()),
                "freq_std": float(layer_freqs.std()),
                "amp_mean": float(layer_amps.mean()),
                "amp_std": float(layer_amps.std()),
                "phase_mean": float(layer_phases.mean()),
                "phase_std": float(layer_phases.std()),
            })
        
        # Global statistics
        all_freqs = torch.cat([f.flatten() for f in all_freqs])
        all_amps = torch.cat([a.flatten() for a in all_amps])
        all_phases = torch.cat([p.flatten() for p in all_phases])
        
        wave_stats["global"]["freq_mean"] = float(all_freqs.mean())
        wave_stats["global"]["freq_std"] = float(all_freqs.std())
        wave_stats["global"]["amp_mean"] = float(all_amps.mean())
        wave_stats["global"]["amp_std"] = float(all_amps.std())
        wave_stats["global"]["phase_mean"] = float(all_phases.mean())
        wave_stats["global"]["phase_std"] = float(all_phases.std())
        
        print(f"Global Frequency: {wave_stats['global']['freq_mean']:.3f} ± {wave_stats['global']['freq_std']:.3f}")
        print(f"Global Amplitude: {wave_stats['global']['amp_mean']:.3f} ± {wave_stats['global']['amp_std']:.3f}")
        print(f"Global Phase: {wave_stats['global']['phase_mean']:.3f} ± {wave_stats['global']['phase_std']:.3f}")
        
        self.results['wave_analysis'] = wave_stats
        return wave_stats
    
    def run_full_evaluation(self, perplexity_samples: int = 1000, speed_iterations: int = 100):
        """Run all evaluation metrics.

        Args:
            perplexity_samples: number of streaming batches/samples used for perplexity estimates.
            speed_iterations: number of iterations used for the speed benchmark.
        """
        print("\n" + "="*60)
        print("🚀 RUNNING FULL EVALUATION SUITE")
        print("="*60)
        
        # 1. Model Statistics
        self.get_model_stats()
        
        # 2. Speed Benchmark
        self.benchmark_speed(batch_size=32, num_iterations=int(speed_iterations))
        
        # 3. Perplexity Evaluation
        self.evaluate_perplexity(dataset_name="c4", num_samples=int(perplexity_samples))
        
        # 3b. UltraChat Perplexity (if model was trained on it)
        if self.data_cfg.get('dataset') == 'ultrachat':
            ultrachat_samples = min(int(perplexity_samples), 500)
            self.evaluate_perplexity(dataset_name="ultrachat", num_samples=ultrachat_samples)
        
        # 4. Text Generation Samples
        self.generate_samples()
        
        # 5. Wave Parameter Analysis
        self.analyze_wave_parameters()
        
        return self.results
    
    def save_results(self, output_dir: str = "./eval_results"):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.checkpoint_path).stem
        
        # Save JSON
        json_path = output_path / f"{model_name}_eval_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved JSON results: {json_path}")
        
        # Save CSV summary
        csv_path = output_path / f"{model_name}_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            # Flatten key metrics
            if 'model_stats' in self.results:
                stats = self.results['model_stats']
                writer.writerow(['Total Parameters (M)', f"{stats['total_parameters_millions']:.2f}"])
                writer.writerow(['Wave Parameters (%)', f"{stats['wave_param_percentage']:.2f}"])
                writer.writerow(['Memory FP32 (GB)', f"{stats['parameter_memory_gb_fp32']:.3f}"])
            
            if 'benchmark' in self.results:
                bench = self.results['benchmark']
                writer.writerow(['Throughput (tokens/sec)', f"{bench['throughput_tokens_per_sec']:.0f}"])
                writer.writerow(['Generation (tokens/sec)', f"{bench['generation_tokens_per_sec']:.1f}"])
                writer.writerow(['Latency (ms/token)', f"{bench['generation_latency_ms_per_token']:.1f}"])
            
            if 'perplexity_c4' in self.results:
                ppl = self.results['perplexity_c4']
                writer.writerow(['Perplexity (C4)', f"{ppl['perplexity']:.2f}"])
                writer.writerow(['Loss (C4)', f"{ppl['average_loss']:.4f}"])
                writer.writerow(['Bits per Byte (C4)', f"{ppl['bits_per_byte']:.4f}"])
            
            if 'perplexity_ultrachat' in self.results:
                ppl = self.results['perplexity_ultrachat']
                writer.writerow(['Perplexity (UltraChat)', f"{ppl['perplexity']:.2f}"])
                writer.writerow(['Loss (UltraChat)', f"{ppl['average_loss']:.4f}"])
        
        print(f"✓ Saved CSV summary: {csv_path}")
        
        # Save markdown report
        md_path = output_path / f"{model_name}_report_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(f"# Wave-Density Model Evaluation Report\n\n")
            f.write(f"**Model:** `{model_name}`\n")
            f.write(f"**Checkpoint:** `{self.checkpoint_path}`\n")
            f.write(f"**Training Step:** {self.step}\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Architecture\n\n")
            if 'model_stats' in self.results:
                stats = self.results['model_stats']
                arch = stats['architecture']
                f.write(f"- **Total Parameters:** {stats['total_parameters_millions']:.2f}M\n")
                f.write(f"- **Wave Parameters:** {stats['wave_parameters']:,} ({stats['wave_param_percentage']:.2f}%)\n")
                f.write(f"- **Embedding Dimension:** {arch['embed_dim']}\n")
                f.write(f"- **Layers:** {arch['num_layers']}\n")
                f.write(f"- **Heads:** {arch['num_heads']}\n")
                f.write(f"- **Sequence Length:** {arch['seq_len']}\n")
                f.write(f"- **Memory (FP16):** {stats['parameter_memory_gb_fp16']:.3f} GB\n\n")
            
            f.write("## Performance Metrics\n\n")
            if 'benchmark' in self.results:
                bench = self.results['benchmark']
                f.write(f"- **Throughput:** {bench['throughput_tokens_per_sec']:.0f} tokens/sec\n")
                f.write(f"- **Generation Speed:** {bench['generation_tokens_per_sec']:.1f} tokens/sec\n")
                f.write(f"- **Inference Latency:** {bench['batch_inference_ms']:.2f} ms (batch={bench['batch_size']})\n\n")
            
            f.write("## Language Modeling Performance\n\n")
            if 'perplexity_c4' in self.results:
                ppl = self.results['perplexity_c4']
                f.write(f"### C4 Dataset\n")
                f.write(f"- **Perplexity:** {ppl['perplexity']:.2f}\n")
                f.write(f"- **Loss:** {ppl['average_loss']:.4f}\n")
                f.write(f"- **Bits per Byte:** {ppl['bits_per_byte']:.4f}\n\n")
            
            if 'perplexity_ultrachat' in self.results:
                ppl = self.results['perplexity_ultrachat']
                f.write(f"### UltraChat Dataset\n")
                f.write(f"- **Perplexity:** {ppl['perplexity']:.2f}\n")
                f.write(f"- **Loss:** {ppl['average_loss']:.4f}\n\n")
            
            f.write("## Wave Parameter Analysis\n\n")
            if 'wave_analysis' in self.results:
                wave = self.results['wave_analysis']['global']
                f.write(f"- **Frequency:** {wave['freq_mean']:.3f} ± {wave['freq_std']:.3f}\n")
                f.write(f"- **Amplitude:** {wave['amp_mean']:.3f} ± {wave['amp_std']:.3f}\n")
                f.write(f"- **Phase:** {wave['phase_mean']:.3f} ± {wave['phase_std']:.3f}\n\n")
            
            f.write("## Generation Samples\n\n")
            if 'generation_samples' in self.results:
                for i, sample in enumerate(self.results['generation_samples'][:3]):
                    f.write(f"### Sample {i+1}\n\n")
                    f.write(f"**Prompt:** {sample['prompt']}\n\n")
                    f.write(f"**Generated:** {sample['generated'][:200]}...\n\n")
        
        print(f"✓ Saved Markdown report: {md_path}")
        print(f"\n{'='*60}")
        print("✅ EVALUATION COMPLETE")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wave-Density Language Model")
    parser.add_argument("--checkpoint", type=str, default="wda-130m-mom.pt",
                        help="Path to model checkpoint")
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to run evaluation on")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--perplexity-samples", type=int, default=1000,
                        help="Number of samples for perplexity evaluation")
    parser.add_argument("--speed-iterations", type=int, default=100,
                        help="Number of iterations for speed benchmark")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Wave-Density Model Evaluation Suite")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    evaluator = ModelEvaluator(args.checkpoint, device=args.device)
    results = evaluator.run_full_evaluation(
        perplexity_samples=args.perplexity_samples,
        speed_iterations=args.speed_iterations,
    )
    evaluator.save_results(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
