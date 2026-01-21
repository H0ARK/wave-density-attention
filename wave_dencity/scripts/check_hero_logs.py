import json
import matplotlib.pyplot as plt
import os
import sys


def plot_logs(log_path):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return

    steps = []
    loss = []
    avg_ts = []
    gammas = {}
    lrs = {}

    with open(log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                steps.append(data["step"])
                loss.append(data["loss"])
                avg_ts.append(data.get("avg_ts", 0))

                for k, v in data.items():
                    if "gamma" in k:
                        if k not in gammas:
                            gammas[k] = []
                        # Smooth gammas slightly
                        val = v if not gammas[k] else (gammas[k][-1] * 0.9 + v * 0.1)
                        gammas[k].append(val)
                    if "_lr" in k:
                        if k not in lrs:
                            lrs[k] = []
                        lrs[k].append(v)
            except:
                continue

    fig, (ax1, ax2, ax3_new) = plt.subplots(3, 1, figsize=(12, 12))

    ax1.plot(steps, loss, label="Loss", color="red", alpha=0.3)
    # Simple smoothing for loss
    import numpy as np

    if len(loss) > 10:
        kernel = np.ones(10) / 10
        smoothed = np.convolve(loss, kernel, mode="valid")
        ax1.plot(steps[9:], smoothed, color="red", label="Loss (Smooth)")

    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax_ts = ax1.twinx()
    ax_ts.plot(steps, avg_ts, label="Avg Teacher Scale", color="blue", linestyle="--")
    ax_ts.set_ylabel("Teacher Scale")
    ax_ts.legend(loc="lower right")

    for k, v in gammas.items():
        layer_idx = k.split("_")[1]
        ax2.plot(steps, v, label=f"L{layer_idx}")
    ax2.set_ylabel("Gamma")
    ax2.set_title("Per-Layer Signal Migration (Gamma)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    for k, v in lrs.items():
        layer_idx = k.split("_")[1]
        ax3_new.plot(steps, v, label=f"LR L{layer_idx}")
    ax3_new.set_xlabel("Step")
    ax3_new.set_ylabel("Learning Rate")
    ax3_new.set_title("Per-Layer Optimizer LR")

    plt.tight_layout()
    plt.savefig("migration_monitor.png")
    print("Updated migration_monitor.png")


if __name__ == "__main__":
    log_file = "private\\checkpoints\\qwen05b_hero_sequential\\training_log.jsonl"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    plot_logs(log_file)
