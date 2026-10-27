import pandas as pd
from pathlib import Path

def calculate_metrics(results_path: str = "experiments/results.jsonl"):
    if not Path(results_path).exists():
        print("No results found. Run the benchmark first.")
        return
    df = pd.read_json(results_path, lines=True)
    metrics = df.groupby("mode").agg(
        task_success_rate=("success", "mean"),
        verification_pass_rate=("verification_passed", "mean"),
        median_latency=("latency_total", "median"),
        p95_latency=("latency_total", lambda x: x.quantile(0.95)),
        escalation_rate=("escalated", "mean")
    ).round(4)
    print("=== Benchmark Results ===")
    print(metrics)
    return metrics

if __name__ == "__main__":
    calculate_metrics()
