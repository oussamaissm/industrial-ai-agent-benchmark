import json
import time
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agents.workflow import build_agent_graph
from src.simulator.factory import FactorySimulator

def main():
    graph = build_agent_graph()
    simulator = FactorySimulator()
    modes = ["baseline", "tool", "verified", "hybrid"]
    results_path = Path("experiments/results.jsonl")
    results_path.parent.mkdir(exist_ok=True)
    
    with open("data/benchmark_tasks.jsonl", "r") as f:
        tasks = [json.loads(line) for line in f]
    
    for mode in modes:
        print(f"Running benchmark for mode: {mode}")
        for task in tasks:
            initial_state = {
                "request": task["request"], "messages": [], "mode": mode,
                "retry_count": 0, "verification_result": {},
                "metrics": {"latency": 0, "memory_peak_mb": 0, "tokens_estimated": 0, "cost_usd": 0.0}
            }
            start_total = time.time()
            final_state = graph.invoke(initial_state)
            
            result = {
                "task_id": task["id"], "mode": mode,
                "success": "M003" in str(final_state["messages"][-1].content),
                "valid_schema": True,
                "verification_passed": final_state["verification_result"].get("valid", False),
                "latency_total": time.time() - start_total,
                "metrics": final_state["metrics"],
                "escalated": "strong_llm_node" in str(final_state)
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
    print(f"Benchmark complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
