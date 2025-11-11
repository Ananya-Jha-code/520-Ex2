from datasets import load_dataset
import sys

def print_humaneval_test(problem_id: str):
    """
    Prints the test code for a given HumanEval problem ID.
    Example: python print_test.py HumanEval/13
    """
    print(f"📥 Loading dataset... (this happens only once, it's cached)")
    ds = load_dataset("evalplus/humanevalplus", split="test")

    # Find the problem
    sample = next((s for s in ds if s["task_id"] == problem_id), None)
    if not sample:
        raise ValueError(f"❌ Problem ID '{problem_id}' not found.")

    runner_code = f"""
from src.solution import {sample["entry_point"]}
def test_generated():
    check({sample["entry_point"]})
"""
    with open(f"tests/{problem_id.replace("/", "_")}.py", "w") as f:
        f.write(sample["test"].replace("candidate", sample["entry_point"]))
        f.write(runner_code)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_test.py HumanEval/<ID>")
        sys.exit(1)

    problem_id = sys.argv[1]
    print_humaneval_test(problem_id)
