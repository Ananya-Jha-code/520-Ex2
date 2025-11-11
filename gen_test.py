import os
import shutil
import json
import subprocess

# ====================== CONFIG ======================

BASE_SOURCE = "ex2"
TARGET_BASE = "ex2p2"
SELECTED = [
    ("HumanEval_24", "GPT-4o-mini_CoT_k0"),
    ("HumanEval_125", "Gemini 2.5 Flash_SCoT_k0"),
]
ITERATIONS = 3

# ====================== HELPERS ======================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_file(path):
    return open(path).read() if os.path.exists(path) else ""

def write_file(path, content):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)

def run_coverage(iter_dir):
    env = os.environ.copy()
    env["PYTHONPATH"] = "./"
    proc = subprocess.run(
        ["pytest", "--cov=src", "--cov-branch", "-q"],
        cwd=iter_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout + "\n" + proc.stderr
    line_cov = 0.0
    branch_cov = 0.0
    for line in out.splitlines():
        if "TOTAL" in line and "%" in line:
            try:
                parts = line.split()
                line_cov = float(parts[-1].replace("%", ""))
            except Exception:
                pass
    return line_cov, branch_cov, out

# ====================== PROMPTS ======================

def build_prompt(code: str, prev_tests: str, iteration: int) -> str:
    prev_tests = prev_tests[-2000:] if len(prev_tests) > 2000 else prev_tests
    base = f"Code:\n{code}\n\nExisting tests:\n{prev_tests}\n\n"
    if iteration == 1:
        return base + (
            "You are a Python QA engineer. Add 5 pytest test cases that increase line and branch coverage "
            "by targeting edge cases, invalid inputs, and alternate branches. Return only valid Python code."
        )
    elif iteration == 2:
        return base + (
            "Add 5 more tests that explore hidden branches, exception cases, and non-trivial input combinations. "
            "Avoid duplicating existing test inputs. Return only valid Python code."
        )
    else:
        return base + (
            "Add 3 unique high-value test cases for rare conditions not yet tested. "
            "Focus on branches skipped so far. Return only valid pytest code."
        )

# ====================== MAIN PIPELINE ======================

def main():
    coverage_history = {}

    for task, combo in SELECTED:
        src_dir = os.path.join(BASE_SOURCE, task, combo)
        dest_root = os.path.join(TARGET_BASE, task, combo)
        ensure_dir(dest_root)
        print(f"\n🧱 Setting up {task}/{combo}")

        # ---------- iteration_0 baseline ----------
        iter0 = os.path.join(dest_root, "iteration_0")
        shutil.copytree(src_dir, iter0, dirs_exist_ok=True)
        line0, branch0, _ = run_coverage(iter0)
        coverage_history[f"{task}/{combo}"] = [{"iteration": 0, "line": line0, "branch": branch0}]
        prev_iter = iter0

        # ---------- Manual iterations ----------
        for i in range(1, ITERATIONS + 1):
            iter_dir = os.path.join(dest_root, f"iteration_{i}")
            ensure_dir(os.path.join(iter_dir, "src"))
            ensure_dir(os.path.join(iter_dir, "tests"))

            # Copy solution
            shutil.copy(
                os.path.join(prev_iter, "src", "solution.py"),
                os.path.join(iter_dir, "src", "solution.py"),
            )

            code = read_file(os.path.join(iter_dir, "src", "solution.py"))
            prev_tests = read_file(os.path.join(prev_iter, "tests", "test_solution.py"))
            if "def test_generated" in prev_tests:
                prev_tests = prev_tests.split("def test_generated")[0]

            prompt = build_prompt(code, prev_tests, i)

            print("\n" + "="*100)
            print(f"📜 Prompt for iteration {i} — {task}/{combo}:\n")
            print(prompt)
            print("="*100)
            print(f"✋ Paste this prompt into ChatGPT and save the generated code in:\n"
                  f"  → {os.path.join(iter_dir, 'llm_output.txt')}\n"
                  f"Then press ENTER to continue.")
            input("Press ENTER when ready...")

            llm_output = read_file(os.path.join(iter_dir, "llm_output.txt"))
            if not llm_output.strip():
                print("⚠️ No llm_output.txt found. Skipping this iteration.")
                continue

            # Extract code portion if wrapped in ```
            if "```" in llm_output:
                parts = llm_output.split("```")
                for i, part in enumerate(parts):
                    if part.strip().startswith("python"):
                        llm_output = parts[i + 1]
                        break

            merged_tests = f"from src.solution import *\nimport pytest\n\n{llm_output.strip()}\n"
            write_file(os.path.join(iter_dir, "tests", "test_solution.py"), merged_tests)

            # Run coverage
            line_cov, branch_cov, out = run_coverage(iter_dir)
            coverage_history[f"{task}/{combo}"].append({
                "iteration": i,
                "line": line_cov,
                "branch": branch_cov,
                "stdout": out[:4000],
            })
            print(f"✅ Iteration {i} complete — line={line_cov:.1f}%, branch={branch_cov:.1f}%\n")

            prev_iter = iter_dir

    write_file(os.path.join(TARGET_BASE, "manual_coverage_history.json"), json.dumps(coverage_history, indent=2))
    print("\n✅ Completed manual test generation for all problems!")

if __name__ == "__main__":
    main()
