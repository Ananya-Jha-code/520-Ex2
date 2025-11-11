import os
import subprocess
import csv
from pathlib import Path
import re

base_dir = Path("ex2")
output_csv = Path("Coverage Reports/coverage_summary.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)

results = []

for task_dir in base_dir.iterdir():
    if not task_dir.is_dir():
        continue

    for combo_dir in task_dir.iterdir():
        if not combo_dir.is_dir():
            continue
        combo = combo_dir.name  
        cwd = combo_dir
        print(f"\n\n▶ Running tests for {task_dir.name} / {combo}\n{'='*80}")

        env = os.environ.copy()
        env["PYTHONPATH"] = "./"

        cmd = [
            "pytest",
            "--cov=src",
            "--cov-branch",
            "-v",
        ]

        # Run pytest and capture+show output
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in proc.stdout:
            print(line)
            output_lines.append(line)
        proc.wait()
        output = "".join(output_lines)

        # Save raw pytest output
        with open(cwd / "pytest_output.log", "w") as logf:
            logf.write(output)

        # Parse test counts
        total_tests = passed_tests = 0
        m_col = re.search(r"collected\s+(\d+)", output)
        print(m_col)
        if m_col:
            total_tests = int(m_col.group(1))
        passed_tests = output.count("PASSED")

        # Parse coverage values
        line_cov = branch_cov = None
        # coverage table typically ends with TOTAL line
        m_line = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", output)
        if m_line:
            line_cov = float(m_line.group(1))

        m_branch = re.search(r"Branch\s+coverage:\s+(\d+)%", output)
        if m_branch:
            branch_cov = float(m_branch.group(1))

        # Add diagnostic note
        if total_tests == 0:
            note = "No tests discovered."
        elif passed_tests < total_tests:
            note = "Tests failed — logic error despite coverage."
        elif (branch_cov or 0) < 80:
            note = "Low branch coverage — missing else/edge conditions."
        else:
            note = "High coverage and all tests passed."

        results.append({
            "task": task_dir.name,
            "combo": combo,
            "tests_total": total_tests,
            "tests_passed": passed_tests,
            "line_coverage": line_cov,
            "branch_coverage": branch_cov,
            "notes": note,
        })

# Write results CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task", "combo", "tests_total", "tests_passed",
        "line_coverage", "branch_coverage", "notes"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Coverage summary saved to: {output_csv}")
