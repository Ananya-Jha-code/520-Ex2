import os
import sys
import shutil
import subprocess
from pathlib import Path
import textwrap

# Ensure current directory is importable (for pytest relative paths)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# === CONFIG ===
BASE_DIR = Path("ex2p2")  # From Part 2
TARGET_DIR = Path("ex2p3")

PROBLEMS = {
    # Problem 1: largest_divisor
    "HumanEval_24/GPT-4o-mini_CoT_k0/iteration_3": {
        "bug_desc": "Removed '+1' in sqrt divisor range (off-by-one bug).",
        "inject_marker": "for i in range(2 , int(n**0.5) + 1 )",
        "replacement": "for i in range(2 , int(n**0.5) ):  # BUG: off-by-one, skips sqrt divisor",
        "function_name": "largest_divisor",
    },
    # Problem 2: split_words
    "HumanEval_125/GPT-4o-mini_SCoT_k0/iteration_3": {
        "bug_desc": "Removed comma-handling 'try' block (missing branch bug).",
        "inject_marker": "except ValueError:",
        "replacement": "# BUG: removed comma-handling try block\n# except ValueError:",
        "function_name": "split_words",
    },
}


def safe_copytree(src, dst):
    """Copy directory tree while overwriting existing files."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def inject_bug(problem_dir: Path, marker: str, replacement: str):
    """Inject a bug into src/solution.py."""
    src_file = problem_dir / "src" / "solution.py"
    buggy_dir = problem_dir / "buggy_src"
    buggy_file = buggy_dir / "solution.py"

    os.makedirs(buggy_dir, exist_ok=True)
    (buggy_dir / "__init__.py").touch()  # ✅ make importable

    text = src_file.read_text()
    if marker not in text:
        print(f"⚠️  Marker not found in {src_file}")
        buggy_file.write_text(text)
        return

    buggy_text = text.replace(marker, replacement)
    buggy_file.write_text(buggy_text)
    print(f"💉 Injected bug in {buggy_file}")


def update_test_import(problem_dir: Path, func_name: str):
    """Modify test file import to point to buggy_src."""
    test_file = problem_dir / "tests" / "test_solution.py"
    text = test_file.read_text()

    new_text = text.replace(
        f"from src.solution import {func_name}",
        f"from buggy_src.solution import {func_name}"
    )

    buggy_tests = problem_dir / "tests_buggy"
    buggy_tests.mkdir(exist_ok=True)
    (buggy_tests / "__init__.py").touch()  # ✅ make importable

    buggy_test_file = buggy_tests / "test_solution.py"
    buggy_test_file.write_text(new_text)
    return buggy_test_file


def run_tests(problem_dir: Path):
    """Run pytest and capture output."""
    print(f"🧪 Running tests for: {problem_dir}")
    result = subprocess.run(
        ["pytest", "-q", "--tb=short", str(problem_dir / "tests_buggy")],
        capture_output=True,
        text=True
    )
    return result.stdout


def main():
    print("🧱 Setting up Part 3 Fault Detection Environment")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path, cfg in PROBLEMS.items():
        src_path = BASE_DIR / rel_path
        dst_path = TARGET_DIR / Path(rel_path).parent  # copy at problem level

        print(f"\n📂 Copying {rel_path} ...")
        safe_copytree(src_path, dst_path)

        # Inject bug
        inject_bug(dst_path, cfg["inject_marker"], cfg["replacement"])

        # Modify tests
        update_test_import(dst_path, cfg["function_name"])

        # Run tests
        out = run_tests(dst_path)
        out_file = dst_path / "buggy_test_output.txt"
        out_file.write_text(out)

        print(textwrap.dedent(f"""
        === Fault Detection Summary ===
        Problem: {rel_path}
        Injected Bug: {cfg["bug_desc"]}
        Output saved at: {out_file}
        -----------------------------
        {out[:600]}...
        """))

    print("\n✅ Done. Results in: exercise2_part3/<problem>/buggy_test_output.txt")


if __name__ == "__main__":
    main()
