import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
ROOT = Path("data/generated-code")
TESTS_DIR = Path("tests")
OUT_DIR = Path("exercise2")
OUT_DIR.mkdir(exist_ok=True)

families = ["Gemini 2.5 Flah", "GPT-4o-mini"]
strategies = ["SCoT", "CoT"]

for task_dir in tqdm(sorted(ROOT.glob("HumanEval_*")), desc="Tasks"):
    task_name = task_dir.name
    test_src = TESTS_DIR / f"{task_name}.py"

    if not test_src.exists():
        print(f"⚠️ No test file for {task_name}, skipping.")
        continue

    for family in families:
        for strat in strategies:
            k0_path = task_dir / family / strat / "k2.py"
            if not k0_path.exists():
                continue

            # Create new structure
            combo_dir = OUT_DIR / task_name / f"{family}_{strat}_k0"
            src_dir = combo_dir / "src"
            tests_dir = combo_dir / "tests"
            src_dir.mkdir(parents=True, exist_ok=True)
            tests_dir.mkdir(exist_ok=True)

            # Copy code
            shutil.copy2(k0_path, src_dir / "solution.py")

            # Copy the matching test file
            shutil.copy2(test_src, tests_dir / "test_solution.py")

print(f"\n✅ Segregation complete.")
print(f"📁 Output directory: {OUT_DIR.resolve()}")
