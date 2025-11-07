from pathlib import Path
import os, subprocess, sys

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "source_documents"

files = [*DATA.glob("*.md"), *DATA.glob("*.txt")]
if not files:
    print("No .md/.txt found.")
    sys.exit(0)

env = os.environ.copy()
env.setdefault("OPENBLAS_NUM_THREADS", "1")
env.setdefault("OMP_NUM_THREADS", "1")

for f in files:
    print(f"\n🚀 Processing {f.name}")
    cmd = ["python", "-m", "rag_chatbot.cli", "ingest", str(f), "--batch-size", "4"]
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ {f.name} failed ({e.returncode}) — continuing.")
