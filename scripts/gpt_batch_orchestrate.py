"""GPT-5.6-Luna batch orchestration: respect the 2M enqueued-token limit.

one_shot (in progress) -> few_shot (chunk 1500, sequential) -> cot (chunk
4000, sequential). Each chunk ~<2M prompt tokens; sequential queuing keeps
only one chunk enqueued at a time.
"""
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=True)
from src.generate import get_client  # noqa: E402

PY = sys.executable  # use the running interpreter (portable)
client = get_client("openai")


def poll(jids, timeout_min=240):
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        sts = []
        for jid in jids:
            b = client.batches.retrieve(jid)
            sts.append(b.status)
            if b.status == "failed":
                print(f"[gpt] {jid[:20]} FAILED: {b.errors}", flush=True)
                return False
        print(f"[gpt] statuses: {sts}", flush=True)
        if all(s in ("completed", "succeeded") for s in sts):
            return True
        time.sleep(120)
    return False


def run(args):
    r = subprocess.run([PY, "-m", "src.generate"] + args,
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-1500:], flush=True)
    return r.returncode == 0


# 1. wait for one_shot chunks
one_jids = [l.strip() for l in open(
    "data/outputs/batch_jobs/GPT-5.6-Luna_one_shot.jobid",
    encoding="utf-8").readlines() if l.strip()]
print("[gpt] waiting for one_shot...", flush=True)
if poll(one_jids):
    run(["--models", "gpt-5.6-luna", "--conditions", "one_shot",
         "--batch-action", "retrieve"])

# 2. few_shot: chunk 1000, sequential
print("[gpt] submitting few_shot (1000/chunk, sequential)...", flush=True)
if not run(["--models", "gpt-5.6-luna", "--conditions", "few_shot",
            "--batch-action", "submit", "--batch-chunk", "1000",
            "--batch-sequential"]):
    print("[gpt] few_shot submit failed; aborting (cot not submitted)",
          flush=True)
    sys.exit(1)
else:
    few_jids = [l.strip() for l in open(
        "data/outputs/batch_jobs/GPT-5.6-Luna_few_shot.jobid",
        encoding="utf-8").readlines() if l.strip()]
    print(f"[gpt] few_shot {len(few_jids)} chunks; waiting...", flush=True)
    if poll(few_jids, timeout_min=480):
        run(["--models", "gpt-5.6-luna", "--conditions", "few_shot",
             "--batch-action", "retrieve"])

# 3. cot: chunk 2000, sequential
print("[gpt] submitting cot (2000/chunk, sequential)...", flush=True)
if not run(["--models", "gpt-5.6-luna", "--conditions", "cot",
            "--batch-action", "submit", "--batch-chunk", "2000",
            "--batch-sequential"]):
    print("[gpt] cot submit failed", flush=True)
else:
    cot_jids = [l.strip() for l in open(
        "data/outputs/batch_jobs/GPT-5.6-Luna_cot.jobid",
        encoding="utf-8").readlines() if l.strip()]
    print(f"[gpt] cot {len(cot_jids)} chunks; waiting...", flush=True)
    if poll(cot_jids, timeout_min=480):
        run(["--models", "gpt-5.6-luna", "--conditions", "cot",
             "--batch-action", "retrieve"])

print("[gpt] ALL DONE", flush=True)
