import os
import shutil
import subprocess
import gradio as gr

DATA_DIR = "data"

def run_ranker(job_file, candidate_file):

    if job_file is None or candidate_file is None:
        raise gr.Error("Please upload both files.")

    # Get actual file paths
    job_path = job_file if isinstance(job_file, str) else job_file.name
    candidate_path = candidate_file if isinstance(candidate_file, str) else candidate_file.name

    shutil.copy2(job_path, os.path.join(DATA_DIR, "job_description.md"))
    shutil.copy2(candidate_path, os.path.join(DATA_DIR, "candidates.jsonl"))

    result = subprocess.run(
        ["python", "rank.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise gr.Error(result.stderr)

    output = os.path.join(DATA_DIR, "submission.csv")

    if not os.path.exists(output):
        raise gr.Error("submission.csv not generated.")

    return output