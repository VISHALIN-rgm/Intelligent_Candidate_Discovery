#!/usr/bin/env python3
"""
app.py — Redrob Intelligent Candidate Ranker (Gradio UI for HuggingFace Spaces)

Thin UI layer on top of rank.py. Imports the scoring engine directly so
there is exactly one source of truth for ranking logic — no duplicated
code, no risk of the UI and CLI drifting out of sync.

Flow:
  1. User uploads candidates.jsonl(.gz) and a job description (.md/.txt/.docx)
  2. Pipeline runs in-process, streaming live progress logs to the UI
  3. Ranked top-K table renders with score breakdown
  4. submission.csv is offered as a download
"""

import csv
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

import rank as R  # the single source of truth for all scoring logic

# Load the bi-encoder ONCE at startup, not on every pipeline run.
# Re-loading SentenceTransformer from disk on every "Run Ranking" click
# was the single biggest source of slowness.
print(f"Loading bi-encoder model ({R.BIENCODER_MODEL})... this happens once at startup.")
_BI_MODEL = R.SentenceTransformer(R.BIENCODER_MODEL)
print("Model loaded. Starting UI...")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner — mirrors rank.py main(), adapted to yield progress for Gradio
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(candidates_path: str, jd_path: str, prefilter: int, top_k: int):
    """
    Generator that yields (log_text, table_rows_or_None, csv_path_or_None)
    at each pipeline stage, exactly mirroring rank.py's main() logic.
    """
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)

    def emit(table=None, csv_path=None):
        return "\n".join(log_lines), table, csv_path

    t0 = time.time()

    # ── [1/5] Parse JD ──────────────────────────────────────────────────────
    log("[1/5] Parsing job description...")
    yield emit()
    try:
        jd_text = R.load_jd(jd_path, candidates_path)
    except SystemExit:
        log("❌ Failed to parse job description. Check the file format.")
        yield emit()
        return
    jd = R.parse_jd(jd_text)
    log(f"      YoE range     : {jd['yoe_min']}-{jd['yoe_max']} yrs")
    log(f"      Target cities : {jd['target_cities'] or 'not specified'}")
    log(f"      Target country: {jd['target_country'] or 'not specified'}")
    log(f"      JD skills     : {len(jd['jd_skills'])} detected")
    log(f"      Seniority     : {jd['seniority']}  |  Work mode: {jd['work_mode']}  |  Prefers product: {jd['prefers_product']}")
    yield emit()

    # ── [2/5] Load + clean candidates ───────────────────────────────────────
    log(f"\n[2/5] Loading candidates from {Path(candidates_path).name}...")
    yield emit()
    all_candidates = R.load_candidates(candidates_path)
    log(f"      Loaded {len(all_candidates):,} candidates")
    yield emit()

    clean = [c for c in all_candidates if not R.is_honeypot(c)]
    log(f"      {len(clean):,} clean  ({len(all_candidates) - len(clean):,} honeypots removed)")
    log(f"      Elapsed: {time.time() - t0:.1f}s")
    yield emit()

    # ── [3/5] Pre-filter + rule scoring ─────────────────────────────────────
    log("\n[3/5] Fast pre-filter + rule scoring...")
    yield emit()
    relevant = [c for c in clean if R.is_relevant(c)]
    log(f"      Fast filter: {len(clean):,} -> {len(relevant):,} relevant")
    yield emit()

    def cheap_score(c: dict) -> float:
        p = c["profile"]
        t = (p.get("current_title", "") + " " + p.get("headline", "")).lower()
        yoe = p.get("years_of_experience", 0)
        fit = (1.0 if R.TITLE_FIT_RE.search(t) else 0.5 if R.ML_ADJACENT_TITLE_RE.search(t) else 0.0)
        pen = 0.3 if R.TITLE_PENALTY_RE.search(t) else 0.0
        wr_pen = 0.4 if R.WRONG_DOMAIN_RE.search(t) else 0.0
        jr_pen = (0.2 if R.JUNIOR_TITLE_RE.search(t) and jd.get("seniority") in ("senior", "staff") else 0.0)
        in_rng = 1.0 if jd["yoe_min"] <= yoe <= jd["yoe_max"] else 0.0
        return fit - pen - wr_pen - jr_pen + in_rng * 0.5

    relevant.sort(key=cheap_score, reverse=True)
    stage_b_pool = relevant[:3000]
    log(f"      Stage-A filter: {len(relevant):,} -> {len(stage_b_pool):,} for full scoring")
    yield emit()

    log("      Rule scoring in progress...")
    yield emit()
    scored_pairs: list[tuple] = [(R.rule_score(c, jd), c) for c in stage_b_pool]
    scored_pairs.sort(key=lambda x: -x[0].total)

    if not scored_pairs:
        log("❌ No candidates survived filtering. Check your JD and candidate data.")
        yield emit()
        return

    pool = [c for rs, c in scored_pairs[:prefilter]]
    pool_scores = {c["candidate_id"]: rs for rs, c in scored_pairs}

    best_rs = scored_pairs[0][0].total
    worst_rs = scored_pairs[min(prefilter - 1, len(scored_pairs) - 1)][0].total
    log(f"      Top {prefilter} score range: {worst_rs:.3f} - {best_rs:.3f}")
    log(f"      Elapsed: {time.time() - t0:.1f}s")
    yield emit()

    # ── [4/5] Bi-encoder semantic search ────────────────────────────────────
    log(f"\n[4/5] Bi-encoder semantic search on top {len(pool)} ({R.BIENCODER_MODEL})...")
    yield emit()
    bi_model = _BI_MODEL  # already loaded at startup — no reload cost per run

    jd_emb = bi_model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    texts = [R.build_candidate_text(c) for c in pool]
    log(f"      Encoding {len(texts)} candidates...")
    yield emit()
    cand_embs = bi_model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    dim = cand_embs.shape[1]
    index = R.faiss.IndexFlatIP(dim)
    index.add(cand_embs.astype(np.float32))
    distances, indices = index.search(jd_emb.astype(np.float32), len(pool))

    sem_raw = distances[0]
    s_min, s_max = float(sem_raw.min()), float(sem_raw.max())
    sem_norm = (sem_raw - s_min) / (s_max - s_min + 1e-9)

    log(f"      Semantic range: {s_min:.3f} - {s_max:.3f}")
    log(f"      Elapsed: {time.time() - t0:.1f}s")
    yield emit()

    # ── [5/5] Hybrid scoring + validation + output ──────────────────────────
    log("\n[5/5] Hybrid scoring, validation and ranking...")
    yield emit()

    results: list[tuple] = []
    for rank_i, idx in enumerate(indices[0]):
        c = pool[idx]
        sem_n = float(sem_norm[rank_i])
        rs = pool_scores[c["candidate_id"]]
        hybrid = R.W_SEMANTIC * sem_n + R.W_RULES * rs.total
        results.append((hybrid, sem_n, rs, c))

    results.sort(key=R.tiebreak_key, reverse=True)

    yoe_min, yoe_max = jd["yoe_min"], jd["yoe_max"]

    def yoe_in_range(c: dict) -> bool:
        return yoe_min <= c["profile"].get("years_of_experience", 0) <= yoe_max

    in_range = [r for r in results if yoe_in_range(r[3])]
    out_range = [r for r in results if not yoe_in_range(r[3])]

    final = in_range[:top_k]
    if len(final) < top_k:
        pad = top_k - len(final)
        final += out_range[:pad]
        log(f"  Note: padded {pad} out-of-range candidates")

    final.sort(key=R.tiebreak_key, reverse=True)
    top_results = final[:top_k]

    if not top_results:
        log("❌ No candidates to rank. Check your filters and data.")
        yield emit()
        return

    top_scores = np.array([r[0] for r in top_results])
    strong_thresh = float(np.percentile(top_scores, 60.0))
    adjacent_thresh = float(np.percentile(top_scores, 30.0))
    log(f"  Tier thresholds: strong >{strong_thresh:.4f}, adjacent >{adjacent_thresh:.4f}, filler <={adjacent_thresh:.4f}")
    yield emit()

    warnings = R._validate_top20(top_results, strong_thresh)
    if warnings:
        log(f"  [VALIDATION] {len(warnings)} warning(s) — see Space logs for details")
    else:
        log("  [VALIDATION] Top-20 passed all checks.")
    yield emit()

    # Build CSV rows + UI table rows
    csv_rows: list[dict] = []
    table_rows: list[list] = []
    for rank_idx, (hybrid, sem_n, rs, c) in enumerate(top_results):
        tier_label = ("strong" if hybrid >= strong_thresh else "adjacent" if hybrid >= adjacent_thresh else "filler")
        reasoning = R.build_reasoning(c, jd, sem_n, rs, tier_label)

        csv_rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank_idx + 1,
            "score": round(hybrid, 6),
            "reasoning": reasoning,
        })

        p = c["profile"]
        table_rows.append([
            rank_idx + 1,
            c["candidate_id"],
            f"{hybrid:.4f}",
            tier_label,
            p.get("current_title", ""),
            p.get("years_of_experience", 0),
            f"{rs.skill:.2f}",
            f"{rs.career:.2f}",
            f"{rs.loc:.2f}",
            f"{rs.behav:.2f}",
            reasoning,
        ])

    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
    writer = csv.DictWriter(tmp_csv, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(csv_rows)
    tmp_csv.close()

    elapsed = time.time() - t0
    n_strong = sum(1 for r in table_rows if r[3] == "strong")
    n_adjacent = sum(1 for r in table_rows if r[3] == "adjacent")
    n_filler = sum(1 for r in table_rows if r[3] == "filler")
    log(
        f"\n✅ Ranking complete — {len(top_results)} candidates ranked in {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"   Strong: {n_strong}  |  Adjacent: {n_adjacent}  |  Filler: {n_filler}\n"
        f"   CSV ready for download ↓"
    )
    yield emit(table_rows, tmp_csv.name)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio glue
# ─────────────────────────────────────────────────────────────────────────────

def gradio_run(candidates_file, jd_file, prefilter, top_k):
    if candidates_file is None:
        yield "❌ Upload a candidates.jsonl file.", None, gr.update(visible=False)
        return
    if jd_file is None:
        yield "❌ Upload a job description file (.md, .txt, or .docx).", None, gr.update(visible=False)
        return

    log_accum, table_out, csv_path = "", None, None
    for log_text, tdata, csv_p in run_pipeline(
        candidates_path=candidates_file.name,
        jd_path=jd_file.name,
        prefilter=int(prefilter),
        top_k=int(top_k),
    ):
        log_accum = log_text
        if tdata is not None:
            table_out = tdata
        if csv_p is not None:
            csv_path = csv_p
        yield log_accum, table_out, gr.update(value=csv_path, visible=csv_path is not None)


CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#title { text-align: center; margin-bottom: 2px; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 22px; font-size: 0.95em; }
#log_box textarea {
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.85em;
}
"""

# NOTE (Gradio 6.0): `theme` and `css` are no longer accepted by gr.Blocks().
# They must be passed to demo.launch() instead. See launch() call at bottom.
with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("# Redrob Intelligent Candidate Ranker", elem_id="title")
    gr.Markdown(
        "Upload your candidate pool and job description — get a ranked, explainable shortlist.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            candidates_file = gr.File(
                label="Candidates file (.jsonl or .jsonl.gz)",
                file_types=[".jsonl", ".gz"],
            )
            jd_file = gr.File(
                label="Job Description (.md, .txt, .docx)",
                file_types=[".md", ".txt", ".docx"],
            )
            prefilter_slider = gr.Slider(
                minimum=50, maximum=500, value=100, step=50,
                label="Semantic pool size",
                info="Top-N by rule score fed into the bi-encoder (lower = faster)",
            )
            top_k_input = gr.Number(
                value=100, precision=0, minimum=1,
                label="Top-K candidates to rank",
            )
            run_btn = gr.Button("Run Ranking", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Live Log")
            log_output = gr.Textbox(
                label="", lines=18, max_lines=18,
                interactive=False, elem_id="log_box",
                placeholder="Logs will appear here once you click Run...",
            )

    gr.Markdown("### Ranked Candidates")
    table_output = gr.Dataframe(
        headers=["Rank", "CandID", "Score", "Tier", "Title", "YoE",
                 "Skill", "Career", "Loc", "Avail", "Reasoning"],
        datatype=["number", "str", "str", "str", "str", "number",
                  "str", "str", "str", "str", "str"],
        row_count=(10, "dynamic"),
        column_count=(11, "fixed"),
        wrap=True,
    )

    csv_download = gr.File(label="Download submission.csv", visible=False)

    run_btn.click(
        fn=gradio_run,
        inputs=[candidates_file, jd_file, prefilter_slider, top_k_input],
        outputs=[log_output, table_output, csv_download],
    )

    gr.Markdown(
        "---\n"
        "**Pipeline:** Honeypot filter → Title pre-filter → Rule scoring "
        "(skills 30% · career 35% · location 10% · availability 25%) → "
        "Bi-encoder FAISS semantic search → Hybrid blend (rules 70% / semantic 30%) "
        "→ Ranked CSV with reasoning.\n\n"
        f"Model: `{R.BIENCODER_MODEL}` · CPU-only · No network calls during ranking."
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS)