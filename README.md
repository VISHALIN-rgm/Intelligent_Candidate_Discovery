# 🎯 Redrob Ranker

**Intelligent Candidate Discovery & Ranking — Redrob Hackathon Submission**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-00599C?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-BGE--small-FF6F00)](https://www.sbert.net/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-yellow)](https://huggingface.co/spaces/kvishalini/Redrob-Ranker)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A hybrid rule-based + semantic ranking pipeline that ranks candidates against the Redrob "Senior AI Engineer — Founding Team" job description, built to satisfy the hackathon's compute, format, and reasoning-quality constraints — CPU-only, ≤5 minutes, no network calls during ranking.

**🔗 Live sandbox:** https://huggingface.co/spaces/kvishalini/Redrob-Ranker

---

## 📋 Table of Contents

- [What this is](#-what-this-is)
- [Architecture](#-architecture)
- [Repository structure](#-repository-structure)
- [Setup](#-setup)
- [Reproduce the submission CSV](#-reproduce-the-submission-csv)
- [Run the sandbox UI locally](#-run-the-sandbox-ui-locally)
- [Scoring methodology](#-scoring-methodology)
- [Compute constraints compliance](#-compute-constraints-compliance)
- [Honeypot & trap handling](#-honeypot--trap-handling)
- [Known limitations](#-known-limitations)

---

## 🧠 What this is

The JD asks for a narrow profile: engineers who've shipped **production retrieval/ranking/recommendation systems** at product companies, not just engineers whose skills list contains AI keywords. The dataset is built to punish naive keyword matching — keyword stuffers, "Marketing Manager" titles with a stuffed skills list, plain-language candidates who never name a framework but clearly built the real thing, and ~80 honeypots with internally-impossible profiles.

So this pipeline is deliberately **not** a single embedding-similarity sort. It's a two-stage hybrid:

1. A **rule engine** that reads career history, responsibilities, and projects as text — not just the skills array — to infer production deployment, seniority progression, and product-vs-services company context.
2. A **bi-encoder semantic layer** (FAISS + sentence-transformers) that catches JD-relevant candidates the keyword rules miss, and vice versa.

The two are blended (`70% rules / 30% semantic`) precisely because keyword rules alone fall for the keyword-stuffing trap, and semantic similarity alone falls for "sounds related but isn't actually a fit."

---

## 🏗 Architecture

```
                         ┌─────────────────────────┐
                         │   job_description.md    │
                         │   (.md / .txt / .docx)  │
                         └────────────┬────────────┘
                                      │
                              parse_jd()
                       (YoE range, target cities/
                        country, JD skills, seniority,
                        work mode, "prefers product" flag)
                                      │
┌──────────────────────┐             │             ┌─────────────────────────┐
│  candidates.jsonl(.gz)│            │             │  Rule Engine             │
│  100,000 candidates   │            ▼             │  rule_score()            │
└──────────┬────────────┘  ┌──────────────────┐    │  ├─ skill_score   (30%) │
           │                │  Honeypot filter  │    │  ├─ career_score  (35%) │
           ▼                │  is_honeypot()    │    │  │   • production      │
┌──────────────────────┐    └────────┬──────────┘    │  │     inference       │
│  Stage A: cheap title │             │               │  │   • career          │
│  pre-filter (regex)   │◄────────────┘               │  │     progression     │
│  is_relevant()        │                              │  │   • product-co     │
└──────────┬─────────────┘                             │  │     vs services    │
           │ top 3,000 by cheap_score()                │  ├─ loc_score     (10%)│
           ▼                                            │  └─ behav_score  (25%)│
┌──────────────────────┐                                └────────────┬─────────┘
│  Stage B: full rule    │───────────────────────────────────────────┘
│  scoring on 3,000      │
│  rule_score()          │  → top N (default 100–300) by rule_score.total
└──────────┬──────────────┘
           ▼
┌───────────────────────────────────────────────────────────────────┐
│  Bi-Encoder Semantic Search                                         │
│  Model: BAAI/bge-small-en-v1.5 (CPU, offline/cached)                 │
│  JD text  ──encode──►  jd_embedding                                  │
│  Candidate cards ──encode──► candidate_embeddings                    │
│  FAISS IndexFlatIP  ──►  cosine-sim search  ──►  sem_score (0-1)     │
└──────────────────────────────────┬──────────────────────────────────┘
                                    ▼
                  hybrid = 0.30 × semantic + 0.70 × rules
                                    │
                          tie-break (within 0.002):
                 prod-retrieval evidence → JD-exact signal →
                 product-co ratio → GitHub → response rate → notice period
                                    │
                          Top-100 validation pass
                       (sort order, production-evidence
                        sanity check on top-20)
                                    │
                                    ▼
                    build_reasoning() → 1–2 sentence,
                  per-candidate, evidence-grounded justification
                                    │
                                    ▼
                          submission.csv
              candidate_id, rank, score, reasoning
```

### Why two stages of filtering before the bi-encoder?

Encoding and FAISS-searching 100,000 candidates on CPU within a 5-minute budget isn't realistic alongside everything else the pipeline does. So:

- **Stage A** (regex title/keyword pre-filter): 100,000 → ~thousands, near-zero cost.
- **Stage B** (full rule scoring, still cheap — no model inference): narrows to the top 3,000, then the top `--prefilter` (default 100–300) by rule score.
- **Bi-encoder** only ever touches that final small pool, keeping embedding time bounded regardless of total candidate count.

This is also why the rule engine has to be good on its own — it's responsible for not losing genuine fits before the semantic stage ever sees them.

---

## 📁 Repository structure

```
redrob-ranker/
├── rank.py                          # CLI entrypoint — the single source of scoring logic
├── app.py                           # Gradio sandbox UI (imports rank.py directly, no duplicated logic)
├── requirements.txt                 # Pinned dependencies
├── submission_metadata.yaml         # Portal metadata mirror (Section 10.2 of submission_spec)
├── README.md                        # You are here
├── data/
│   ├── candidates.jsonl.gz          # Hackathon-provided candidate pool (not committed — see Setup)
│   ├── job_description.md           # Hackathon-provided JD
│   └── submission.csv               # Output — generated by rank.py
└── .gitignore
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/redrob-ranker.git
cd redrob-ranker

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`** (pin exact versions you tested with):

```
sentence-transformers==3.0.1
faiss-cpu==1.8.0
numpy==1.26.4
tqdm==4.66.4
python-docx==1.1.2
gradio==5.x
```

Place the hackathon-provided files under `data/`:

```
data/candidates.jsonl.gz
data/job_description.md
```

> The bi-encoder model (`BAAI/bge-small-en-v1.5`) is loaded via `sentence-transformers` and cached locally on first run (`~/.cache/huggingface`). Pre-download it once with network access — the ranking step itself runs fully offline (`HF_HUB_OFFLINE=1` is set in `rank.py`), per the hackathon's no-network-during-ranking rule.

---

## ▶️ Reproduce the submission CSV

Single command, per Section 10.3 of the submission spec:

```bash
python rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --out data/submission.csv
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--prefilter` | `300` | Size of the pool fed into the bi-encoder after rule scoring |
| `--team-id` | `team_xxx` | Used for logging only — rename the output file to your registered team ID before upload |
| `--preview` | `100` | How many ranked rows to print to console |
| `--strong-pct` / `--adjacent-pct` | `60.0` / `30.0` | Percentile thresholds used to tag reasoning tier (strong/adjacent/filler) |

Console output streams progress through 5 stages (JD parsing → candidate loading → rule scoring → semantic search → hybrid ranking + validation) and prints elapsed time at each stage so the 5-minute budget is easy to monitor.

Validate the output before submitting:

```bash
python data/validate_submission.py data/submission.csv
```

---

## 🖥 Run the sandbox UI locally

The sandbox (also hosted live on HuggingFace Spaces) is a thin Gradio wrapper around the exact same `rank.py` — no duplicated logic, so what you see in the UI is what produced the CSV.

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. Upload a `candidates.jsonl`/`.jsonl.gz` sample (≤100 candidates works fine for a sandbox check) and a JD file, adjust the semantic pool size if you want, and click **Run Ranking**. Live logs stream stage-by-stage, and the ranked table + downloadable CSV appear once complete.

**Live deployed instance:** https://huggingface.co/spaces/kvishalini/Redrob-Ranker

---

## 📊 Scoring methodology

### Rule score (70% of hybrid) — `RuleScores`

| Component | Weight | Captures |
|---|---|---|
| Skill score | 30% | Tier-A/Tier-B skill match, with core JD skills (retrieval, ranking, embeddings, vector DB, Python) weighted 1.5× over supporting skills (LangChain, prompt engineering) |
| Career score | 35% | Title fit, years-of-experience fit to JD range, **inferred production deployment** (not just the word "production" — also infra/scale signal combos), product-company vs. IT-services context, career-seniority progression, academic/demo-only penalty |
| Location score | 10% | Target-city / target-country match, relocation willingness, work-mode preference alignment |
| Behavioural/availability score | 25% | Recency of activity, open-to-work flag, recruiter response rate, notice period, interview completion, offer acceptance, profile signals — per the 23 `redrob_signals` fields |

### Semantic score (30% of hybrid)

`BAAI/bge-small-en-v1.5` bi-encoder embeds the JD and each candidate's title + headline + summary + top skills + recent career/project/achievement text, cosine-similarity searched via FAISS `IndexFlatIP`, min-max normalized.

### Tie-breaking

When two hybrid scores land within `0.002` of each other, ties are broken in order by: confirmed production-retrieval evidence → exact JD-terminology match (NDCG/MRR/A-B test mentions) → time-at-product-company ratio → any production evidence → product-engineering ownership signals → GitHub activity → recruiter response rate → shorter notice period. Never by candidate ID alone unless every other signal is also tied.

### Reasoning generation

`build_reasoning()` produces a 1–2 sentence justification per candidate, grounded only in fields actually present in that candidate's profile (no hallucinated skills/employers), referencing the specific JD-relevant evidence found (named technologies, production company, ops signals like NDCG/embedding-drift handling), and honestly flagging concerns (notice period, inactivity, international location, low skill overlap) rather than uniformly positive boilerplate — per Stage 4's manual-review checks in the submission spec.

---

## ✅ Compute constraints compliance

| Constraint | How it's met |
|---|---|
| ≤ 5 min wall-clock | Two-stage pre-filtering keeps the bi-encoder pool small (≤300 by default); rule scoring on the 3,000-candidate Stage-B pool is pure Python/regex, no model inference |
| ≤ 16 GB RAM | No GPU tensors held in memory; `bge-small` is a ~130MB model; FAISS `IndexFlatIP` over ≤300 vectors is negligible |
| CPU only | `faiss-cpu`, `sentence-transformers` run on CPU; no `.cuda()` calls anywhere in the codebase |
| No network during ranking | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` set at the top of `rank.py` — the model must already be cached locally before running the ranking step (see Setup) |
| ≤ 5 GB disk | No large intermediate artifacts are written; embeddings are computed in-memory per run |

---

## 🪤 Honeypot & trap handling

Per `redrob_signals_doc.md` and the JD's hackathon note, the dataset includes deliberate traps:

- **Honeypots** (`is_honeypot()`): impossible experience-to-tenure ratios, 5+ "expert" skills with zero endorsements, non-technical titles (Marketing Manager, Accountant, etc.) with no corroborating technical evidence in career text — filtered before scoring.
- **Keyword stuffers**: countered by weighting career *evidence* (production deployment, ops signals, responsibilities text) over raw skills-array presence; a skills list full of AI keywords with no supporting career narrative scores poorly on the career component even if skill score is high.
- **Plain-language Tier-5 candidates**: the bi-encoder semantic layer and the production-inference text scan (`_infer_production`, `PROD_RETRIEVAL_KW`) deliberately don't require exact keyword matches like "RAG" or "Pinecone" — they catch career narratives that describe the work without naming the framework.
- **Behavioral twins**: tie-breaking and the behavioural score component differentiate otherwise-identical skill profiles using the 23 `redrob_signals` fields (response rate, notice period, recency, GitHub activity).

---

## ⚠️ Known limitations

- The bi-encoder pool size (`--prefilter`) trades runtime against semantic recall — a too-small pool risks missing strong candidates the rule engine under-scores; current default balances this against the 5-minute budget.
- Production-inference relies on text-pattern heuristics (`PRODUCTION_KW`, `PROD_RETRIEVAL_KW`) rather than structured signals, so it can occasionally miss production evidence phrased in an unusual way.
- No GPU/hosted-LLM re-ranking step is used by design, per the compute constraints — this is a deliberate latency-quality tradeoff, not an oversight.

---

## 📄 License

MIT — see [LICENSE](LICENSE).