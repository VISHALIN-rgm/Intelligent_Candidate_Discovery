# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Track 1 | Team:** `team_xxx`
**Model:** Hybrid Rule + Semantic Bi-Encoder Pipeline
**Runtime:** ~1.6 min on 16 GB CPU | **100K candidates → Top 100 ranked**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Pipeline Deep Dive](#pipeline-deep-dive)
4. [Scoring System](#scoring-system)
5. [Reasoning Engine](#reasoning-engine)
6. [Design Decisions](#design-decisions)
7. [Files & Dependencies](#files--dependencies)
8. [Reproduce the Submission](#reproduce-the-submission)

---

## Quick Start

```bash
# Install dependencies
pip install sentence-transformers faiss-cpu numpy tqdm

# Download the model once (requires network — do this before ranking)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run the ranker (no network required after model download)
python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.md \
  --out data/submission.csv

# Validate before submitting
python data/validate_submission.py data/submission.csv
```

> **Gzipped input is supported** — `candidates.jsonl.gz` works directly without unpacking.

---

## Architecture Overview

```
100,000 candidates
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 1 — Honeypot Filter          │  removes ~21,630 trap/impossible profiles
└─────────────────────────────────────┘
       │ ~78,370 clean candidates
       ▼
┌─────────────────────────────────────┐
│  STAGE 2 — Title Pre-filter         │  fast keyword check on title + headline
└─────────────────────────────────────┘
       │ ~28,760 relevant candidates
       ▼
┌─────────────────────────────────────┐
│  STAGE 3A — Cheap Sort              │  title fit + YoE in-range, microseconds/candidate
└─────────────────────────────────────┘
       │ top 3,000
       ▼
┌─────────────────────────────────────┐
│  STAGE 3B — Full Rule Score         │  4-component weighted score on all 23 signals
└─────────────────────────────────────┘
       │ top 300
       ▼
┌─────────────────────────────────────┐
│  STAGE 4 — Bi-Encoder + FAISS       │  all-MiniLM-L6-v2 semantic similarity
└─────────────────────────────────────┘
       │ 300 candidates with semantic scores
       ▼
┌─────────────────────────────────────┐
│  STAGE 5 — Hybrid Score             │  Rules 70% + Semantic 30%
│            YoE Priority Filter      │  in-range candidates fill top 100 first
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 6 — Reasoning Engine         │  per-candidate grounded natural language
│            Score Breakdown Tag      │  [skills · career · loc · avail]
└─────────────────────────────────────┘
       │
       ▼
  submission.csv (100 rows)
```

**Why this architecture?**
Scoring 28,760 candidates through a full ML pipeline would take 13+ minutes. The two-stage approach (cheap sort → full score on top 3,000) reduces that to ~2 seconds while keeping the quality of a full rule-based scorer on every candidate that matters.

---

## Pipeline Deep Dive

### Stage 1 — Honeypot Filter

Removes profiles with impossible or trap signals before any scoring:

| Check | Condition | Why |
|---|---|---|
| Timeline impossibility | `total_career_months / stated_years > 2.5` | Can't work 2.5× longer than you've existed |
| Expert stuffing | `≥5 "expert" skills + 0 total endorsements` | No real expert has zero peer validation |
| Non-tech title + zero tech evidence | `"accountant"` title + no ML keywords in descriptions | Keyword stuffers |

### Stage 2 — Title Pre-filter

Fast string scan on `current_title + headline` for any of:
`engineer, scientist, ml, ai, nlp, research, machine learning, data sci, deep learning, llm, python, ranking, retrieval, search, embedding`

No regex — pure `in` check across a frozenset. Runs in milliseconds across 78K candidates.

### Stage 3A — Cheap Sort (title + YoE only)

Each candidate gets a micro-score:
```
cheap_score = title_fit(1.0) - title_penalty(0.3) - wrong_domain(0.4) - junior_pen(0.2)
              + yoe_in_range(0.5)
```
All candidates sorted descending. Top 3,000 pass to full scoring. This reduces the expensive stage from 28,760 to 3,000 candidates — a **9.6× speedup** with negligible quality loss.

### Stage 4 — Bi-Encoder + FAISS Semantic Search

Model: `all-MiniLM-L6-v2` (384-dim, 80MB, CPU-fast)

Candidate text is built from title + headline + summary + top 12 skills + top 3 job descriptions (600 char cap). The full JD text is encoded once. FAISS `IndexFlatIP` computes cosine similarity (via normalized inner product) across all 300 pool candidates in milliseconds.

Semantic scores are min-max normalised to [0, 1] before hybrid blending.

### Stage 5 — Hybrid Score + YoE Priority

```
hybrid_score = 0.30 × semantic_norm + 0.70 × rule_total
```

**YoE Priority Filter:** In-range candidates (5-9yr) fill the top 100 first. Out-of-range candidates only pad if fewer than 100 in-range candidates exist. This prevents a 12yr candidate from displacing a 7yr candidate purely on semantic similarity.

**Score tie-breaking:** `sort(key=lambda x: (-x[0], x[3]["candidate_id"]))` — deterministic, candidate_id ascending.

---

## Scoring System

The rule score is a weighted sum of four components, all derived from candidate data and JD signals — no hardcoded rank positions.

```
rule_total = 0.30 × skill_score
           + 0.35 × career_score
           + 0.10 × loc_score
           + 0.25 × behav_score
```

### Component 1 — Skill Score (30%)

Scores each skill against three tiers derived from the JD:

| Tier | Examples | Weight multiplier |
|---|---|---|
| **Tier A** (JD must-haves) | embeddings, FAISS, NDCG, RAG, LTR, vector search, BM25 | Full weight |
| **Tier B** (nice-to-have) | PyTorch, Docker, Kafka, fine-tuning, AWS | 40% weight |
| **Penalty** | computer vision, SAP, Tableau, Salesforce | −30% weight |

Per-skill weight is further adjusted by proficiency (`beginner 0.25 → expert 1.2`), endorsements, and duration. Tier A hits above 3 get a capped bonus (`min(0.2, hits × 0.04)`).

Assessment scores from `skill_assessment_scores` add up to `+0.05` per matched skill per 100 points.

### Component 2 — Career Score (35%)

| Signal | Score |
|---|---|
| ML-primary title (AI Engineer, ML Engineer, NLP Engineer…) | `+0.35` |
| ML-adjacent title (Data Scientist, SWE-ML…) | `+0.20` (partial credit) |
| Wrong domain title (CV Engineer, Speech…) | **Hard cap: career ≤ 0.30** |
| Junior title for staff role | `−0.15` from title_score |
| YoE in 5-9yr range | `+0.20` |
| YoE 1yr outside range | `+0.08` |
| YoE above ceiling | `0.08 − (overage × 0.03)` continuous decay |
| Product company production hits (×2+) | `+0.25` |
| Product company production hits (×1) | `+0.15` |
| IT services production hits (×2+) | `+0.10` (half credit) |
| JD exact signals (embedding drift / NDCG / MRR / A/B test in descriptions) | `+0.10` |
| GitHub score > 50 | `+0.15` |
| IT services only career + JD prefers product | `−0.10` |

### Component 3 — Location Score (10%)

| Situation | Score |
|---|---|
| In target city (Pune/Noida/Hyderabad/Mumbai/Delhi) | `1.00` |
| In India + willing to relocate | `0.80` |
| In India, not relocating | `0.60` |
| Outside India + willing to relocate | `0.35` |
| Outside India, not relocating | `0.15` |
| Work mode matches JD (hybrid) | `+0.05` bonus |

### Component 4 — Behavioural Score (25%)

All 20 trackable signals from the 23 Redrob signals are used:

| Signal | Max contribution |
|---|---|
| `last_active_date` (≤14d) | 0.20 |
| `open_to_work_flag` | 0.12 |
| `recruiter_response_rate` (≥0.7) | 0.12 |
| `notice_period_days` (≤15d) | 0.15 |
| `interview_completion_rate` (≥0.8) | 0.08 |
| `offer_acceptance_rate` (≥0.7) | 0.08 |
| `avg_response_time_hours` (≤4h) | 0.06 |
| `saved_by_recruiters_30d` (≥10) | 0.05 |
| `profile_completeness_score` | 0.05 |
| `profile_views_received_30d` (≥10) | 0.03 |
| `applications_submitted_30d` (≥3) | 0.03 |
| `verified_email` | 0.02 |
| `verified_phone` | 0.02 |
| `search_appearance_30d` (≥10) | 0.02 |
| `connection_count` (≥300) | 0.02 |
| `linkedin_connected` | 0.01 |

> No artificial floor — a candidate with poor availability signals genuinely scores low on this component.

---

## Reasoning Engine

Every row in the submission CSV has a grounded reasoning string built entirely from the candidate's actual data — no templates, no hallucination.

### Three-tier system (score-percentile driven, not rank position)

| Tier | Score threshold | Tone | What's included |
|---|---|---|---|
| **Strong** | ≥ p60 of top-100 scores | Confident | Title + YoE + what they shipped + where + ops signals + assessment + GitHub + location + concerns |
| **Adjacent** | p30 – p60 | Honest | Title + YoE + skills + prod company + one ops signal + assessment + location + one concern |
| **Filler** | < p30 | Specific gaps | Title + YoE + top 2 skills + location + up to 3 specific derived gaps |

Thresholds are computed from `np.percentile(top100_scores, [60, 30])` after scoring — they adapt to each run's actual distribution.

### Concern surfacing rules

**Strong tier** — collects ALL applicable concerns (not just the first):
1. No confirmed production deployment
2. Junior title for senior/staff role
3. Adjacent title (Data Scientist, SWE-ML) — ML depth unconfirmed
4. Low skill match (skills < 0.45)
5. International candidate
6. Low recruiter response rate
7. Long notice period
8. Inactivity

**Filler tier** — builds a specific gap list from actual data:
- Production gap → states exactly which company type or "no deployment"
- Title mismatch → states exact title and JD seniority
- Location → states exact city, whether relocating
- YoE → states exact years vs ceiling
- Skill → states score if below 0.35
- Never uses "marginal JD fit" or any vague filler phrase

### Score breakdown tag (on every row)

```
[skills 0.85 · career 1.00 · loc 1.00 · avail 0.86]
```

Appended to every reasoning string. Makes every rank auditable — a recruiter can immediately see which component drove the placement.

### Truncation

`_safe_truncate(text, max_chars)` cuts at the last word boundary before the limit — never mid-word.

---

## Design Decisions

### Why 70% rules / 30% semantic?

Semantic similarity (`all-MiniLM-L6-v2`) is a strong signal for relevance but it's undiscriminating — it can't tell the difference between a candidate who *listed* "FAISS" as a skill versus one who *deployed FAISS to production at Zomato handling 10M queries*. The rule scorer captures that distinction precisely. Pure semantic would surface keyword stuffers; pure rules would miss strong candidates with non-standard vocabulary.

### Why two-stage scoring instead of scoring all 28K?

Scoring 28,760 candidates through the full `rule_score` function at 35 candidates/sec = 13+ minutes. The cheap sort (title regex + YoE check, ~microseconds each) narrows to 3,000 with near-zero quality loss because the cheap sort uses the same primary signals (title fit, YoE) that dominate the full score. This achieves sub-2-minute total runtime.

### Why raise skills weight to 30% and lower behavioural to 25%?

In initial runs, candidates with `skills 1.00` were landing at rank 78 (filler) because their availability signals (not open to work, no recent login) were depressing the score. Skills are a stable, long-term signal of fit. Availability is a tie-breaker — it shouldn't override someone who has shipped embedding systems to production. Raising skills weight by 5pp and lowering behavioural by 5pp fixed this distortion.

### Why partial credit for Data Scientist / SWE-ML titles?

The JD explicitly wants an ML Engineer / AI Engineer building ranking and retrieval systems. A Data Scientist title is adjacent — the person may or may not have the systems engineering depth the role needs. Giving these titles `0.20` instead of `0.35` title_score reflects genuine uncertainty about fit without hard-disqualifying them. The concern is then surfaced in reasoning so a human recruiter can verify.

### Why a continuous YoE ceiling penalty instead of a hard cliff?

A binary "in range / out of range" created unfair cliffs: an 8yr candidate and a 12yr candidate got the same penalty. The continuous formula `exp_score = 0.08 − (overage × 0.03)` means 8yr gets `0.05`, 9yr gets `0.02`, 10yr gets `−0.01` — a smooth gradient that reflects the JD's own language: "5-9 is a range, not a requirement."

### Why `all-MiniLM-L6-v2` and not `BAAI/bge-small-en-v1.5`?

Both were tested. `bge-small` has marginally better MTEB scores but `MiniLM-L6` is 2× faster on CPU encode and the quality difference doesn't meaningfully change top-100 composition when the semantic score is only 30% of the hybrid. Speed matters more given the 5-minute constraint.

---

## Files & Dependencies

### Repository structure

```
redrob-ranker/
├── rank.py                      # Main ranker — single command produces submission
├── data/
│   ├── candidates.jsonl         # 100K candidate pool (or .jsonl.gz)
│   ├── job_description.md       # JD file
│   ├── submission.csv           # Output
│   └── validate_submission.py   # Format validator
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

### `requirements.txt`

```
sentence-transformers==2.7.0
faiss-cpu==1.8.0
numpy>=1.24
tqdm>=4.65
```

> Python 3.9+ required. Tested on Windows 11 (PowerShell) and Ubuntu 22.04.

### Pre-computation note

`all-MiniLM-L6-v2` (~80 MB) must be downloaded before offline ranking:

```bash
# One-time download (requires network)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

The model is cached at `~/.cache/huggingface/hub/`. The ranking step itself makes **zero network calls** — enforced via `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`.

---

## Reproduce the Submission

### Single command

```bash
python rank.py --candidates data/candidates.jsonl --out data/submission.csv
```

### With all options

```bash
python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.md \
  --out data/submission.csv \
  --prefilter 300 \
  --strong-pct 60 \
  --adjacent-pct 30 \
  --preview 100
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--candidates` | required | Path to `.jsonl` or `.jsonl.gz` |
| `--jd` | auto-detect | Path to JD markdown (auto-finds `job_description.md` if not given) |
| `--out` | required | Output CSV path |
| `--prefilter` | `300` | Pool size passed to bi-encoder (increase for better semantic coverage) |
| `--strong-pct` | `60` | Score percentile threshold for strong tier |
| `--adjacent-pct` | `30` | Score percentile threshold for adjacent tier |
| `--preview` | `100` | Rows to print in terminal preview |

### Expected output

```
[1/5] Parsing job description...
      Loaded: job_description.md (1514 words)
      YoE range     : 5-9 yrs
      Target cities : ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr']
      Target country: india
      JD skills     : 47 detected
      Seniority     : staff  |  Work mode: hybrid  |  Prefers product: True
      Weights       : skill 0.3 · career 0.35 · loc 0.1 · avail 0.25

[2/5] Loading candidates...
      Loaded 100,000 candidates
      78,370 clean  (21,630 honeypots removed)

[3/5] Fast pre-filter + rule scoring...
      Fast filter: 78,370 → 28,760 relevant
      Stage-A filter: 28,760 → 3,000 for full scoring
      Rule scoring: 100%|████████| 3000/3000 [00:01<00:00]

[4/5] Bi-encoder semantic search (all-MiniLM-L6-v2)...
      Encoding 300 candidates...

[5/5] Hybrid scoring and writing top 100...
  Tier thresholds (p60/p30): strong >0.75, adjacent >0.72, filler <=0.72

  Submission written → data/submission.csv
  Total runtime     : 94.4s (1.6 min)
```

### Validate

```bash
python data/validate_submission.py data/submission.csv
# Expected: "All checks passed. 100 rows, ranks 1-100, scores non-increasing."
```

---

## Compute Constraints Compliance

| Constraint | Limit | This submission |
|---|---|---|
| Runtime | ≤ 5 min | ~1.6 min |
| Memory | ≤ 16 GB | ~2 GB peak |
| GPU | CPU only | ✅ No CUDA |
| Network | Off | ✅ `TRANSFORMERS_OFFLINE=1` |
| Disk | ≤ 5 GB | ~80 MB (model weights) |
| Honeypot rate in top 100 | < 10% | ✅ 0% (explicit filter) |

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.*