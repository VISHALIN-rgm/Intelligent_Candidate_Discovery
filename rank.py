#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

Usage:
    python rank.py --candidates data/candidates.jsonl --jd data/job_description.md --out data/submission.csv
    python rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --out data/submission.csv

Pipeline (target: <5 minutes on CPU, 16GB RAM, no network):
  1. JD Parser         — extracts YoE, cities, skills, seniority from any JD text
  2. Honeypot Filter   — removes impossible/trap profiles
  3. Title Pre-filter  — fast keyword check, drops non-tech profiles
  4. Two-stage scoring — cheap title sort then full rule score on top 3000
  5. Bi-Encoder+FAISS  — semantic search on top 300
  6. Hybrid Score      — Semantic(30%) + Rules(70%)
  7. Output            — top 100 with grounded reasoning + score breakdown

Score calibration notes:
  - Skill score weighted 30%: JD-matched skills are the strongest discriminator
  - Career score weighted 35%: title fit + YoE + production evidence + GitHub
  - Location score weighted 10%: target city > same country > relocatable > international
  - Behavioural score weighted 25%: availability signals (tie-breaker)
  - Production detection: requires ≥2 production keywords in a job description
  - Non-ML-primary titles (Data Scientist, SWE-ML) receive partial title_score (0.20 vs 0.35)
  - YoE over ceiling: continuous penalty (−0.03/yr) instead of a hard cliff
  - Reasoning is fully grounded: every gap named from actual candidate data
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
from collections import namedtuple
from datetime import date, datetime
from pathlib import Path

# Disable network access for HuggingFace — run fully offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# UTF-8 stdout/stderr — handles non-ASCII candidate names gracefully
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

TODAY            = date.today()
BIENCODER_MODEL  = "all-MiniLM-L6-v2"

# ── Hybrid weights ────────────────────────────────────────────────────────────
W_SEMANTIC = 0.30   # normalised cosine similarity from bi-encoder
W_RULES    = 0.70   # grounded rule score (skills + career + location + behaviour)

# ── Rule sub-component weights (must sum to 1.0) ──────────────────────────────
W_SKILL  = 0.30   # JD-matched skill depth — best discriminator
W_CAREER = 0.35   # title fit, YoE, production evidence, GitHub
W_LOC    = 0.10   # geography + work-mode fit
W_BEHAV  = 0.25   # availability / engagement signals (tie-breaker)

assert abs(W_SKILL + W_CAREER + W_LOC + W_BEHAV - 1.0) < 1e-9, \
    "Rule sub-component weights must sum to 1.0"

RuleScores = namedtuple("RuleScores", ["total", "skill", "career", "loc", "behav"])


# ══════════════════════════════════════════════════════════════════════════════
# SKILL REFERENCE SETS  (frozensets → O(1) lookup)
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_TIER_A = frozenset({
    # Vector / retrieval infrastructure
    "embeddings", "sentence-transformers", "sentence transformers",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus", "chroma",
    "opensearch", "elasticsearch", "vector search", "hybrid search",
    "semantic search", "dense retrieval", "information retrieval",
    "neural search", "vector index", "embedding pipeline",
    # Ranking & evaluation
    "ndcg", "mrr", "map", "reranking", "learning to rank", "ltr",
    "learning-to-rank", "ranking systems", "ai ranking", "intelligent ranking",
    "recall@k", "precision@k", "ranking evaluation", "retrieval evaluation",
    "a/b testing", "a/b test", "ab testing", "offline evaluation", "online evaluation",
    "offline-to-online", "evaluation framework",
    "retrieval quality", "retrieval quality regression", "ranking regression",
    "embedding drift", "index refresh", "index rebuild",
    "retrieval quality regression", "retrieval-quality regression",
    # Core NLP / LLM
    "nlp", "llm", "large language model", "transformers", "bert",
    "rag", "retrieval augmented generation", "text embeddings",
    "openai embeddings", "bge", "e5",
    "llm workflows", "llm powered", "llm-powered",
    # Systems & ML ops
    "python", "mlops", "production ml", "ranking", "bm25",
    "retrieval pipeline", "retrieval pipelines",
    "search pipeline", "search systems",
    # Domain-specific signals relevant to talent-tech JDs
    "recommendation system", "recommendation engine",
    "candidate ranking", "talent intelligence", "candidate discovery",
    "intelligent recommendation",
})

SKILLS_TIER_B = frozenset({
    # Fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    # Classical ML
    "xgboost", "lightgbm", "tfidf", "neural ranking",
    "learning to rank models",
    # Infra & serving
    "kafka", "spark", "fastapi", "bentoml", "triton",
    "model serving", "inference optimization", "distributed systems",
    "large-scale inference", "large scale inference",
    # ML frameworks
    "hugging face", "huggingface", "pytorch", "tensorflow", "scikit-learn",
    # Cloud & DevOps
    "aws", "gcp", "azure", "docker", "kubernetes",
    # Data engineering
    "feature engineering", "data pipelines",
    # Experimentation
    "experimentation", "online evaluation",
    # Domain context
    "hr-tech", "hr tech", "recruiting tech", "recruitech",
    "marketplace", "marketplace products", "talent marketplace",
    # Open source
    "open-source", "open source", "open source contribution",
    "open source contributions",
})

SKILLS_PENALTY = frozenset({
    "computer vision", "image classification", "object detection",
    "speech recognition", "asr", "tts", "text to speech",
    "robotics", "solidworks", "ansys", "autocad",
    "sap", "six sigma", "photoshop", "illustrator", "figma",
    "marketing", "seo", "content writing", "accounting",
    "crm", "salesforce", "tableau", "power bi",
})


# ══════════════════════════════════════════════════════════════════════════════
# TITLE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# ML-adjacent but not ML-primary: partial title credit only
ML_ADJACENT_TITLE_RE = re.compile(
    r'data scientist|senior data scientist|data science|'
    r'software engineer.*\bml\b|swe.*\bml\b|software engineer.*\bai\b|'
    r'senior software engineer',
    re.IGNORECASE,
)

# Primary ML/AI/NLP/Retrieval titles: full title credit
TITLE_FIT_RE = re.compile(
    r'\bml\b|machine learning|ai engineer|\bnlp\b|'
    r'research engineer|applied (scientist|ml|ai)|search engineer|'
    r'retrieval|ranking engineer|recommendation|\bllm\b|generative ai|'
    r'deep learning|senior.*engineer|staff engineer|principal engineer|'
    r'embedding|intelligence engineer|ml engineer|applied scientist',
    re.IGNORECASE,
)

# Non-tech titles that should be penalised
TITLE_PENALTY_RE = re.compile(
    r'\bmarketing\b|operations manager|hr manager|accountant|'
    r'civil engineer|mechanical engineer|customer support|'
    r'\bsales\b|content writer|business analyst|financial analyst|'
    r'brand manager',
    re.IGNORECASE,
)

# CV / vision / robotics: wrong technical domain → hard cap career score
WRONG_DOMAIN_RE = re.compile(
    r'computer vision|cv engineer|\bimage\b|speech|robotics|mechanical|civil|hardware',
    re.IGNORECASE,
)

# Junior seniority indicators
JUNIOR_TITLE_RE = re.compile(
    r'\bjunior\b|\bentry.level\b|\bfresher\b|\bintern\b|\bgraduate trainee\b',
    re.IGNORECASE,
)

# Titles normalised to lowercase for exact-set membership checks
NON_ML_PRIMARY_TITLES = frozenset({
    "data scientist", "senior data scientist", "data science lead",
    "senior software engineer", "software engineer", "senior data engineer",
    "senior software engineer (ml)", "software engineer (ml)",
    "software engineer (ai)", "senior software engineer (ai)",
    "data engineer",
})

# Keywords that pass the fast title pre-filter
TECH_TITLE_KW = frozenset({
    "engineer", "scientist", "ml", "ai", "nlp", "research",
    "machine learning", "data sci", "deep learning", "llm",
    "python", "ranking", "retrieval", "search", "embedding",
})


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION / CAREER SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

SERVICES_COMPANIES = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "hexaware", "mphasis", "mindtree",
    "l&t infotech", "ltimindtree",
})

# Production keywords: ≥2 hits in a job description = confirmed production role
PRODUCTION_KW = frozenset({
    "production", "deployed", "shipped", "serving", "at scale",
    "real users", "latency", "inference", "live system",
    "millions", "query per second", "qps", "p99", "sla", "uptime",
    "embedding drift", "index refresh", "retrieval quality",
})

# JD-specific exact phrases: surfaced as an explicit positive signal
JD_EXACT_KW = frozenset({
    "embedding drift", "index refresh", "retrieval quality regression",
    "retrieval-quality regression", "a/b test", "offline-to-online",
    "ndcg", "mrr", "eval framework", "ranking regression",
    "retrieval quality", "index rebuild", "vector index",
})

# Operational signals mapped to readable labels for reasoning
OPS_SIGNALS: list[tuple[str, str]] = [
    ("embedding drift",   "handled embedding drift"),
    ("index refresh",     "managed index refresh"),
    ("ndcg",              "NDCG evaluation"),
    ("mrr",               "MRR evaluation"),
    ("a/b test",          "A/B testing"),
    ("offline-to-online", "offline-to-online eval"),
    ("retrieval quality", "retrieval quality monitoring"),
]

# Known tech cities for location matching
KNOWN_TECH_CITIES = [
    "pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai",
    "delhi", "gurgaon", "gurugram", "ncr", "chennai", "kolkata",
    "ahmedabad", "kochi", "jaipur", "indore", "coimbatore",
    "new york", "san francisco", "london", "singapore", "dubai",
    "berlin", "amsterdam", "toronto", "sydney", "seattle", "boston",
]

# Proficiency weight multipliers for skill scoring
PROFICIENCY_W: dict[str, float] = {
    "beginner":     0.25,
    "intermediate": 0.60,
    "advanced":     1.00,
    "expert":       1.20,
}

# Non-tech honeypot titles
NON_TECH_HONEYPOT_TITLES = frozenset({
    "accountant", "hr manager", "marketing manager", "civil engineer",
    "mechanical engineer", "sales manager", "operations manager",
    "customer support", "content writer", "brand manager",
})

# Minimum tech evidence keywords to rescue a borderline honeypot title
TECH_EVIDENCE_KW = frozenset({
    "python", "machine learning", "nlp", "vector", "embedding",
    "retrieval", "transformer", "deep learning", "model",
    "ranking", "recommendation", "search",
})


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def parse_date(s: str | None) -> date | None:
    """Parse ISO-8601 date string; return None on failure."""
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_since(d: date | None) -> int:
    """Days between today and date d; 9999 if d is None."""
    if d is None:
        return 9999
    return max(0, (TODAY - d).days)


def _safe_truncate(text: str, max_chars: int) -> str:
    """Truncate at a word boundary, never mid-word, appending ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(".,;") + "…"


def _count_production_hits(desc_lower: str) -> int:
    """Count distinct production keywords in a lowercased job description."""
    return sum(1 for kw in PRODUCTION_KW if kw in desc_lower)


# ══════════════════════════════════════════════════════════════════════════════
# JD PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_jd(jd_text: str) -> dict:
    """
    Extract structured signals from raw JD text.

    Returns a dict with:
        yoe_min, yoe_max, target_cities, target_country, jd_skills,
        seniority, work_mode, prefers_product, raw_text,
        _effective_tier_a  (SKILLS_TIER_A ∪ jd-detected skills)
    """
    t = jd_text.lower()

    # ── YoE range ─────────────────────────────────────────────────────────────
    m = re.search(
        r'(\d+)\s*(?:to|-|–)\s*(\d+)\s*years?|(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        t,
    )
    if m:
        if m.group(1) and m.group(2):
            yoe_min, yoe_max = int(m.group(1)), int(m.group(2))
        else:
            yoe_min = int(m.group(3))
            yoe_max = yoe_min + 4
    else:
        # Fall back to seniority keywords
        if any(w in t for w in ("senior", "staff", "lead", "principal")):
            yoe_min, yoe_max = 5, 12
        elif any(w in t for w in ("junior", "entry", "fresher", "graduate")):
            yoe_min, yoe_max = 0, 3
        else:
            yoe_min, yoe_max = 3, 8

    # ── Target cities ─────────────────────────────────────────────────────────
    target_cities = [city for city in KNOWN_TECH_CITIES if city in t]

    # ── Target country ────────────────────────────────────────────────────────
    country_hints: dict[str, list[str]] = {
        "india":     ["india", "indian", "inr", "lpa", "lakhs",
                      "bengaluru", "hyderabad", "pune", "noida"],
        "usa":       ["united states", " usa", "usd", "silicon valley"],
        "uk":        ["united kingdom", " uk ", "gbp", "london"],
        "singapore": ["singapore", "sgd"],
        "uae":       ["dubai", "aed", "uae"],
    }
    target_country: str | None = None
    for country, hints in country_hints.items():
        if any(h in t for h in hints):
            target_country = country
            break

    # ── JD skill detection ────────────────────────────────────────────────────
    jd_skills = {s for s in (SKILLS_TIER_A | SKILLS_TIER_B) if s in t}

    # ── Seniority ─────────────────────────────────────────────────────────────
    seniority = "mid"
    for level, kws in (
        ("staff",  ("staff engineer", "principal", "architect")),
        ("senior", ("senior", "sr.", "lead")),
        ("junior", ("junior", "entry level", "fresher")),
    ):
        if any(k in t for k in kws):
            seniority = level
            break

    # ── Work mode ─────────────────────────────────────────────────────────────
    if "fully remote" in t or "100% remote" in t:
        work_mode = "remote"
    elif "onsite" in t or "on-site" in t:
        work_mode = "onsite"
    elif "remote" in t:
        work_mode = "remote"
    else:
        work_mode = "hybrid"

    # ── Product vs services preference ────────────────────────────────────────
    prefers_product = any(w in t for w in (
        "product company", "founding team", "startup",
        "not consulting", "not it services", "product-first",
    ))

    effective_tier_a = SKILLS_TIER_A | jd_skills

    return {
        "raw_text":          jd_text,
        "yoe_min":           yoe_min,
        "yoe_max":           yoe_max,
        "target_cities":     target_cities,
        "target_country":    target_country,
        "jd_skills":         jd_skills,
        "seniority":         seniority,
        "work_mode":         work_mode,
        "prefers_product":   prefers_product,
        "_effective_tier_a": effective_tier_a,
    }


def load_jd(jd_path: str | None = None, candidates_path: str | None = None) -> str:
    """Load JD text from file; auto-discover if path not given."""
    if jd_path is None:
        search_dirs = []
        if candidates_path:
            search_dirs.append(Path(candidates_path).parent)
        search_dirs.append(Path("."))
        for d in search_dirs:
            for name in ("job_description.md", "job_description.txt",
                         "jd.md", "jd.txt", "JD.md"):
                candidate = d / name
                if candidate.exists():
                    jd_path = str(candidate)
                    break
            if jd_path:
                break

    if jd_path is None:
        print("ERROR: No JD file found. Pass --jd path/to/job_description.md", flush=True)
        sys.exit(1)

    p = Path(jd_path)
    if not p.exists():
        print(f"ERROR: JD file not found: {jd_path}", flush=True)
        sys.exit(1)

    text = p.read_text(encoding="utf-8")
    print(f"      Loaded: {p.name} ({len(text.split())} words)", flush=True)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# HONEYPOT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def is_honeypot(c: dict) -> bool:
    """
    Return True if the candidate profile shows clear fabrication signals:
      1. Career history implies >2.5× the stated YoE (impossible timeline)
      2. ≥5 self-declared expert skills with zero endorsements (implausible)
      3. Non-tech title with no tech evidence anywhere in career descriptions
    """
    p      = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    # Signal 1: impossible timeline
    total_months = sum(j.get("duration_months", 0) for j in career)
    stated_years = p.get("years_of_experience", 0)
    if stated_years > 0 and total_months > 0:
        if (total_months / 12) / stated_years > 2.5:
            return True

    # Signal 2: implausible expertise claims
    expert_count  = sum(1 for s in skills if s.get("proficiency") == "expert")
    total_endorsements = sum(s.get("endorsements", 0) for s in skills)
    if expert_count >= 5 and total_endorsements == 0:
        return True

    # Signal 3: non-tech title without any tech evidence in career
    title = p.get("current_title", "").lower()
    if any(nt in title for nt in NON_TECH_HONEYPOT_TITLES):
        has_tech_evidence = any(
            kw in j.get("description", "").lower()
            for j in career
            for kw in TECH_EVIDENCE_KW
        )
        if not has_tech_evidence:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# FAST TITLE PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════

def is_relevant(c: dict) -> bool:
    """Fast keyword scan — drops obviously non-tech profiles before full scoring."""
    p        = c["profile"]
    combined = (p.get("current_title", "") + " " + p.get("headline", "")).lower()
    return any(kw in combined for kw in TECH_TITLE_KW)


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE TEXT FOR BI-ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def build_candidate_text(c: dict) -> str:
    """
    Produce a rich, compact text representation for semantic encoding.
    Caps at 600 chars so the bi-encoder receives meaningful signal without
    exceeding its context window.
    """
    p   = c["profile"]
    sig = c.get("redrob_signals", {})

    parts = [
        p.get("current_title", ""),
        p.get("headline", ""),
        p.get("summary", "")[:200],
        "Skills: " + ", ".join(s["name"] for s in c.get("skills", [])[:12]),
    ]

    for job in c.get("career_history", [])[:3]:
        parts.append(
            f"{job.get('title', '')} at {job.get('company', '')}. "
            f"{job.get('description', '')[:150]}"
        )

    if sig.get("open_to_work_flag"):
        parts.append("Open to work.")

    gh = sig.get("github_activity_score", -1)
    if gh > 30:
        parts.append(f"GitHub {gh}/100.")

    return " ".join(parts)[:600]


# ══════════════════════════════════════════════════════════════════════════════
# RULE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def rule_score(c: dict, jd: dict) -> RuleScores:
    """
    Compute a grounded RuleScores namedtuple for a single candidate.

    All scores are in [0, 1].  No hardcoded candidate IDs, ranks, or scores.
    """
    p      = c["profile"]
    sig    = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    eff_a   = jd["_effective_tier_a"]
    yoe_min = jd["yoe_min"]
    yoe_max = jd["yoe_max"]

    # ── 1. Skill score ─────────────────────────────────────────────────────────
    # Each skill is weighted by proficiency, endorsements (social proof), and
    # duration (sustained use).  Tier-B skills get 40% credit.  Penalty skills
    # reduce the score.  A bonus for tier-A hit count rewards breadth.
    tier_a_score = tier_b_score = penalty_score = total_weight = 0.0
    tier_a_hits  = 0

    for sk in skills:
        name  = sk["name"].lower()
        pw    = PROFICIENCY_W.get(sk.get("proficiency", "beginner"), 0.25)
        end_t = min(1.0, (sk.get("endorsements", 0) + 1) / 10.0)
        dur_t = min(1.0, sk.get("duration_months", 0) / 24.0)
        # Combined weight: proficiency dominates; endorsements and duration
        # each contribute 30% of the social/usage component.
        w = pw * (0.4 + 0.3 * end_t + 0.3 * dur_t)
        total_weight += w

        in_a = name in eff_a
        in_b = (not in_a) and (name in SKILLS_TIER_B)
        in_p = name in SKILLS_PENALTY

        if in_a:
            tier_a_score += w
            tier_a_hits  += 1
        elif in_b:
            tier_b_score += w * 0.4

        if in_p:
            penalty_score += w * 0.3

    skill_score = 0.0
    if total_weight > 0:
        raw = max(0.0, (tier_a_score + tier_b_score - penalty_score) / total_weight)
        # Breadth bonus: up to 0.20 for multiple distinct tier-A skills
        skill_score = min(1.0, raw + min(0.20, tier_a_hits * 0.04))

    # Assessment scores can nudge skill_score slightly upward
    for skill_name, score_val in sig.get("skill_assessment_scores", {}).items():
        if skill_name.lower() in eff_a:
            skill_score = min(1.0, skill_score + (score_val / 100.0) * 0.05)

    # ── 2. Career score ────────────────────────────────────────────────────────
    title    = p.get("current_title", "").lower()
    headline = p.get("headline", "").lower()
    combined = title + " " + headline

    # Title fit: ML-primary → full credit; ML-adjacent → partial credit
    if TITLE_FIT_RE.search(combined) and not ML_ADJACENT_TITLE_RE.search(title):
        title_score = 0.35
    elif ML_ADJACENT_TITLE_RE.search(title):
        title_score = 0.20
    else:
        title_score = 0.0

    if TITLE_PENALTY_RE.search(title):
        title_score = max(0.0, title_score - 0.20)
    if WRONG_DOMAIN_RE.search(title):
        title_score = max(0.0, title_score - 0.30)
    if JUNIOR_TITLE_RE.search(title) and jd.get("seniority") in ("senior", "staff"):
        title_score = max(0.0, title_score - 0.15)

    # YoE score: full credit in range, continuous penalty above ceiling
    yoe = p.get("years_of_experience", 0)
    if yoe_min <= yoe <= yoe_max:
        exp_score = 0.20
    elif (yoe_min - 1) <= yoe < yoe_min:
        exp_score = 0.08                          # 1 year below floor: partial
    elif yoe > yoe_max:
        overage   = yoe - yoe_max
        exp_score = max(-0.10, 0.08 - overage * 0.03)   # −0.03/yr above ceiling
    else:
        exp_score = -0.10                         # significantly under

    # Production evidence: scan career history for production signals
    svc_months = prod_months = total_months = 0
    prod_hits_product = prod_hits_any = 0
    consulting_only   = True
    jd_exact_signal   = False

    for job in career:
        co    = job.get("company", "").lower()
        ind   = job.get("industry", "").lower()
        desc  = job.get("description", "").lower()
        dur   = job.get("duration_months", 0)
        total_months += dur

        is_svc = (
            any(s in co for s in SERVICES_COMPANIES) or
            "it services" in ind or
            "consulting" in ind
        )

        if is_svc:
            svc_months += dur
        else:
            prod_months  += dur
            consulting_only = False

        hits = _count_production_hits(desc)
        if hits >= 2:
            prod_hits_any += 1
            if not is_svc:
                prod_hits_product += 1

        if not jd_exact_signal and any(kw in desc for kw in JD_EXACT_KW):
            jd_exact_signal = True

    # Product-company exposure ratio
    prod_ratio  = (prod_months / total_months * 0.20) if total_months > 0 else 0.05

    # Consulting-only penalty (only when JD prefers product)
    consult_pen = (
        -0.10 if (consulting_only and len(career) > 1 and jd.get("prefers_product"))
        else 0.0
    )

    # Production bonus: product-company deployments valued > IT services
    if prod_hits_product >= 2:
        prod_bonus = 0.25
    elif prod_hits_product == 1:
        prod_bonus = 0.15
    elif prod_hits_any >= 2:
        prod_bonus = 0.10
    elif prod_hits_any == 1:
        prod_bonus = 0.05
    else:
        prod_bonus = 0.0

    jd_signal_bonus = 0.10 if jd_exact_signal else 0.0

    gh = sig.get("github_activity_score", -1)
    gh_bonus = (
        0.15 if gh > 50 else
        0.08 if gh > 20 else
        0.03 if gh > 0  else 0.0
    )

    career_score = min(1.0, max(0.0,
        title_score + exp_score + prod_ratio +
        consult_pen + prod_bonus + jd_signal_bonus + gh_bonus
    ))

    # Hard cap for wrong-domain candidates
    if WRONG_DOMAIN_RE.search(title):
        career_score = min(career_score, 0.30)

    # ── 3. Location score ──────────────────────────────────────────────────────
    country        = p.get("country", "").lower()
    location       = p.get("location", "").lower()
    relocate       = sig.get("willing_to_relocate", False)
    target_country = (jd.get("target_country") or "").lower()

    in_target_city    = any(city in location for city in jd["target_cities"])
    in_target_country = (not target_country) or (country == target_country)

    if in_target_city:
        loc_score = 1.0
    elif in_target_country and relocate:
        loc_score = 0.80
    elif in_target_country:
        loc_score = 0.60
    elif relocate:
        loc_score = 0.35
    else:
        loc_score = 0.15

    # Work-mode alignment bonus
    cand_mode = sig.get("preferred_work_mode", "flexible")
    if cand_mode == "flexible" or cand_mode == jd.get("work_mode", "hybrid"):
        loc_score = min(1.0, loc_score + 0.05)

    # ── 4. Behavioural / availability score ───────────────────────────────────
    # Additive signals; no artificial floor.  Acts as a tie-breaker.
    behav = 0.0

    # Recency of activity
    days_inactive = days_since(parse_date(sig.get("last_active_date")))
    behav += (
        0.20 if days_inactive <= 14  else
        0.17 if days_inactive <= 30  else
        0.13 if days_inactive <= 60  else
        0.08 if days_inactive <= 90  else
        0.04 if days_inactive <= 180 else 0.01
    )

    if sig.get("open_to_work_flag"):
        behav += 0.12

    rr = sig.get("recruiter_response_rate", 0)
    behav += 0.12 if rr >= 0.7 else 0.08 if rr >= 0.4 else 0.04 if rr >= 0.2 else 0.0

    art = sig.get("avg_response_time_hours", 999)
    behav += 0.06 if art <= 4 else 0.04 if art <= 24 else 0.02 if art <= 72 else 0.0

    notice = sig.get("notice_period_days", 90)
    behav += (
        0.15 if notice <= 15 else
        0.12 if notice <= 30 else
        0.07 if notice <= 60 else
        0.03 if notice <= 90 else 0.01
    )

    icr = sig.get("interview_completion_rate", 0)
    behav += 0.08 if icr >= 0.8 else 0.05 if icr >= 0.6 else 0.02 if icr >= 0.4 else 0.0

    oar = sig.get("offer_acceptance_rate", -1)
    behav += (
        0.08 if oar >= 0.7 else
        0.05 if oar >= 0.4 else
        0.02 if oar >= 0.1 else
        0.03 if oar == -1  else 0.0    # unknown: mild positive (not flagged bad)
    )

    behav += sig.get("profile_completeness_score", 0) / 100.0 * 0.05

    saved = sig.get("saved_by_recruiters_30d", 0)
    behav += 0.05 if saved >= 10 else 0.03 if saved >= 5 else 0.01 if saved >= 2 else 0.0

    apps = sig.get("applications_submitted_30d", 0)
    behav += 0.03 if apps >= 3 else 0.01 if apps >= 1 else 0.0

    if sig.get("verified_email"):     behav += 0.02
    if sig.get("verified_phone"):     behav += 0.02
    if sig.get("linkedin_connected"): behav += 0.01

    conn = sig.get("connection_count", 0)
    behav += 0.02 if conn >= 300 else 0.01 if conn >= 100 else 0.0

    views = sig.get("profile_views_received_30d", 0)
    behav += 0.03 if views >= 10 else 0.02 if views >= 5 else 0.01 if views >= 2 else 0.0

    appearances = sig.get("search_appearance_30d", 0)
    behav += 0.02 if appearances >= 10 else 0.01 if appearances >= 3 else 0.0

    behav_score = min(1.0, max(0.0, behav))

    total = (
        W_SKILL  * skill_score  +
        W_CAREER * career_score +
        W_LOC    * loc_score    +
        W_BEHAV  * behav_score
    )
    return RuleScores(
        total=total,
        skill=skill_score,
        career=career_score,
        loc=loc_score,
        behav=behav_score,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REASONING BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _find_production_company(career: list[dict]) -> tuple[str, bool]:
    """
    Scan career history and return (company_name, is_services).
    Prefers a product-company hit over an IT-services hit.
    Returns ("", False) if no production evidence found.
    """
    fallback_svc = ("", False)
    for job in career[:4]:
        desc   = job.get("description", "").lower()
        co     = job.get("company", "").lower()
        is_svc = any(s in co for s in SERVICES_COMPANIES)
        if _count_production_hits(desc) >= 2:
            if not is_svc:
                return job.get("company", ""), False   # product company — stop here
            elif not fallback_svc[0]:
                fallback_svc = (job.get("company", ""), True)
    return fallback_svc


def _find_ops_signals(career: list[dict]) -> list[str]:
    """Return list of operational signal labels found in career descriptions."""
    found: list[str] = []
    for keyword, label in OPS_SIGNALS:
        for job in career:
            if keyword in job.get("description", "").lower():
                if label not in found:
                    found.append(label)
                break
    return found


def _build_location_string(
    location: str,
    country: str,
    target_country: str,
    in_target: bool,
    is_intl: bool,
    relocate: bool,
) -> str:
    """Compose a concise, accurate location phrase for reasoning."""
    if in_target:
        return f"{location} (target city)"
    if country.lower() == target_country and relocate:
        return f"{location}, open to relocate"
    if country.lower() == target_country:
        return location
    suffix = ", open to relocate" if relocate else ""
    return f"{location}, {country}{suffix}"


def build_reasoning(
    c: dict,
    jd: dict,
    sem_n: float,
    rs: RuleScores,
    tier_label: str,
) -> str:
    """
    Produce a grounded, ≤300-char reasoning string followed by a score tag.

    Rules:
    - Every positive claim is derived from actual candidate data.
    - Every concern names a specific, derived gap — never a vague phrase.
    - Strong tier: positive highlights then concerns.
    - Adjacent tier: condensed highlights then concerns.
    - Filler tier: brief positive then itemised gap list.
    - Score tag appended after truncation.
    """
    p      = c["profile"]
    sig    = c.get("redrob_signals", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])

    eff_a   = jd["_effective_tier_a"]
    yoe_min = jd["yoe_min"]
    yoe_max = jd["yoe_max"]
    title   = p.get("current_title", "")
    yoe     = p.get("years_of_experience", 0)

    # Matched skills: top 3 by endorsements
    matched_skills = sorted(
        [sk for sk in skills if sk["name"].lower() in eff_a],
        key=lambda x: x.get("endorsements", 0),
        reverse=True,
    )
    top_skills = [s["name"] for s in matched_skills[:3]]

    # Production evidence
    prod_company, prod_is_services = _find_production_company(career)

    # Operational signals
    ops_found = _find_ops_signals(career)

    # Best relevant assessment
    best_assessment = ""
    best_val = -1.0
    for k, v in sig.get("skill_assessment_scores", {}).items():
        if k.lower() in eff_a and v > best_val:
            best_val = v
            best_assessment = f"{k} {v:.0f}/100"

    gh = sig.get("github_activity_score", -1)

    # Location signals
    location       = p.get("location", "")
    country        = p.get("country", "")
    target_country = (jd.get("target_country") or "").lower()
    relocate       = sig.get("willing_to_relocate", False)
    in_target      = any(city in location.lower() for city in jd["target_cities"])
    is_intl        = bool(target_country and country.lower() != target_country)

    loc_str = _build_location_string(
        location, country, target_country, in_target, is_intl, relocate
    )

    # Availability signals
    notice     = sig.get("notice_period_days", 90)
    open_flag  = "open to work" if sig.get("open_to_work_flag") else "not flagged open"
    inactive   = days_since(parse_date(sig.get("last_active_date")))
    active_str = f"active {inactive}d ago" if inactive < 365 else "inactive >1yr"
    rr         = sig.get("recruiter_response_rate", 0)
    art        = sig.get("avg_response_time_hours", 999)

    # Experience note — always surfaces ceiling breach
    if yoe_min <= yoe <= yoe_max:
        exp_note = f"{yoe:.0f}yr"
    elif yoe > yoe_max:
        exp_note = f"{yoe:.0f}yr (above {yoe_max}yr ceiling)"
    elif yoe == yoe_min - 1:
        exp_note = f"{yoe:.0f}yr (1yr below floor)"
    else:
        exp_note = f"{yoe:.0f}yr (outside {yoe_min}-{yoe_max}yr range)"

    # Title flags
    title_lower       = title.lower()
    is_junior_title   = bool(JUNIOR_TITLE_RE.search(title_lower))
    jd_wants_senior   = jd.get("seniority") in ("senior", "staff")
    is_adjacent_title = bool(ML_ADJACENT_TITLE_RE.search(title_lower))
    is_non_ml_primary = any(t in title_lower for t in NON_ML_PRIMARY_TITLES)

    # Score tag appended after body is truncated — always present
    score_tag = (
        f"[skills {rs.skill:.2f} · career {rs.career:.2f} · "
        f"loc {rs.loc:.2f} · avail {rs.behav:.2f}]"
    )
    max_body = 300 - len(score_tag) - 1

    # ── STRONG TIER ────────────────────────────────────────────────────────────
    if tier_label == "strong":
        parts = [f"{title}, {exp_note}"]

        if prod_company and top_skills and not prod_is_services:
            parts.append(
                f"shipped {' + '.join(top_skills[:2])} to production at {prod_company}"
            )
        elif prod_company and top_skills and prod_is_services:
            parts.append(
                f"deployed {' + '.join(top_skills[:2])} at {prod_company} (IT services)"
            )
        elif prod_company and not prod_is_services:
            parts.append(f"production deployment at {prod_company}")
        elif top_skills:
            parts.append(f"strong skills: {', '.join(top_skills)}")

        if ops_found:
            parts.append(f"ops: {', '.join(ops_found[:2])}")
        if best_assessment:
            parts.append(f"assessed {best_assessment}")
        if gh > 50:
            parts.append(f"GitHub {gh}/100")
        parts.append(f"{loc_str}; {open_flag}, {active_str}")

        # Collect all disqualifying signals — never stop at the first match
        concerns: list[str] = []
        if not prod_company:
            concerns.append("No confirmed production deployment.")
        if is_junior_title and jd_wants_senior:
            concerns.append(
                f"Title suggests junior level — verify seniority for {jd['seniority']} role."
            )
        elif is_non_ml_primary and jd_wants_senior:
            concerns.append(
                f"'{title}' is an adjacent title — verify ML depth for {jd['seniority']} role."
            )
        if rs.skill < 0.45:
            concerns.append(
                f"Low JD skill match (skills {rs.skill:.2f}) — verify technical depth."
            )
        if is_intl:
            concerns.append(
                f"International candidate ({country}) — relocation or visa required."
            )
        if rr < 0.2:
            concerns.append(f"Low recruiter response ({rr:.0%}).")
        elif notice > 90:
            concerns.append(f"Long notice ({notice}d).")
        elif inactive > 180:
            concerns.append(f"Inactive {inactive}d — verify availability.")
        elif art > 120:
            concerns.append(f"Avg response {art:.0f}h.")

        concern_str = (" " + " ".join(concerns[:3])) if concerns else ""
        body = ". ".join(parts) + "." + concern_str

    # ── ADJACENT TIER ──────────────────────────────────────────────────────────
    elif tier_label == "adjacent":
        parts = [f"{title}, {exp_note}"]
        if top_skills:
            parts.append(f"skills: {', '.join(top_skills)}")
        if prod_company and not prod_is_services:
            parts.append(f"prod at {prod_company}")
        elif prod_company and prod_is_services:
            parts.append(f"deployed at {prod_company} (IT services)")
        if ops_found:
            parts.append(ops_found[0])
        if best_assessment:
            parts.append(f"assessed {best_assessment}")
        parts.append(f"{loc_str}; {open_flag}, {active_str}")

        concerns: list[str] = []
        if not prod_company:
            concerns.append("No confirmed production deployment.")
        if is_junior_title and jd_wants_senior:
            concerns.append(f"Title suggests junior level for {jd['seniority']} role.")
        elif is_non_ml_primary:
            concerns.append("Adjacent title — ML depth unconfirmed.")
        if is_intl:
            concerns.append("International candidate — relocation needed.")
        if rr < 0.2:
            concerns.append(f"Low recruiter response ({rr:.0%}).")
        elif notice > 90:
            concerns.append(f"Long notice ({notice}d).")
        elif inactive > 120:
            concerns.append(f"Inactive {inactive}d.")
        elif art > 120:
            concerns.append(f"Slow response ({art:.0f}h).")

        concern_str = (" " + " ".join(concerns[:2])) if concerns else ""
        body = ". ".join(parts) + "." + concern_str

    # ── FILLER TIER ────────────────────────────────────────────────────────────
    else:
        gaps: list[str] = []

        # Production gap
        if not prod_company:
            gaps.append("no confirmed production deployment")
        elif prod_is_services:
            gaps.append(f"production only at IT services ({prod_company})")

        # Title gaps
        if is_junior_title and jd_wants_senior:
            gaps.append(f"junior title for {jd['seniority']} role")
        elif is_non_ml_primary:
            gaps.append(f"adjacent title ({title}) — ML depth unconfirmed")

        # Location
        if is_intl:
            gaps.append(f"international candidate ({country})")
        elif not in_target and not relocate and target_country and country.lower() == target_country:
            gaps.append(f"non-target city ({location}) — not open to relocate")

        # YoE
        if yoe > yoe_max:
            gaps.append(f"{yoe:.0f}yr above {yoe_max}yr ceiling")
        elif yoe < yoe_min - 1:
            gaps.append(f"{yoe:.0f}yr below {yoe_min}yr floor")

        # Availability
        if rr < 0.3:
            gaps.append(f"low recruiter response ({rr:.0%})")
        if notice > 90:
            gaps.append(f"long notice ({notice}d)")
        if inactive > 90:
            gaps.append(f"inactive {inactive}d")
        if not sig.get("open_to_work_flag"):
            gaps.append("not flagged open to work")

        # Skill gap — only when genuinely low
        if rs.skill < 0.35:
            gaps.append(f"low JD skill overlap (skills {rs.skill:.2f})")
        elif not top_skills:
            gaps.append("limited JD skill overlap")

        # Guarantee at least one specific gap — no vague fallback phrases
        if not gaps:
            if yoe > yoe_max:
                gaps.append(f"{yoe:.0f}yr above {yoe_max}yr ceiling")
            elif not in_target and not relocate:
                gaps.append(f"non-target city ({location}), not open to relocate")
            elif not in_target:
                gaps.append(f"non-target city ({location})")
            elif rs.skill < 0.55:
                gaps.append(f"below-average JD skill match (skills {rs.skill:.2f})")
            else:
                gaps.append("below top-tier hybrid score threshold")

        skill_str = ", ".join(top_skills[:2]) if top_skills else "general ML background"
        body = (
            f"{title}, {exp_note}; skills: {skill_str}; {loc_str}. "
            f"Completing top 100; gaps: {'; '.join(gaps[:3])}."
        )

    body = _safe_truncate(body, max_body)
    return f"{body} {score_tag}"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER — fast streaming JSONL (plain or gzip)
# ══════════════════════════════════════════════════════════════════════════════

def load_candidates(path_str: str) -> list[dict]:
    """Stream JSONL (or .jsonl.gz); skip malformed lines silently."""
    path   = Path(path_str)
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[dict] = []
    with opener(path, "rt", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        for line in f:
            if line and line[0] == "{":
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Redrob Intelligent Candidate Ranker")
    ap.add_argument("--candidates",   required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--jd",           default=None,
                    help="Path to job description file (auto-discovered if omitted)")
    ap.add_argument("--out",          required=True,
                    help="Output CSV path for submission")
    ap.add_argument("--prefilter",    type=int,   default=300,
                    help="Number of candidates forwarded to semantic stage (default: 300)")
    ap.add_argument("--team-id",      default="team_xxx",
                    help="Team identifier (used in output metadata)")
    ap.add_argument("--preview",      type=int,   default=100,
                    help="Number of rows to print in console preview (default: 100)")
    ap.add_argument("--strong-pct",   type=float, default=60.0,
                    help="Percentile threshold for 'strong' tier (default: 60)")
    ap.add_argument("--adjacent-pct", type=float, default=30.0,
                    help="Percentile threshold for 'adjacent' tier (default: 30)")
    args = ap.parse_args()
    t0   = time.time()

    # ── Step 1: Parse JD ───────────────────────────────────────────────────────
    print("[1/5] Parsing job description...", flush=True)
    jd_text = load_jd(args.jd, args.candidates)
    jd      = parse_jd(jd_text)
    print(f"      YoE range     : {jd['yoe_min']}-{jd['yoe_max']} yrs", flush=True)
    print(f"      Target cities : {jd['target_cities'] or 'not specified'}", flush=True)
    print(f"      Target country: {jd['target_country'] or 'not specified'}", flush=True)
    print(f"      JD skills     : {len(jd['jd_skills'])} detected", flush=True)
    print(f"      Seniority     : {jd['seniority']}  |  "
          f"Work mode: {jd['work_mode']}  |  "
          f"Prefers product: {jd['prefers_product']}", flush=True)
    print(f"      Weights       : skill {W_SKILL} · career {W_CAREER} · "
          f"loc {W_LOC} · avail {W_BEHAV}", flush=True)

    # ── Step 2: Load candidates + honeypot filter ──────────────────────────────
    print(f"\n[2/5] Loading candidates from {args.candidates}...", flush=True)
    all_candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(all_candidates):,} candidates", flush=True)

    clean = [c for c in all_candidates if not is_honeypot(c)]
    print(f"      {len(clean):,} clean  "
          f"({len(all_candidates) - len(clean):,} honeypots removed)", flush=True)
    print(f"      Elapsed: {time.time() - t0:.1f}s", flush=True)

    # ── Step 3: Title pre-filter → two-stage rule scoring ─────────────────────
    print(f"\n[3/5] Fast pre-filter + rule scoring...", flush=True)
    relevant = [c for c in clean if is_relevant(c)]
    print(f"      Fast filter: {len(clean):,} → {len(relevant):,} relevant", flush=True)

    # Stage A: microsecond cheap-score sort
    def cheap_score(c: dict) -> float:
        p  = c["profile"]
        t  = (p.get("current_title", "") + " " + p.get("headline", "")).lower()
        yoe = p.get("years_of_experience", 0)
        fit    = (1.0 if TITLE_FIT_RE.search(t)
                  else 0.5 if ML_ADJACENT_TITLE_RE.search(t)
                  else 0.0)
        pen    = 0.3 if TITLE_PENALTY_RE.search(t)  else 0.0
        wr_pen = 0.4 if WRONG_DOMAIN_RE.search(t)   else 0.0
        jr_pen = (0.2 if JUNIOR_TITLE_RE.search(t)
                       and jd.get("seniority") in ("senior", "staff")
                  else 0.0)
        in_rng = 1.0 if jd["yoe_min"] <= yoe <= jd["yoe_max"] else 0.0
        return fit - pen - wr_pen - jr_pen + in_rng * 0.5

    relevant.sort(key=cheap_score, reverse=True)
    stage_b_pool = relevant[:3000]
    print(f"      Stage-A filter: {len(relevant):,} → "
          f"{len(stage_b_pool):,} for full scoring", flush=True)

    # Stage B: full rule_score on top 3000
    scored_pairs: list[tuple[RuleScores, dict]] = [
        (rule_score(c, jd), c)
        for c in tqdm(stage_b_pool, desc="      Rule scoring", ncols=80)
    ]
    scored_pairs.sort(key=lambda x: -x[0].total)

    pool        = [c for rs, c in scored_pairs[:args.prefilter]]
    pool_scores = {c["candidate_id"]: rs for rs, c in scored_pairs}

    best_rs  = scored_pairs[0][0].total
    worst_rs = scored_pairs[min(args.prefilter - 1, len(scored_pairs) - 1)][0].total
    print(f"      Top {args.prefilter} score range: {worst_rs:.3f} – {best_rs:.3f}", flush=True)
    print(f"      Elapsed: {time.time() - t0:.1f}s", flush=True)

    # ── Step 4: Bi-encoder + FAISS semantic search ────────────────────────────
    print(f"\n[4/5] Bi-encoder semantic search on top {len(pool)} "
          f"({BIENCODER_MODEL})...", flush=True)
    bi_model = SentenceTransformer(BIENCODER_MODEL)

    jd_emb = bi_model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    texts = [build_candidate_text(c) for c in pool]
    print(f"      Encoding {len(texts)} candidates...", flush=True)
    cand_embs = bi_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    dim   = cand_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cand_embs.astype(np.float32))
    distances, indices = index.search(jd_emb.astype(np.float32), len(pool))

    sem_raw  = distances[0]
    s_min, s_max = float(sem_raw.min()), float(sem_raw.max())
    sem_norm = (sem_raw - s_min) / (s_max - s_min + 1e-9)

    print(f"      Semantic range: {s_min:.3f} – {s_max:.3f}", flush=True)
    print(f"      Elapsed: {time.time() - t0:.1f}s", flush=True)

    # ── Step 5: Hybrid scoring → top-100 selection → output ───────────────────
    print(f"\n[5/5] Hybrid scoring and writing top 100...", flush=True)

    results: list[tuple[float, float, RuleScores, dict]] = []
    for rank_i, idx in enumerate(indices[0]):
        c      = pool[idx]
        sem_n  = float(sem_norm[rank_i])
        rs     = pool_scores[c["candidate_id"]]
        hybrid = W_SEMANTIC * sem_n + W_RULES * rs.total
        results.append((hybrid, sem_n, rs, c))

    # Deterministic sort: hybrid score (desc) then candidate_id (asc) as tiebreak
    results.sort(key=lambda x: (-x[0], x[3]["candidate_id"]))

    yoe_min = jd["yoe_min"]
    yoe_max = jd["yoe_max"]

    def yoe_in_range(c: dict) -> bool:
        return yoe_min <= c["profile"].get("years_of_experience", 0) <= yoe_max

    in_range  = [r for r in results if yoe_in_range(r[3])]
    out_range = [r for r in results if not yoe_in_range(r[3])]

    final = in_range[:100]
    if len(final) < 100:
        pad = 100 - len(final)
        final += out_range[:pad]
        print(f"  Note: padded {pad} out-of-range candidates", flush=True)

    # Re-sort after potential padding — ensure deterministic order
    final.sort(key=lambda x: (-x[0], x[3]["candidate_id"]))
    top100 = final[:100]

    # Tier thresholds derived from this run's actual score distribution
    top100_scores   = np.array([r[0] for r in top100])
    strong_thresh   = float(np.percentile(top100_scores, args.strong_pct))
    adjacent_thresh = float(np.percentile(top100_scores, args.adjacent_pct))
    print(
        f"  Tier thresholds (p{args.strong_pct:.0f}/p{args.adjacent_pct:.0f}): "
        f"strong >{strong_thresh:.4f}, adjacent >{adjacent_thresh:.4f}, "
        f"filler <={adjacent_thresh:.4f}",
        flush=True,
    )

    # Write submission CSV
    out_path = Path(args.out)
    rows: list[dict] = []
    for rank_idx, (hybrid, sem_n, rs, c) in enumerate(top100):
        tier_label = (
            "strong"   if hybrid >= strong_thresh   else
            "adjacent" if hybrid >= adjacent_thresh else
            "filler"
        )
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank":         rank_idx + 1,
            "score":        round(hybrid, 6),
            "reasoning":    build_reasoning(c, jd, sem_n, rs, tier_label),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n  Submission written → {out_path}", flush=True)
    print(f"  Total runtime     : {elapsed:.1f}s ({elapsed / 60:.1f} min)", flush=True)

    # Console preview
    preview_n = min(args.preview, 100)
    print(f"\n  Top {preview_n} preview:", flush=True)
    print(
        f"  {'Rank':>4}  {'CandID':>12}  {'Final':>6}  "
        f"{'Sem':>5}  {'Rules':>5}  {'Tier':>8}  Title",
        flush=True,
    )
    for rank_idx, (hybrid, sem_n, rs, c) in enumerate(top100[:preview_n]):
        tier_label = (
            "strong"   if hybrid >= strong_thresh   else
            "adjacent" if hybrid >= adjacent_thresh else
            "filler"
        )
        print(
            f"  {rank_idx + 1:>4}  {c['candidate_id']:>12}  {hybrid:.4f}  "
            f"{sem_n:.3f}  {rs.total:.3f}  {tier_label:>8}  "
            f"{c['profile']['current_title'][:30]}",
            flush=True,
        )

    print(f"\n  Validate: python data/validate_submission.py {out_path}", flush=True)


if __name__ == "__main__":
    main()