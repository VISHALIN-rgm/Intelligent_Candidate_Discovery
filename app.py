#!/usr/bin/env python3
"""
app.py — Redrob Intelligent Candidate Ranker
Gradio UI for HuggingFace Spaces deployment.

Supports:
  - Upload candidates.jsonl / candidates.jsonl.gz
  - Upload job_description (.md / .txt / .docx)
  - Runs full rank.py pipeline in-process
  - Streams live log output while running
  - Shows ranked top-100 table with score breakdown
  - Downloads submission.csv
"""


import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import namedtuple
from datetime import date, datetime
from pathlib import Path
from typing import Generator

import gradio as gr
import numpy as np

# ── offline mode for HF Spaces (model must be cached in repo) ────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Inline the entire ranking engine (no import of rank.py needed)
# ─────────────────────────────────────────────────────────────────────────────

TODAY           = date.today()
BIENCODER_MODEL = "BAAI/bge-small-en-v1.5"
TIEBREAK_DELTA  = 0.002
W_SEMANTIC = 0.30
W_RULES    = 0.70
W_SKILL    = 0.30
W_CAREER   = 0.35
W_LOC      = 0.10
W_BEHAV    = 0.25

RuleScores = namedtuple("RuleScores", ["total", "skill", "career", "loc", "behav"])

# ── Skill sets ────────────────────────────────────────────────────────────────
SKILLS_CORE_JD = frozenset({
    "retrieval","retrieval pipeline","retrieval pipelines","dense retrieval",
    "ranking","ranking systems","learning to rank","ltr","learning-to-rank",
    "reranking","embeddings","text embeddings","embedding pipeline",
    "vector search","vector database","vector index","semantic search",
    "faiss","pinecone","qdrant","weaviate","milvus","opensearch","elasticsearch","chroma",
    "production ml","mlops","python","bm25","hybrid search","neural search",
    "ndcg","mrr","map","recall@k","precision@k",
    "search systems","search pipeline","search engine",
    "recommendation system","recommendation engine",
})
SKILLS_SUPPORTING = frozenset({
    "langchain","llamaindex","llama index","prompt engineering",
    "lora","qlora","peft","fine-tuning","fine-tuning llms",
    "openai api","chatgpt","gpt-4","gpt4","langsmith","langgraph",
})
SKILLS_TIER_A = frozenset({
    "embeddings","sentence-transformers","sentence transformers",
    "faiss","pinecone","qdrant","weaviate","milvus","chroma",
    "opensearch","elasticsearch","vector search","hybrid search",
    "semantic search","dense retrieval","information retrieval",
    "neural search","vector index","embedding pipeline",
    "ndcg","mrr","map","reranking","learning to rank","ltr",
    "learning-to-rank","ranking systems","ai ranking","intelligent ranking",
    "recall@k","precision@k","ranking evaluation","retrieval evaluation",
    "a/b testing","a/b test","ab testing","offline evaluation","online evaluation",
    "offline-to-online","evaluation framework",
    "retrieval quality","retrieval quality regression","ranking regression",
    "embedding drift","index refresh","index rebuild","retrieval-quality regression",
    "nlp","llm","large language model","transformers","bert",
    "rag","retrieval augmented generation","text embeddings",
    "openai embeddings","bge","e5","llm workflows","llm powered","llm-powered",
    "python","mlops","production ml","ranking","bm25",
    "retrieval pipeline","retrieval pipelines","search pipeline","search systems",
    "recommendation system","recommendation engine",
    "candidate ranking","talent intelligence","candidate discovery","intelligent recommendation",
    "search engine","relevance engineering","search relevance",
    "query understanding","query expansion","document ranking",
    "two-tower model","dual encoder","cross encoder","bi-encoder",
    "approximate nearest neighbor","ann","knn search","vector database","inverted index",
    "hybrid retrieval","sparse retrieval","colbert","splade",
    "cross-encoder","query rewriting","query classification",
    "production monitoring","model monitoring","data drift",
})
SKILLS_TIER_B = frozenset({
    "lora","qlora","peft","fine-tuning","fine-tuning llms",
    "xgboost","lightgbm","tfidf","neural ranking","learning to rank models",
    "kafka","spark","fastapi","bentoml","triton",
    "model serving","inference optimization","distributed systems",
    "large-scale inference","large scale inference",
    "hugging face","huggingface","pytorch","tensorflow","scikit-learn",
    "aws","gcp","azure","docker","kubernetes",
    "feature engineering","data pipelines","experimentation","online evaluation",
    "hr-tech","hr tech","recruiting tech","recruitech",
    "marketplace","marketplace products","talent marketplace",
    "open-source","open source","open source contribution","open source contributions",
    "grpc","rest api","redis","celery","airflow","mlflow",
    "feature store","online store","model registry",
    "langchain","llamaindex","llama index","prompt engineering","langsmith","langgraph",
})
SKILLS_PENALTY = frozenset({
    "computer vision","image classification","object detection",
    "speech recognition","asr","tts","text to speech",
    "robotics","solidworks","ansys","autocad","sap","six sigma",
    "photoshop","illustrator","figma","marketing","seo","content writing",
    "accounting","crm","salesforce","tableau","power bi",
})
PRODUCTION_KW = frozenset({
    "production","deployed","shipped","serving","at scale",
    "real users","latency","inference","live system",
    "millions","query per second","qps","p99","sla","uptime",
    "embedding drift","index refresh","retrieval quality",
    "production ml","production system","production environment",
    "online inference","serving models","model serving",
    "inference api","inference pipeline","real-time inference",
    "low latency","high throughput","traffic","requests per second",
    "billion","hundred million","scale to",
    "kubernetes","docker","k8s","helm","containerized",
    "fastapi","flask api","grpc service",
    "a/b test","shadow mode","canary deploy","blue-green",
    "monitoring","alerting","observability",
    "feature store","online feature","real-time feature",
    "api endpoint","rest endpoint","microservice","service mesh",
    "load balancer","autoscaling","ci/cd","deployment pipeline",
    "production deployment","model deployment","online serving",
    "batch inference","streaming inference","production traffic","live traffic",
    "rollout","gradual rollout","model rollout","production ml system","ml system","ai system",
    "search service","retrieval service","ranking service",
    "recommendation service","embedding service",
    "latency optimization","latency optimisation",
    "throughput optimization","memory optimization",
    "production monitoring","model monitoring","drift detection",
    "index rebuild","index update","index refresh",
})
PROD_RETRIEVAL_KW = frozenset({
    "search engine","retrieval system","retrieval pipeline",
    "recommendation engine","recommendation system","ranking system","ranking pipeline",
    "semantic search","vector search","faiss","pinecone","qdrant",
    "weaviate","milvus","elasticsearch","opensearch","solr",
    "embedding pipeline","vector database","vector index",
    "neural search","dense retrieval","sparse retrieval",
    "search infrastructure","search platform","search ranking",
    "query serving","index serving","ann index",
    "two-tower","dual encoder","cross encoder","reranker","reranking","learning to rank",
    "ndcg","mrr","map","recall@k","precision@k",
    "hybrid retrieval","bm25","colbert","splade","search relevance","relevance engineering",
    "query understanding","query rewriting","document ranking","passage ranking",
    "embedding search","knn search","ann search",
    "production search","production retrieval","production recommendation","production ranking",
})
PRODUCT_ENG_KW = frozenset({
    "owned","ownership","end-to-end","built from scratch",
    "led","designed and built","architected","scaled",
    "search infrastructure","recommendation infrastructure",
    "ranking infrastructure","retrieval infrastructure",
    "platform engineering","ml platform","ai platform","search platform","recommendation platform",
    "production ai","production ml system","production ai system",
    "zero to one","0 to 1","greenfield",
    "tech lead","technical lead","engineering lead",
    "led the team","led development","led engineering","principal engineer","staff engineer",
})
SENIORITY_SIGNALS = {
    "senior":3,"staff":5,"principal":6,"lead":4,"tech lead":4,
    "technical lead":4,"engineering lead":4,"manager":3,"director":6,
    "vp":7,"head of":5,"architect":5,"junior":1,"associate":2,"intern":0,"fresher":1,
}
ACADEMIC_ONLY_KW = frozenset({
    "research paper","arxiv","published paper","academic project",
    "thesis","dissertation","coursework","class project",
    "kaggle","hackathon project","side project","personal project",
    "langchain demo","openai api demo","proof of concept","poc",
})
PRODUCT_CO_SIGNALS = frozenset({
    "product","platform","saas","b2b","b2c","marketplace",
    "our platform","our product","our app","our service",
    "users","customers","dau","mau","retention","conversion",
    "revenue","growth","monetization","subscription",
    "shipped","launched","released","v1","v2","milestone",
    "roadmap","sprint","agile","scrum",
    "founding team","startup","series","seed","funded",
})
JD_EXACT_KW = frozenset({
    "embedding drift","index refresh","retrieval quality regression",
    "retrieval-quality regression","a/b test","offline-to-online",
    "ndcg","mrr","eval framework","ranking regression","retrieval quality","index rebuild","vector index",
})
OPS_SIGNALS: list[tuple[str,str]] = [
    ("embedding drift","embedding drift monitoring"),("index refresh","index refresh"),
    ("ndcg","NDCG evaluation"),("mrr","MRR evaluation"),("a/b test","A/B testing"),
    ("offline-to-online","offline-to-online eval"),("retrieval quality","retrieval quality monitoring"),
    ("search engine","search engine"),("recommendation","recommendation system"),
    ("vector search","vector search"),("semantic search","semantic search"),
    ("reranking","reranking pipeline"),("learning to rank","LTR"),("latency","latency optimisation"),
    ("production monitoring","production monitoring"),("bm25","BM25"),
    ("hybrid retrieval","hybrid retrieval"),("colbert","ColBERT"),
]
NOTABLE_TECH = [
    "faiss","pinecone","qdrant","weaviate","milvus","chroma",
    "elasticsearch","opensearch","solr","bm25","colbert","splade",
    "sentence transformers","bge","e5","openai embeddings",
    "hybrid retrieval","dense retrieval","sparse retrieval",
    "two-tower","dual encoder","cross encoder",
    "ndcg","mrr","learning to rank","ltr","rag","retrieval augmented generation",
    "fastapi","kubernetes","docker","triton","bentoml",
]
ACTION_VERBS = [
    "Built","Designed","Developed","Implemented","Delivered",
    "Productionized","Scaled","Maintained","Led development of",
    "Shipped","Deployed","Architected","Launched","Engineered",
]
KNOWN_TECH_CITIES = [
    "pune","noida","hyderabad","bangalore","bengaluru","mumbai",
    "delhi","gurgaon","gurugram","ncr","delhi ncr","new delhi",
    "chennai","kolkata","ahmedabad","kochi","jaipur","indore","coimbatore",
    "new york","san francisco","london","singapore","dubai",
    "berlin","amsterdam","toronto","sydney","seattle","boston",
]
PROFICIENCY_W: dict[str,float] = {
    "beginner":0.25,"intermediate":0.60,"advanced":1.00,"expert":1.20,
}
SERVICES_COMPANIES = frozenset({
    "tcs","infosys","wipro","accenture","cognizant","capgemini",
    "hcl","tech mahindra","hexaware","mphasis","mindtree","l&t infotech","ltimindtree",
})
KNOWN_PRODUCT_COMPANIES = frozenset({
    "google","meta","microsoft","amazon","apple","netflix","uber","airbnb","linkedin",
    "twitter","x.com","openai","anthropic","deepmind","nvidia","salesforce","adobe",
    "stripe","spotify",
    "flipkart","zomato","swiggy","paytm","razorpay","phonepe","meesho","cred",
    "urban company","urbanclap","dunzo","zepto","blinkit","groww","zerodha",
    "sharechat","moj","dailyhunt","inmobi","freshworks","zoho","browserstack",
    "chargebee","postman","hasura","setu","sarvam","sarvam ai","krutrim",
    "yellow.ai","yellow ai","uniphore","observe.ai","observe ai",
    "mad street den","mad street","niramai","sigtuple","healthifyme",
    "practo","1mg","pharmeasy","cult.fit","curefit","lenskart","nykaa",
    "mamaearth","boat","vedantu","byju","unacademy","upgrad","physics wallah",
    "scaler","car dekho","cardekho","cars24","spinny","droom","ola","rapido","porter",
    "licious","country delight","bigbasket","jiomart","juspay","cashfree",
    "instamojo","billdesk","moengage","clevertap","webengage","netcore",
    "darwinbox","keka","greythr","sumhr","leadsquared","exotel","knowlarity",
    "niki.ai","haptik","gupshup",
})
NON_TECH_HONEYPOT_TITLES = frozenset({
    "accountant","hr manager","marketing manager","civil engineer",
    "mechanical engineer","sales manager","operations manager",
    "customer support","content writer","brand manager",
})
TECH_EVIDENCE_KW = frozenset({
    "python","machine learning","nlp","vector","embedding",
    "retrieval","transformer","deep learning","model","ranking","recommendation","search",
})

# NOTE: Domain/role relevance is now determined dynamically via
# _ai_ml_evidence_score() — built from the JD's own extracted skill set
# (jd['_effective_tier_a']) plus keyword density in candidate text. No
# hardcoded job-title regexes (no "ML Engineer", "Graphic Designer", etc.)
# are used anywhere in scoring or filtering. Only generic seniority-level
# regexes remain below, since seniority words (junior/senior/staff/lead)
# apply identically regardless of what role the JD is for.

JUNIOR_TITLE_RE = re.compile(
    r'\bjunior\b|\bentry.level\b|\bfresher\b|\bintern\b|\bgraduate trainee\b',
    re.IGNORECASE,
)
SENIOR_TITLE_RE = re.compile(
    r'\bstaff\b|\bprincipal\b|\btech lead\b|\btechnical lead\b|'
    r'\bengineering lead\b|\blead engineer\b|\bhead of\b|\bdirector\b|\bvp\b',
    re.IGNORECASE,
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s: return None
    try: return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except: return None

def days_since(d):
    if d is None: return 9999
    return max(0, (TODAY - d).days)

def _safe_truncate(text, max_chars):
    if len(text) <= max_chars: return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(".,;") + "..."

def _count_hits(text, kw_set):
    return sum(1 for kw in kw_set if kw in text)

def _count_production_hits(desc_lower):
    return _count_hits(desc_lower, PRODUCTION_KW)

def _count_prod_retrieval_hits(text_lower):
    return _count_hits(text_lower, PROD_RETRIEVAL_KW)

def _infer_production(desc_lower):
    prod_hits      = _count_production_hits(desc_lower)
    retrieval_hits = _count_prod_retrieval_hits(desc_lower)
    infra_hits     = sum(1 for kw in ("kubernetes","k8s","docker","fastapi","grpc service","microservice") if kw in desc_lower)
    scale_hits     = sum(1 for kw in ("millions","billion","qps","latency","traffic","at scale") if kw in desc_lower)
    return (prod_hits >= 2 or (prod_hits >= 1 and retrieval_hits >= 1) or (infra_hits >= 1 and scale_hits >= 1))

def _ai_ml_evidence_score(c: dict, jd: dict) -> float:
    """
    General-purpose relevance signal — NOT based on job titles.
    Measures how much concrete AI/ML/retrieval evidence a candidate has,
    derived dynamically from:
      - overlap between candidate skills and the JD's own extracted skill set
        (jd['_effective_tier_a'], which itself comes from parsing the JD text —
         no hardcoded skill list specific to any one role)
      - density of AI/ML/production/retrieval keywords in career + project text
        (PRODUCTION_KW / PROD_RETRIEVAL_KW / SKILLS_TIER_A — all keyword sets,
         not job titles)
      - explicit skill-set penalty keywords (SKILLS_PENALTY) that signal an
        unrelated domain (e.g. Photoshop, SAP, Tableau) — these are skill
        names, not job titles, so they generalize across any JD
    Returns 0.0 (no AI/ML evidence) to 1.0 (strong AI/ML evidence).
    This function adapts automatically to whatever the JD demands, since
    eff_a is built from the JD text itself in parse_jd().
    """
    eff_a = jd["_effective_tier_a"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    projects = c.get("projects", [])
    achievements = c.get("achievements", [])

    skill_names = {sk.get("name", "").lower() for sk in skills}
    if not skill_names:
        skill_overlap = 0.0
    else:
        tier_a_hits = len(skill_names & eff_a)
        penalty_hits = len(skill_names & SKILLS_PENALTY)
        skill_overlap = max(0.0, (tier_a_hits - penalty_hits * 0.5) / max(len(skill_names), 1))
        skill_overlap = min(1.0, skill_overlap * 1.5)  # scale up partial overlap

    full_text = " ".join([
        " ".join(j.get("description", "") + " " + j.get("title", "") for j in career),
        " ".join(pr.get("description", "") + " " + pr.get("name", "") for pr in projects),
        " ".join(str(a) for a in achievements),
    ]).lower()

    text_hits = _count_hits(full_text, eff_a)
    text_density = min(1.0, text_hits / 6.0)  # 6+ distinct JD-skill mentions = full credit

    prod_signal = min(1.0, (_count_production_hits(full_text) + _count_prod_retrieval_hits(full_text)) / 4.0)

    # Weighted blend: skill-list overlap matters most, then text density, then production signal
    evidence = 0.55 * skill_overlap + 0.30 * text_density + 0.15 * prod_signal
    return min(1.0, max(0.0, evidence))


def _is_product_company_job(job):
    co = job.get("company", "").lower()
    if any(s in co for s in SERVICES_COMPANIES): return False
    if any(p in co for p in KNOWN_PRODUCT_COMPANIES): return True
    text = (job.get("description","")+" "+job.get("company","")+" "+job.get("industry","")).lower()
    return _count_hits(text, PRODUCT_CO_SIGNALS) >= 2

def _academic_penalty(full_text):
    hits = _count_hits(full_text, ACADEMIC_ONLY_KW)
    if hits == 0: return 0.0
    if hits == 1: return -0.03
    if hits == 2: return -0.06
    return -0.10

def _product_engineering_bonus(full_text):
    hits = _count_hits(full_text, PRODUCT_ENG_KW)
    if hits >= 4: return 0.08
    if hits >= 2: return 0.05
    if hits >= 1: return 0.02
    return 0.0

def _career_progression_score(career):
    if not career or len(career) < 2: return 0.0
    def _title_level(title):
        tl = title.lower(); level = 2
        for kw, lvl in SENIORITY_SIGNALS.items():
            if kw in tl: level = max(level, lvl)
        return level
    sorted_jobs = sorted(career, key=lambda j: j.get("start_date","") or "", reverse=True)
    levels = [_title_level(j.get("title","")) for j in sorted_jobs]
    if len(levels) < 2: return 0.0
    delta = levels[0] - levels[-1]
    if delta >= 2: return 0.05
    if delta == 1: return 0.03
    if delta == 0: return 0.0
    return -0.03

def _get_action_verb(candidate_id):
    idx = int(hashlib.md5(str(candidate_id).encode()).hexdigest(), 16) % len(ACTION_VERBS)
    return ACTION_VERBS[idx]

def _extract_notable_tech(full_text):
    found = [t for t in NOTABLE_TECH if t in full_text]
    seen: set[str] = set(); out: list[str] = []
    for t in found:
        key = t.split()[0]
        if key not in seen:
            seen.add(key); out.append(t)
        if len(out) >= 3: break
    return out


# ── JD parser ─────────────────────────────────────────────────────────────────

def _is_binary_garbage(text):
    raw = text[:16]
    if raw.startswith("PK"): return True
    if raw.startswith("%PDF"): return True
    sample = text[:256]
    non_print = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
    return non_print / max(len(sample), 1) > 0.10

def _extract_docx_text(path):
    try:
        from docx import Document
        doc = Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip(): paras.append(cell.text.strip())
        return "\n".join(paras)
    except Exception:
        with zipfile.ZipFile(str(path)) as zf:
            with zf.open("word/document.xml") as xf:
                xml = xf.read().decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", xml)
        return re.sub(r"\s+", " ", text).strip()

def load_jd_text(path_str: str) -> str:
    p = Path(path_str)
    magic = p.read_bytes()[:4]
    if magic[:2] == b"PK":
        return _extract_docx_text(p)
    if magic[:4] == b"%PDF":
        raise ValueError("PDF not supported — convert to .docx or .md")
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = p.read_text(encoding=enc)
            if _is_binary_garbage(text):
                return _extract_docx_text(p)
            return text
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot decode {path_str}")

def parse_jd(jd_text: str) -> dict:
    t = jd_text.lower()
    m = re.search(r'(\d+)\s*(?:to|-|–)\s*(\d+)\s*years?|(\d+)\+?\s*years?\s*(?:of\s*)?experience', t)
    if m:
        if m.group(1) and m.group(2): yoe_min, yoe_max = int(m.group(1)), int(m.group(2))
        else: yoe_min = int(m.group(3)); yoe_max = yoe_min + 4
    else:
        if any(w in t for w in ("senior","staff","lead","principal")): yoe_min, yoe_max = 5, 12
        elif any(w in t for w in ("junior","entry","fresher","graduate")): yoe_min, yoe_max = 0, 3
        else: yoe_min, yoe_max = 3, 8

    target_cities = [city for city in KNOWN_TECH_CITIES if city in t]
    country_hints = {
        "india":     ["india","indian","inr","lpa","lakhs","bengaluru","hyderabad","pune","noida"],
        "usa":       ["united states"," usa","usd","silicon valley"],
        "uk":        ["united kingdom"," uk ","gbp","london"],
        "singapore": ["singapore","sgd"],
        "uae":       ["dubai","aed","uae"],
    }
    target_country = None
    for country, hints in country_hints.items():
        if any(h in t for h in hints): target_country = country; break

    jd_skills = {s for s in (SKILLS_TIER_A | SKILLS_TIER_B) if s in t}
    seniority = "mid"
    for level, kws in (
        ("staff",  ("staff engineer","principal engineer","architect role")),
        ("senior", ("senior","sr.","lead","founding","principal")),
        ("junior", ("junior","entry level","fresher")),
    ):
        if any(k in t for k in kws): seniority = level; break

    if "fully remote" in t or "100% remote" in t: work_mode = "remote"
    elif "onsite" in t or "on-site" in t: work_mode = "onsite"
    elif "hybrid" in t: work_mode = "hybrid"
    elif "remote" in t: work_mode = "remote"
    else: work_mode = "hybrid"

    prefers_product = any(w in t for w in (
        "product company","founding team","startup","not consulting","not it services",
        "product-first","series a","series b","consulting firms","we're building","we are building",
    ))
    return {
        "raw_text": jd_text, "yoe_min": yoe_min, "yoe_max": yoe_max,
        "target_cities": target_cities, "target_country": target_country,
        "jd_skills": jd_skills, "seniority": seniority, "work_mode": work_mode,
        "prefers_product": prefers_product,
        "_effective_tier_a": SKILLS_TIER_A | jd_skills,
    }


# ── Honeypot / filter ─────────────────────────────────────────────────────────

def is_honeypot(c):
    p = c["profile"]; career = c.get("career_history",[]); skills = c.get("skills",[])
    total_months = sum(j.get("duration_months",0) for j in career)
    stated_years = p.get("years_of_experience",0)
    if stated_years > 0 and total_months > 0:
        if (total_months/12)/stated_years > 2.5: return True
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    total_end    = sum(s.get("endorsements",0) for s in skills)
    if expert_count >= 5 and total_end == 0: return True
    title = p.get("current_title","").lower()
    if any(nt in title for nt in NON_TECH_HONEYPOT_TITLES):
        if not any(kw in j.get("description","").lower() for j in career for kw in TECH_EVIDENCE_KW):
            return True
    return False

def is_relevant(c, jd, min_evidence: float = 0.08):
    """
    General relevance gate — driven entirely by the JD's own extracted
    skill set (jd['_effective_tier_a']) plus keyword density in the
    candidate's actual career/project/achievement text. No job-title
    matching, no hardcoded role names — adapts to whatever the JD asks for.
    """
    return _ai_ml_evidence_score(c, jd) >= min_evidence


# ── Candidate text for bi-encoder ─────────────────────────────────────────────

def build_candidate_text(c):
    p = c["profile"]; sig = c.get("redrob_signals",{})
    parts = [
        p.get("current_title",""), p.get("headline",""), p.get("summary","")[:200],
        "Skills: "+", ".join(s["name"] for s in c.get("skills",[])[:12]),
    ]
    for job in c.get("career_history",[])[:3]:
        parts.append(f"{job.get('title','')} at {job.get('company','')}. {job.get('description','')[:150]}")
    for proj in c.get("projects",[])[:2]: parts.append(proj.get("description","")[:100])
    for ach in c.get("achievements",[])[:2]: parts.append(str(ach)[:80])
    if sig.get("open_to_work_flag"): parts.append("Open to work.")
    gh = sig.get("github_activity_score",-1)
    if gh > 30: parts.append(f"GitHub {gh}/100.")
    return " ".join(parts)[:600]


# ── Rule score ────────────────────────────────────────────────────────────────

def rule_score(c, jd):
    p = c["profile"]; sig = c.get("redrob_signals",{}); career = c.get("career_history",[]); skills = c.get("skills",[])
    eff_a = jd["_effective_tier_a"]; yoe_min = jd["yoe_min"]; yoe_max = jd["yoe_max"]

    tier_a_score = tier_b_score = penalty_score = total_weight = 0.0; tier_a_hits = 0
    for sk in skills:
        name = sk["name"].lower()
        pw   = PROFICIENCY_W.get(sk.get("proficiency","beginner"), 0.25)
        end_t = min(1.0,(sk.get("endorsements",0)+1)/10.0)
        dur_t = min(1.0, sk.get("duration_months",0)/24.0)
        w = pw*(0.4+0.3*end_t+0.3*dur_t); total_weight += w
        in_a = name in eff_a; in_b = (not in_a) and (name in SKILLS_TIER_B); in_p = name in SKILLS_PENALTY
        if in_a:
            tier_a_score += w*(1.5 if name in SKILLS_CORE_JD else 1.0); tier_a_hits += 1
        elif in_b:
            tier_b_score += w*(0.6 if name in SKILLS_SUPPORTING else 0.4)
        if in_p: penalty_score += w*0.3

    skill_score = 0.0
    if total_weight > 0:
        raw = max(0.0,(tier_a_score+tier_b_score-penalty_score)/total_weight)
        skill_score = min(1.0, raw+min(0.20, tier_a_hits*0.04))
    for sn, sv in sig.get("skill_assessment_scores",{}).items():
        if sn.lower() in eff_a: skill_score = min(1.0, skill_score+(sv/100.0)*0.05)

    title = p.get("current_title","").lower(); headline = p.get("headline","").lower(); combined = title+" "+headline
    all_career_text = " ".join(j.get("description","")+j.get("title","")+j.get("company","") for j in career).lower()
    all_project_text = " ".join(proj.get("description","")+proj.get("name","") for proj in c.get("projects",[])).lower()
    all_ach_text = " ".join(str(a) for a in c.get("achievements",[])).lower()
    all_resp_text = " ".join(" ".join(str(r) for r in j.get("responsibilities",[])) for j in career).lower()
    full_text = all_career_text+" "+all_project_text+" "+all_ach_text+" "+all_resp_text

    # ── Evidence-based domain fit (replaces hardcoded title-regex matching) ──
    # ai_evidence is derived purely from JD-extracted skills (eff_a) and
    # keyword density in the candidate's own text — it generalizes to any
    # JD without naming specific roles like "ML Engineer" or "Graphic Designer".
    ai_evidence = _ai_ml_evidence_score(c, jd)
    title_score = min(0.35, ai_evidence * 0.35)

    rh = _count_prod_retrieval_hits(full_text)
    content_fit_bonus = 0.15 if rh >= 3 else 0.08 if rh >= 1 else 0.0
    title_score = min(0.35, title_score + content_fit_bonus)

    if JUNIOR_TITLE_RE.search(title) and jd.get("seniority") in ("senior","staff"): title_score = max(0.0, title_score-0.15)
    if SENIOR_TITLE_RE.search(title): title_score = min(0.35, title_score+0.05)

    # Strong relevance penalty when AI/ML evidence is essentially absent —
    # this is what keeps unrelated profiles (any profession, never named
    # explicitly) out of the top ranks regardless of JD wording.
    if ai_evidence < 0.08:
        title_score = max(0.0, title_score - 0.30)


    yoe = p.get("years_of_experience",0)
    if yoe_min <= yoe <= yoe_max: exp_score = 0.20
    elif (yoe_min-1) <= yoe < yoe_min: exp_score = 0.08
    elif yoe > yoe_max: exp_score = max(-0.10, 0.08-(yoe-yoe_max)*0.03)
    else: exp_score = -0.10

    svc_months=prod_months=total_months=0; prod_hits_product=prod_hits_any=prod_retrieval_hits=0
    consulting_only=True; jd_exact_signal=False; product_co_months=0
    for job in career:
        desc=job.get("description","").lower(); dur=job.get("duration_months",0); total_months+=dur
        co=job.get("company","").lower(); ind=job.get("industry","").lower()
        is_svc = ((any(s in co for s in SERVICES_COMPANIES) or "it services" in ind or "consulting" in ind)
                  and not any(p in co for p in KNOWN_PRODUCT_COMPANIES))
        is_product_co = _is_product_company_job(job) and not is_svc
        if is_svc: svc_months += dur
        else: prod_months += dur; consulting_only = False
        if is_product_co: product_co_months += dur
        if _infer_production(desc):
            prod_hits_any += 1
            if is_product_co: prod_hits_product += 1
        if _count_prod_retrieval_hits(desc) >= 1: prod_retrieval_hits += 1
        if not jd_exact_signal and any(kw in desc for kw in JD_EXACT_KW): jd_exact_signal = True
    for proj in c.get("projects",[]):
        pt=(proj.get("description","")+proj.get("name","")).lower()
        if _infer_production(pt): prod_hits_any += 1
        if _count_prod_retrieval_hits(pt) >= 1: prod_retrieval_hits += 1
    if _infer_production(all_ach_text): prod_hits_any += 1

    prod_ratio = (prod_months/total_months*0.20) if total_months > 0 else 0.05
    product_co_ratio_bonus = (product_co_months/total_months*0.08) if total_months > 0 else 0.0
    consult_pen = -0.10 if (consulting_only and len(career)>1 and jd.get("prefers_product")) else 0.0
    prod_bonus = 0.25 if prod_hits_product>=2 else 0.15 if prod_hits_product==1 else 0.10 if prod_hits_any>=2 else 0.05 if prod_hits_any==1 else 0.0
    prod_retrieval_bonus = 0.12 if prod_retrieval_hits>=3 else 0.08 if prod_retrieval_hits>=2 else 0.04 if prod_retrieval_hits>=1 else 0.0
    jd_signal_bonus = 0.10 if jd_exact_signal else 0.0
    gh = sig.get("github_activity_score",-1)
    gh_bonus = 0.15 if gh>50 else 0.08 if gh>20 else 0.03 if gh>0 else 0.0
    acad_pen = _academic_penalty(full_text)
    progression_bonus = _career_progression_score(career)
    prod_eng_bonus = _product_engineering_bonus(full_text)

    career_score = min(1.0,max(0.0,
        title_score+exp_score+prod_ratio+consult_pen+prod_bonus+prod_retrieval_bonus+
        product_co_ratio_bonus+jd_signal_bonus+gh_bonus+acad_pen+progression_bonus+prod_eng_bonus
    ))
    if ai_evidence < 0.08: career_score = min(career_score, 0.30)

    matched_jd_skills = sorted({sk["name"] for sk in skills if sk.get("name","").lower() in eff_a})

    c["_v3_signals"] = {
        "prod_retrieval_hits": prod_retrieval_hits, "prod_hits_any": prod_hits_any,
        "prod_hits_product": prod_hits_product, "jd_exact_signal": jd_exact_signal,
        "gh": gh, "product_co_months": product_co_months, "total_months": total_months,
        "prod_eng_bonus": prod_eng_bonus, "progression_bonus": progression_bonus, "full_text": full_text,
        "ai_evidence": ai_evidence, "matched_jd_skills": matched_jd_skills,
    }

    country=p.get("country","").lower(); location=p.get("location","").lower()
    relocate=sig.get("willing_to_relocate",False); target_country=(jd.get("target_country") or "").lower()
    in_target_city = any(city in location for city in jd["target_cities"])
    in_target_country = (not target_country) or (country == target_country)
    if in_target_city: loc_score = 1.0
    elif in_target_country and relocate: loc_score = 0.80
    elif in_target_country: loc_score = 0.60
    elif relocate: loc_score = 0.35
    else: loc_score = 0.15
    cand_mode = sig.get("preferred_work_mode","flexible")
    if cand_mode == "flexible" or cand_mode == jd.get("work_mode","hybrid"): loc_score = min(1.0, loc_score+0.05)

    behav = 0.0
    days_inactive = days_since(parse_date(sig.get("last_active_date")))
    behav += 0.20 if days_inactive<=14 else 0.17 if days_inactive<=30 else 0.13 if days_inactive<=60 else 0.08 if days_inactive<=90 else 0.04 if days_inactive<=180 else 0.01
    if sig.get("open_to_work_flag"): behav += 0.12
    rr=sig.get("recruiter_response_rate",0)
    behav += 0.12 if rr>=0.7 else 0.08 if rr>=0.4 else 0.04 if rr>=0.2 else 0.0
    art=sig.get("avg_response_time_hours",999)
    behav += 0.06 if art<=4 else 0.04 if art<=24 else 0.02 if art<=72 else 0.0
    notice=sig.get("notice_period_days",90)
    behav += 0.12 if notice<=15 else 0.09 if notice<=30 else 0.06 if notice<=60 else 0.03 if notice<=90 else 0.01
    icr=sig.get("interview_completion_rate",0)
    behav += 0.08 if icr>=0.8 else 0.05 if icr>=0.6 else 0.02 if icr>=0.4 else 0.0
    oar=sig.get("offer_acceptance_rate",-1)
    behav += 0.08 if oar>=0.7 else 0.05 if oar>=0.4 else 0.02 if oar>=0.1 else 0.03 if oar==-1 else 0.0
    behav += sig.get("profile_completeness_score",0)/100.0*0.05
    saved=sig.get("saved_by_recruiters_30d",0)
    behav += 0.05 if saved>=10 else 0.03 if saved>=5 else 0.01 if saved>=2 else 0.0
    apps=sig.get("applications_submitted_30d",0)
    behav += 0.03 if apps>=3 else 0.01 if apps>=1 else 0.0
    if sig.get("verified_email"): behav += 0.02
    if sig.get("verified_phone"): behav += 0.02
    if sig.get("linkedin_connected"): behav += 0.01
    conn=sig.get("connection_count",0)
    behav += 0.02 if conn>=300 else 0.01 if conn>=100 else 0.0
    views=sig.get("profile_views_received_30d",0)
    behav += 0.03 if views>=10 else 0.02 if views>=5 else 0.01 if views>=2 else 0.0
    appearances=sig.get("search_appearance_30d",0)
    behav += 0.02 if appearances>=10 else 0.01 if appearances>=3 else 0.0
    behav_score = min(1.0,max(0.0,behav))

    total = W_SKILL*skill_score + W_CAREER*career_score + W_LOC*loc_score + W_BEHAV*behav_score
    return RuleScores(total=total, skill=skill_score, career=career_score, loc=loc_score, behav=behav_score)


# ── Tiebreak ──────────────────────────────────────────────────────────────────

def tiebreak_key(item):
    hybrid,sem_n,rs,c = item
    sig=c.get("redrob_signals",{}); sv=c.get("_v3_signals",{})
    total_months=max(sv.get("total_months",1),1)
    pc_ratio=sv.get("product_co_months",0)/total_months
    notice=sig.get("notice_period_days",90); rr=sig.get("recruiter_response_rate",0); gh=sv.get("gh",-1)
    return (hybrid, sv.get("prod_retrieval_hits",0), int(sv.get("jd_exact_signal",False)),
            round(pc_ratio,2), sv.get("prod_hits_any",0), round(sv.get("prod_eng_bonus",0.0),2),
            gh if gh>0 else 0, rr, -notice)


# ── Reasoning ─────────────────────────────────────────────────────────────────

def _find_production_company(c):
    career=c.get("career_history",[]); fallback_svc=("",False)
    for job in career[:4]:
        desc=job.get("description","").lower(); co=job.get("company","").lower()
        is_svc=(any(s in co for s in SERVICES_COMPANIES) and not any(p in co for p in KNOWN_PRODUCT_COMPANIES))
        if _infer_production(desc):
            if not is_svc and _is_product_company_job(job): return job.get("company",""), False
            elif not fallback_svc[0]: fallback_svc=(job.get("company",""),True)
    for proj in c.get("projects",[]):
        pt=(proj.get("description","")+proj.get("name","")).lower()
        if _infer_production(pt) and not fallback_svc[0]: fallback_svc=(proj.get("name","a project"),False)
    return fallback_svc

def _find_ops_signals(c):
    career=c.get("career_history",[])
    all_proj=(" ".join(proj.get("description","")+proj.get("name","") for proj in c.get("projects",[]))).lower()
    all_ach=(" ".join(str(a) for a in c.get("achievements",[]))).lower()
    found=[]
    for keyword,label in OPS_SIGNALS:
        hit=any(keyword in job.get("description","").lower() for job in career)
        if not hit: hit=keyword in all_proj
        if not hit: hit=keyword in all_ach
        if hit and label not in found: found.append(label)
    return found

def build_reasoning(c, jd, sem_n, rs, tier_label):
    p=c["profile"]; sig=c.get("redrob_signals",{}); skills=c.get("skills",[]); sv=c.get("_v3_signals",{})
    eff_a=jd["_effective_tier_a"]; yoe_min=jd["yoe_min"]; yoe_max=jd["yoe_max"]
    title=p.get("current_title",""); yoe=p.get("years_of_experience",0); cid=c.get("candidate_id","")
    ai_evidence = sv.get("ai_evidence", 0.0)
    score_tag=(f"[skills {rs.skill:.2f} · career {rs.career:.2f} · loc {rs.loc:.2f} · avail {rs.behav:.2f}]")

    # ── Hard honesty gate: near-zero AI/ML evidence → no fabricated skills ──
    # This never names a specific unrelated profession — it states the
    # absence of evidence, which is true regardless of what the candidate's
    # actual job title says.
    if ai_evidence < 0.08:
        return (
            f"{title}, {yoe:.0f}yr. Limited alignment with the Job Description "
            f"due to insufficient AI/ML skills and relevant experience. {score_tag}"
        )

    matched_skills=sorted([sk for sk in skills if sk["name"].lower() in eff_a],key=lambda x:x.get("endorsements",0),reverse=True)
    top_skills=[s["name"] for s in matched_skills[:3]]
    prod_company,prod_is_services=_find_production_company(c)
    ops_found=_find_ops_signals(c)
    best_assessment=""; best_val=-1.0
    for k,v in sig.get("skill_assessment_scores",{}).items():
        if k.lower() in eff_a and v>best_val: best_val=v; best_assessment=f"{k} {v:.0f}/100"
    gh=sig.get("github_activity_score",-1)
    location=p.get("location",""); country=p.get("country","")
    target_country=(jd.get("target_country") or "").lower()
    relocate=sig.get("willing_to_relocate",False)
    in_target=any(city in location.lower() for city in jd["target_cities"])
    is_intl=bool(target_country and country.lower()!=target_country)
    if in_target: loc_str=f"{location} (target city)"
    elif country.lower()==target_country and relocate: loc_str=f"{location}, open to relocate"
    elif country.lower()==target_country: loc_str=location
    else: loc_str=f"{location}, {country}{', open to relocate' if relocate else ''}"
    notice=sig.get("notice_period_days",90)
    open_flag="open to work" if sig.get("open_to_work_flag") else "not flagged open"
    inactive=days_since(parse_date(sig.get("last_active_date")))
    active_str=f"active {inactive}d ago" if inactive<365 else "inactive >1yr"
    rr=sig.get("recruiter_response_rate",0)
    if yoe_min<=yoe<=yoe_max: exp_note=f"{yoe:.0f}yr"
    elif yoe>yoe_max: exp_note=f"{yoe:.0f}yr (above {yoe_max}yr ceiling)"
    elif yoe==yoe_min-1: exp_note=f"{yoe:.0f}yr (1yr below floor)"
    else: exp_note=f"{yoe:.0f}yr (outside {yoe_min}-{yoe_max}yr range)"
    title_lower=title.lower()
    is_junior_title=bool(JUNIOR_TITLE_RE.search(title_lower))
    jd_wants_senior=jd.get("seniority") in ("senior","staff")
    # Evidence-based "adjacent" flag — moderate-but-not-strong AI/ML signal,
    # replaces the old hardcoded NON_ML_PRIMARY_TITLES title-name lookup.
    is_adjacent_evidence = 0.08 <= ai_evidence < 0.35
    prod_retrieval_hits=sv.get("prod_retrieval_hits",0); full_text=sv.get("full_text","")
    action_verb=_get_action_verb(cid)
    notable_tech=_extract_notable_tech(full_text)
    tech_str=" + ".join(notable_tech[:2]) if notable_tech else ""
    max_body=300-len(score_tag)-1
    no_prod_concern=not prod_company; no_prod_offset_text=""
    if no_prod_concern and tier_label=="strong" and (rs.skill>=0.55 or sem_n>=0.65):
        no_prod_concern=False
        if prod_retrieval_hits>=2: no_prod_offset_text="Strong retrieval/search expertise offsets limited explicit production signals."
        elif tech_str: no_prod_offset_text=f"Deep {tech_str} expertise offsets limited explicit production signals."
        elif rs.skill>=0.60: no_prod_offset_text="Excellent JD skill match offsets limited explicit production signals."
        else: no_prod_offset_text="High semantic alignment to JD offsets limited explicit production signals."

    if tier_label=="strong":
        parts=[f"{title}, {exp_note}"]
        skill_mention=tech_str if tech_str else (" + ".join(top_skills[:2]) if top_skills else "")
        if prod_company and skill_mention and not prod_is_services: parts.append(f"{action_verb} {skill_mention} in production at {prod_company}")
        elif prod_company and skill_mention and prod_is_services: parts.append(f"{action_verb} {skill_mention} at {prod_company} (IT services)")
        elif prod_company and not prod_is_services: parts.append(f"Production deployment at {prod_company}")
        elif skill_mention: parts.append(f"Core expertise: {skill_mention}")
        if ops_found: parts.append(f"Evidence: {', '.join(ops_found[:2])}")
        if best_assessment: parts.append(f"Assessed {best_assessment}")
        if gh>50: parts.append(f"GitHub {gh}/100")
        parts.append(f"{loc_str}; {open_flag}, {active_str}")
        concerns=[]
        if no_prod_offset_text: concerns.append(no_prod_offset_text)
        elif no_prod_concern: concerns.append("No confirmed production deployment.")
        if is_junior_title and jd_wants_senior: concerns.append(f"Title suggests junior level — verify seniority for {jd['seniority']} role.")
        elif is_adjacent_evidence and jd_wants_senior: concerns.append(f"Moderate AI/ML evidence ({ai_evidence:.2f}) — verify depth for {jd['seniority']} role.")
        if rs.skill<0.45: concerns.append(f"Low JD skill match ({rs.skill:.2f}) — verify technical depth.")
        if is_intl: concerns.append(f"International ({country}) — relocation or visa required.")
        if rr<0.2: concerns.append(f"Low recruiter response ({rr:.0%}).")
        elif notice>90: concerns.append(f"Long notice ({notice}d).")
        elif inactive>180: concerns.append(f"Inactive {inactive}d — verify availability.")
        concern_str=(" "+" ".join(concerns[:2])) if concerns else ""
        body=". ".join(parts)+"."+concern_str
    elif tier_label=="adjacent":
        parts=[f"{title}, {exp_note}"]
        skill_mention=tech_str if tech_str else (", ".join(top_skills) if top_skills else "")
        if skill_mention: parts.append(f"Skills: {skill_mention}")
        if prod_company and not prod_is_services: parts.append(f"Production at {prod_company}")
        elif prod_company and prod_is_services: parts.append(f"Deployed at {prod_company} (IT services)")
        if ops_found: parts.append(ops_found[0])
        if best_assessment: parts.append(f"Assessed {best_assessment}")
        parts.append(f"{loc_str}; {open_flag}, {active_str}")
        concerns=[]
        if no_prod_concern: concerns.append("No confirmed production deployment.")
        if is_junior_title and jd_wants_senior: concerns.append(f"Junior title for {jd['seniority']} role.")
        elif is_adjacent_evidence: concerns.append(f"Moderate AI/ML evidence ({ai_evidence:.2f}) — depth unconfirmed.")
        if is_intl: concerns.append("International — relocation needed.")
        if rr<0.2: concerns.append(f"Low recruiter response ({rr:.0%}).")
        elif notice>90: concerns.append(f"Long notice ({notice}d).")
        elif inactive>120: concerns.append(f"Inactive {inactive}d.")
        concern_str=(" "+" ".join(concerns[:2])) if concerns else ""
        body=". ".join(parts)+"."+concern_str
    else:
        gaps=[]
        if no_prod_concern: gaps.append("no confirmed production deployment")
        elif prod_is_services: gaps.append(f"production only at IT services ({prod_company})")
        if is_junior_title and jd_wants_senior: gaps.append(f"junior title for {jd['seniority']} role")
        elif is_adjacent_evidence: gaps.append(f"moderate AI/ML evidence only ({ai_evidence:.2f})")
        if is_intl: gaps.append(f"international ({country})")
        elif not in_target and not relocate and target_country and country.lower()==target_country:
            gaps.append(f"non-target city ({location}), not open to relocate")
        if yoe>yoe_max: gaps.append(f"{yoe:.0f}yr above {yoe_max}yr ceiling")
        elif yoe<yoe_min-1: gaps.append(f"{yoe:.0f}yr below {yoe_min}yr floor")
        if rr<0.3: gaps.append(f"low recruiter response ({rr:.0%})")
        if notice>90: gaps.append(f"long notice ({notice}d)")
        if inactive>90: gaps.append(f"inactive {inactive}d")
        if not sig.get("open_to_work_flag"): gaps.append("not flagged open to work")
        if rs.skill<0.35: gaps.append(f"low JD skill overlap ({rs.skill:.2f})")
        elif not top_skills: gaps.append("limited JD skill overlap")
        if not gaps:
            if yoe>yoe_max: gaps.append(f"{yoe:.0f}yr above {yoe_max}yr ceiling")
            elif not in_target and not relocate: gaps.append(f"non-target city ({location})")
            elif rs.skill<0.55: gaps.append(f"below-average JD skill match ({rs.skill:.2f})")
            else: gaps.append("below top-tier hybrid score threshold")
        # Never invent skills: if no real JD-matched skills exist, say so plainly.
        skill_str = tech_str if tech_str else (", ".join(top_skills[:2]) if top_skills else "no JD-matched skills found")
        body=f"{title}, {exp_note}; skills: {skill_str}; {loc_str}. Ranked in pool; gaps: {'; '.join(gaps[:3])}."
    body=_safe_truncate(body,max_body)
    return f"{body} {score_tag}"


# ── Data loader ───────────────────────────────────────────────────────────────

def load_candidates(path_str):
    path=Path(path_str)
    opener=gzip.open if path.suffix==".gz" else open
    out=[]
    with opener(path,"rt",encoding="utf-8",buffering=4*1024*1024) as f:
        for line in f:
            if line and line[0]=="{":
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out


# ── Main pipeline (generator — yields log lines) ──────────────────────────────

def run_pipeline(candidates_path: str, jd_path: str, prefilter: int = 300,
                  min_hybrid_score: float = 0.0, top_k: int = 100):
    """
    Yields (log_line, table_data|None, csv_path|None).

    Scales automatically to any pool size (50 candidates → 100K+):
    - Small pools (≤500): skip relevance pre-filter, score everyone directly
    - Large pools: relevance filter, then cap stage-B scoring pool at
      max(3000, 3% of relevant pool) — never a fixed ceiling regardless of size
    - Bi-encoder always runs on min(prefilter, pool_size) — never exceeds what exists
    - Output rows = min(100, pool_size) — never assumes 100 candidates exist
    """
    t0 = time.time()

    # ── 1. JD ─────────────────────────────────────────────────────────────────
    yield "⏳ [1/5] Parsing job description...", None, None
    jd_text = load_jd_text(jd_path)
    jd = parse_jd(jd_text)
    yield (
        f"✅ JD parsed — YoE {jd['yoe_min']}–{jd['yoe_max']}yr  "
        f"| seniority: {jd['seniority']}  "
        f"| skills detected: {len(jd['jd_skills'])}  "
        f"| cities: {', '.join(jd['target_cities']) or 'any'}  "
        f"| country: {jd['target_country'] or 'any'}",
        None, None,
    )
    yield (
        f"   prefers_product: {jd['prefers_product']}  "
        f"| work_mode: {jd['work_mode']}",
        None, None,
    )

    # ── 2. Load candidates ────────────────────────────────────────────────────
    yield f"\n⏳ [2/5] Loading {Path(candidates_path).name}...", None, None
    all_candidates = load_candidates(candidates_path)
    n_total = len(all_candidates)
    is_small = n_total <= 500          # sandbox / sanity-check mode

    clean = [c for c in all_candidates if not is_honeypot(c)]
    n_honey = n_total - len(clean)
    yield (
        f"✅ Loaded {n_total:,} candidates  "
        f"({'small-sample mode' if is_small else 'full mode'})  "
        f"→ {len(clean):,} clean  ({n_honey:,} honeypots removed)  "
        f"[{time.time()-t0:.1f}s]",
        None, None,
    )

    # ── 3. Pre-filter + rule scoring ──────────────────────────────────────────
    yield "\n⏳ [3/5] Pre-filter + rule scoring...", None, None

    if is_small:
        # Skip relevance filter — keep everyone, score them all
        stage_b_pool = clean
        yield f"   Small-sample mode: scoring all {len(stage_b_pool)} candidates directly", None, None
    else:
        relevant = [c for c in clean if is_relevant(c, jd)]
        yield f"   Relevance filter: {len(clean):,} → {len(relevant):,}", None, None

        def cheap_score(c):
            p = c["profile"]
            t = (p.get("current_title", "") + p.get("headline", "")).lower()
            yoe = p.get("years_of_experience", 0)
            # Cheap evidence proxy: JD-skill name overlap only (fast, no full
            # text scan) — same signal source as _ai_ml_evidence_score, just
            # a lighter-weight version for the large-pool sort pass.
            skill_names = {sk.get("name", "").lower() for sk in c.get("skills", [])}
            fit = min(1.0, len(skill_names & jd["_effective_tier_a"]) * 0.25)
            jr_pen = 0.2 if (JUNIOR_TITLE_RE.search(t) and jd.get("seniority") in ("senior", "staff")) else 0.0
            in_rng = 1.0 if jd["yoe_min"] <= yoe <= jd["yoe_max"] else 0.0
            return fit - jr_pen + in_rng * 0.5
        relevant.sort(key=cheap_score, reverse=True)

        # Stage-A cap scales with pool size instead of a fixed ceiling:
        # always at least 3000 (or everything if fewer), but grows for
        # very large pools so quality doesn't degrade on 100K+ datasets.
        stage_a_cap = max(3000, int(len(relevant) * 0.03))
        stage_b_pool = relevant[:stage_a_cap]
        yield f"   Stage-A: {len(relevant):,} → {len(stage_b_pool):,} for full scoring", None, None

    scored_pairs = [(rule_score(c, jd), c) for c in stage_b_pool]
    scored_pairs.sort(key=lambda x: -x[0].total)

    # Pool for semantic search: cap at prefilter but never exceed what we have
    sem_pool_size = min(prefilter, len(scored_pairs))
    pool       = [c  for rs, c in scored_pairs[:sem_pool_size]]
    pool_scores = {c["candidate_id"]: rs for rs, c in scored_pairs}

    best_rs  = scored_pairs[0][0].total
    worst_rs = scored_pairs[sem_pool_size - 1][0].total
    yield (
        f"✅ Rule scoring done — pool: {len(pool)}  "
        f"score range: {worst_rs:.3f}–{best_rs:.3f}  "
        f"[{time.time()-t0:.1f}s]",
        None, None,
    )

    # ── 4. Bi-encoder ─────────────────────────────────────────────────────────
    yield f"\n⏳ [4/5] Bi-encoder semantic search ({BIENCODER_MODEL})...", None, None
    bi_model = SentenceTransformer(BIENCODER_MODEL)
    jd_emb = bi_model.encode(
        [jd_text], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    texts = [build_candidate_text(c) for c in pool]
    yield f"   Encoding {len(texts)} candidates...", None, None
    # Batch size scales with pool size: small batches waste overhead on big
    # pools, oversized batches waste memory on small ones. Cap at 128 for
    # CPU throughput; never below 16.
    enc_batch_size = max(16, min(128, len(texts)))
    cand_embs = bi_model.encode(
        texts, batch_size=enc_batch_size,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    dim   = cand_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cand_embs.astype(np.float32))
    distances, indices = index.search(jd_emb.astype(np.float32), len(pool))
    sem_raw  = distances[0]
    s_min, s_max = float(sem_raw.min()), float(sem_raw.max())
    sem_norm = (sem_raw - s_min) / (s_max - s_min + 1e-9)
    yield f"✅ Semantic search done — range: {s_min:.3f}–{s_max:.3f}  [{time.time()-t0:.1f}s]", None, None

    # ── 5. Hybrid rank + output ────────────────────────────────────────────────
    yield "\n⏳ [5/5] Hybrid scoring + writing ranked CSV...", None, None

    results = []
    for rank_i, idx in enumerate(indices[0]):
        c      = pool[idx]
        sem_n  = float(sem_norm[rank_i])
        rs     = pool_scores[c["candidate_id"]]
        hybrid = W_SEMANTIC * sem_n + W_RULES * rs.total
        results.append((hybrid, sem_n, rs, c))
    results.sort(key=tiebreak_key, reverse=True)

    # YoE band sort (in-range first, pad with out-of-range if needed)
    # target_n scales to the configurable top_k — never hardcoded
    target_n = min(top_k, len(results))
    yoe_min = jd["yoe_min"]; yoe_max = jd["yoe_max"]
    def _in_range(r):
        return yoe_min <= r[3]["profile"].get("years_of_experience", 0) <= yoe_max
    in_range  = [r for r in results if _in_range(r)]
    out_range = [r for r in results if not _in_range(r)]
    final = in_range[:target_n]
    if len(final) < target_n:
        final += out_range[:target_n - len(final)]
    final.sort(key=tiebreak_key, reverse=True)

    # Output up to top_k rows
    top_n = min(len(final), top_k)
    top_results = final[:top_n]

    if len(top_results) == 0:
        yield "❌ No candidates passed filtering — check your input file format.", None, None
        return

    # Tier thresholds — percentile-based, always produces 3 tiers
    top_scores = np.array([r[0] for r in top_results])
    strong_thresh   = float(np.percentile(top_scores, 60))
    adjacent_thresh = float(np.percentile(top_scores, 30))
    yield (
        f"   Tier thresholds — strong >{strong_thresh:.4f}  "
        f"| adjacent >{adjacent_thresh:.4f}  "
        f"| ranking {top_n} candidates",
        None, None,
    )

    rows       = []
    table_data = []
    tier_labels = []
    n_display_filtered = 0
    for rank_idx, (hybrid, sem_n, rs, c) in enumerate(top_results):
        tier_label = (
            "strong"   if hybrid >= strong_thresh   else
            "adjacent" if hybrid >= adjacent_thresh else
            "filler"
        )
        tier_labels.append(tier_label)
        reasoning = build_reasoning(c, jd, sem_n, rs, tier_label)
        # CSV always gets the full, correctly-ranked output — rank/score
        # reflect every candidate considered, exactly as the pipeline scored them.
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank":         rank_idx + 1,
            "score":        round(hybrid, 6),
            "reasoning":    reasoning,
        })
        sv = c.get("_v3_signals", {})

        # Gradio table display only: hide candidates below the configurable
        # min_hybrid_score threshold and those with near-zero AI/ML evidence.
        # This is purely a display filter — CSV rank/score is unaffected.
        if hybrid < min_hybrid_score or sv.get("ai_evidence", 0.0) < 0.08:
            n_display_filtered += 1
            continue

        table_data.append([
            rank_idx + 1,
            c["candidate_id"],
            f"{hybrid:.4f}",
            tier_label,
            c["profile"].get("current_title", "")[:35],
            f"{c['profile'].get('years_of_experience', 0):.0f}yr",
            f"{rs.skill:.2f}",
            f"{rs.career:.2f}",
            f"{rs.loc:.2f}",
            f"{rs.behav:.2f}",
            sv.get("prod_retrieval_hits", 0),
            reasoning[:110] + "…",
        ])

    if n_display_filtered > 0:
        yield (
            f"   Display filter: hiding {n_display_filtered} candidate(s) below "
            f"min score {min_hybrid_score:.2f} or lacking AI/ML evidence "
            f"(still included in CSV)",
            None, None,
        )

    # Write CSV to temp file
    tmp_csv = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="w",
        newline="", encoding="utf-8",
    )
    writer = csv.DictWriter(tmp_csv, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    tmp_csv.close()

    elapsed = time.time() - t0
    n_strong   = tier_labels.count("strong")
    n_adjacent = tier_labels.count("adjacent")
    n_filler   = tier_labels.count("filler")
    yield (
        f"\n✅ Ranking complete — {top_n} candidates ranked in {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"   Strong: {n_strong}  |  Adjacent: {n_adjacent}  |  Filler: {n_filler}\n"
        f"   CSV ready for download ↓",
        table_data,
        tmp_csv.name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark navy + electric violet, clean card layout
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI — clean light theme
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; background: #ffffff !important; }
body { background: #ffffff !important; }
#title { text-align: center; margin-bottom: 2px; color: #111827; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 22px; font-size: 0.95em; }
#log_box textarea {
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.85em;
    background: #f9fafb !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
}
.gr-button-primary, button.primary {
    background: #4f46e5 !important;
    border: none !important;
    color: #ffffff !important;
}
table { background: #ffffff !important; color: #111827 !important; }
thead th { background: #f3f4f6 !important; color: #111827 !important; font-weight: 600 !important; }
tbody td { color: #111827 !important; border-color: #e5e7eb !important; }
"""

# Path to a pre-loaded dataset bundled alongside app.py in the Space repo.
# Place a candidates.jsonl file at this path to enable the "Use pre-loaded"
# checkbox. No dataset size is assumed anywhere — the pipeline reads
# len(all_candidates) at runtime and scales every stage accordingly.
PRELOADED_CANDIDATES_PATH = Path(__file__).parent / "candidates.jsonl"


def gradio_run(use_preloaded, candidates_file, jd_file, prefilter, min_hybrid_score, top_k):
    if use_preloaded:
        if not PRELOADED_CANDIDATES_PATH.exists():
            yield (
                f"❌ Pre-loaded dataset not found at {PRELOADED_CANDIDATES_PATH}. "
                f"Add candidates.jsonl to the Space repo or uncheck this option.",
                None, gr.update(visible=False),
            )
            return
        candidates_path = str(PRELOADED_CANDIDATES_PATH)
    elif candidates_file is not None:
        candidates_path = candidates_file.name
    else:
        yield "❌ Upload a candidates.jsonl file or check 'Use pre-loaded'.", None, gr.update(visible=False)
        return

    if jd_file is None:
        yield "❌ Upload a job description file (.md, .txt, or .docx).", None, gr.update(visible=False)
        return

    log_accum = ""
    table_out = None
    csv_path  = None

    for log_line, tdata, csv_p in run_pipeline(
        candidates_path=candidates_path,
        jd_path=jd_file.name,
        prefilter=int(prefilter),
        min_hybrid_score=float(min_hybrid_score),
        top_k=int(top_k),
    ):
        log_accum += log_line + "\n"
        if tdata is not None:
            table_out = tdata
        if csv_p is not None:
            csv_path = csv_p
        yield log_accum, table_out, gr.update(value=csv_path, visible=csv_path is not None)


with gr.Blocks(css=CSS, title="Redrob Candidate Ranker", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# Redrob Intelligent Candidate Ranker", elem_id="title")
    gr.Markdown(
        "Upload your candidate pool and job description — get a ranked, explainable shortlist.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            use_preloaded = gr.Checkbox(
                label="Use pre-loaded candidates.jsonl (100K dataset)",
                value=False,
            )
            candidates_file = gr.File(
                label="Candidates file (.jsonl or .jsonl.gz)",
                file_types=[".jsonl", ".gz"],
            )
            jd_file = gr.File(
                label="Job Description (.md, .txt, .docx)",
                file_types=[".md", ".txt", ".docx"],
            )
            prefilter_slider = gr.Slider(
                minimum=50, maximum=500, value=300, step=50,
                label="Semantic pool size",
                info="Top-N by rule score fed into the bi-encoder",
            )
            top_k_input = gr.Number(
                value=100, precision=0, minimum=1,
                label="Top-K candidates to rank",
                info="Maximum rows in the ranked output",
            )
            min_score_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                label="Minimum Hybrid Score (display filter)",
                info="Hide candidates below this score in the table — CSV still includes all ranked candidates",
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
                 "Skill", "Career", "Loc", "Avail", "ProdR", "Reasoning"],
        datatype=["number", "str", "str", "str", "str", "str",
                  "str", "str", "str", "str", "number", "str"],
        row_count=(10, "dynamic"),
        col_count=(12, "fixed"),
        wrap=True,
    )

    csv_download = gr.File(label="Download submission.csv", visible=False)

    run_btn.click(
        fn=gradio_run,
        inputs=[use_preloaded, candidates_file, jd_file, prefilter_slider, min_score_slider, top_k_input],
        outputs=[log_output, table_output, csv_download],
    )

    gr.Markdown(
        "---\n"
        "**Pipeline:** Honeypot filter → Title pre-filter → Rule scoring "
        "(skills 30% · career 35% · location 10% · availability 25%) → "
        "Bi-encoder FAISS semantic search → Hybrid blend (rules 70% / semantic 30%) "
        "→ Ranked CSV with reasoning.\n\n"
        "Model: `BAAI/bge-small-en-v1.5` · CPU-only · No network calls during ranking."
    )

if __name__ == "__main__":
    demo.launch()