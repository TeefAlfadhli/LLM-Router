# ab_test.py
import os
import json
import time
import sqlite3
import warnings
from itertools import combinations
from datetime import datetime, UTC

import requests
from dotenv import load_dotenv

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

load_dotenv()
OPENROUTER_API_KEY = os.getenv("ROUTER_KEY")
DB_FILE = "routing_logs.db"

# ------------------------
# DB setup / insert helpers
# ------------------------
def init_ab_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ab_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            task TEXT,
            model_a TEXT,
            model_b TEXT,
            wins_a INTEGER,
            wins_b INTEGER,
            ties INTEGER,
            avg_score_a REAL,
            avg_score_b REAL,
            avg_latency_a REAL,
            avg_latency_b REAL,
            t_stat REAL,
            p_value REAL
        )
    """)
    conn.commit()
    conn.close()

def log_ab_summary(task, model_a, model_b, wins_a, wins_b, ties,
                   avg_score_a, avg_score_b, avg_latency_a, avg_latency_b,
                   t_stat, p_value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO ab_summaries (
            timestamp, task, model_a, model_b, wins_a, wins_b, ties,
            avg_score_a, avg_score_b, avg_latency_a, avg_latency_b, t_stat, p_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(UTC).isoformat(),
        task, model_a, model_b, wins_a, wins_b, ties,
        float(avg_score_a), float(avg_score_b),
        float(avg_latency_a), float(avg_latency_b),
        None if t_stat is None else float(t_stat),
        None if p_value is None else float(p_value),
    ))
    conn.commit()
    conn.close()

# ------------------------
# OpenRouter call
# ------------------------
def call_model(model: str, prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "LLM-AB-Test",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    start = time.time()
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=data)
    latency_ms = round((time.time() - start) * 1000)

    if resp.status_code == 200:
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            content = "[ERROR] malformed response"
        return content, latency_ms
    return f"[ERROR] {resp.status_code}", latency_ms

# ------------------------
# Prompts loader  (expects eval_prompts/prompts.json)
# {
#   "qa": ["...", "..."],
#   "summarization": ["...", "..."],
#   "code": ["...", "..."]
# }
# ------------------------
def load_prompts():
    path = os.path.join("eval_prompts", "prompts.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Normalize to lists
            return {
                "qa": list(data.get("qa", [])),
                "summarization": list(data.get("summarization", [])),
                "code": list(data.get("code", [])),
            }
    except FileNotFoundError:
        return {"qa": [], "summarization": [], "code": []}

# ------------------------
# Autoscorer (simple & deterministic)
# ------------------------
def auto_score(resp: str, prompt: str, task: str) -> float:
    """Return a float in [0,5]. Very simple heuristic, task-aware."""
    r = (resp or "").lower()
    p = (prompt or "").lower()

    # Penalize obvious API errors
    if r.startswith("[error]"):
        return 1.0

    # Task-specific quick checks
    if task == "qa":
        # Reward short, direct answers (<= 40 words) and presence of named-entity-ish tokens
        words = r.split()
        brevity = 1.0 if len(words) <= 40 else max(0.2, 40 / max(1, len(words)))
        has_capitalized = any(w[:1].isupper() for w in resp.split())
        return 3.0 * brevity + (2.0 if has_capitalized else 1.0)

    if task == "summarization":
        # Reward compression vs prompt length and presence of connectives
        rw = len(r.split())
        pw = len(p.split())
        compression = min(1.0, pw / max(1, 3 * max(1, rw)))
        connective = any(k in r for k in ["overall", "in summary", "key", "main"])
        return 3.0 * compression + (2.0 if connective else 1.0)

    if task == "code":
        # Reward presence of code fences / function keywords
        hints = sum(kw in r for kw in ["```", "def ", "function ", "return "])
        return min(5.0, 1.5 + hints)

    # Default
    return 3.0

def safe_ttest(scores_a, scores_b):
    """Guarded Welch t-test; returns (t_stat, p_val) or (None, None)."""
    if not SCIPY_OK:
        return (None, None)
    if len(scores_a) < 2 or len(scores_b) < 2:
        return (None, None)
    # If variance is zero in both, test is undefined
    var_a = 0.0 if len(scores_a) == 0 else float((sum((x - (sum(scores_a)/len(scores_a)))**2 for x in scores_a)))
    var_b = 0.0 if len(scores_b) == 0 else float((sum((x - (sum(scores_b)/len(scores_b)))**2 for x in scores_b)))
    if var_a == 0.0 and var_b == 0.0:
        return (None, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress precision-loss warnings
        t_stat, p_val = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    try:
        return (None if t_stat is None else float(t_stat),
                None if p_val is None else float(p_val))
    except Exception:
        return (None, None)

# ------------------------
# All-vs-all for a task
# ------------------------
def run_ab_tests(task: str, models: list[str], prompts: list[str]):
    if not prompts:
        print(f"No prompts found for task {task}")
        return

    for model_a, model_b in combinations(models, 2):
        wins_a = wins_b = ties = 0
        scores_a, scores_b = [], []
        lat_a, lat_b = [], []

        for prompt in prompts:
            # Call both models
            resp_a, la = call_model(model_a, prompt)
            resp_b, lb = call_model(model_b, prompt)

            # Score
            sa = auto_score(resp_a, prompt, task)
            sb = auto_score(resp_b, prompt, task)

            scores_a.append(sa)
            scores_b.append(sb)
            lat_a.append(la)
            lat_b.append(lb)

            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            else:
                ties += 1

        # Averages
        avg_sa = sum(scores_a) / len(scores_a)
        avg_sb = sum(scores_b) / len(scores_b)
        avg_la = round(sum(lat_a) / len(lat_a), 1)
        avg_lb = round(sum(lat_b) / len(lat_b), 1)

        # Guarded t-test
        t_stat, p_val = safe_ttest(scores_a, scores_b)

        # Store in DB
        log_ab_summary(
            task, model_a, model_b,
            wins_a, wins_b, ties,
            avg_sa, avg_sb,
            avg_la, avg_lb,
            t_stat, p_val
        )

# ------------------------
# Main
# ------------------------
def main():
    init_ab_db()
    prompts_by_task = load_prompts()

    tasks_and_models = {
        "qa": [
            "google/gemini-2.5-flash",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
        ],
        "summarization": [
            "anthropic/claude-3-opus",
            "openai/gpt-4",
            "mistralai/mistral-7b-instruct",
        ],
        "code": [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "inception/mercury-coder",
        ],
    }

    for task, model_list in tasks_and_models.items():
        prompts = prompts_by_task.get(task, [])
        run_ab_tests(task, model_list, prompts)

if __name__ == "__main__":
    main()