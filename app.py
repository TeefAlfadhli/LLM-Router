from flask import Flask, render_template, request
from dotenv import load_dotenv
import requests
import os
import time
import json
import sqlite3
from datetime import datetime
from statistics import mean

load_dotenv()
OPENROUTER_API_KEY = os.getenv("ROUTER_KEY")
app = Flask(__name__)

# ------------------------
# DATABASE SETUP FOR LOGS
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "routing_logs.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prompt TEXT,
            task_type TEXT,
            optimize_for TEXT,
            model TEXT,
            latency_ms INTEGER,
            cost REAL,
            total_tokens INTEGER
        )
    """)
    # A/B results table (needed for dashboard)
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

def log_request(prompt, task_type, optimize_for, model, latency_ms, cost, total_tokens):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO logs (timestamp, prompt, task_type, optimize_for, model, latency_ms, cost, total_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), prompt, task_type, optimize_for, model, latency_ms, cost, total_tokens))
    conn.commit()
    conn.close()

# Initialize database
init_db()

# ------------------------
# API CALL FUNCTION
# ------------------------
def call_openrouter(model, prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "LLM-Router-App"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }

    start_time = time.time()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000)

    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        usage = response.json().get("usage", {})
        return content, latency_ms, usage
    else:
        return f"[ERROR] {response.status_code}: {response.text}", latency_ms, {}

# ------------------------
# PROMPT CLASSIFICATION
# ------------------------
def classify_prompt(prompt):
    prompt_lower = prompt.lower()
    if "summarize" in prompt_lower:
        return "summarization"
    elif "code" in prompt_lower or "function" in prompt_lower:
        return "code"
    elif any(q in prompt_lower for q in ["who", "what", "when", "where"]):
        return "qa"
    else:
        return "general"

# ------------------------
# MODEL LOADING & SELECTION
# ------------------------
def load_models_for(task_type):
    try:
        with open(os.path.join(BASE_DIR, f"{task_type}.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def select_best_model(models, optimize_for, task_type):
    # Hardcoded routing logic
    if task_type == "code":
        if optimize_for == "cost":
            return "openai/gpt-3.5-turbo"
        elif optimize_for == "latency":
            return "inception/mercury-coder"
        elif optimize_for == "quality":
            return "openai/gpt-4"

    elif task_type == "summarization":
        if optimize_for == "cost":
            return "openai/gpt-3.5-turbo"
        elif optimize_for == "latency":
            return "mistralai/mistral-7b-instruct"
        elif optimize_for == "quality":
            return "anthropic/claude-3-opus"

    elif task_type == "qa":
        if optimize_for == "cost":
            return "qwen/qwen3-235b-a22b-2507"
        elif optimize_for == "latency":
            return "google/gemini-2.5-flash"
        elif optimize_for == "quality":
            return "openai/gpt-4"

    elif task_type == "general":
        if optimize_for == "cost":
            return "openai/gpt-3.5-turbo"
        elif optimize_for == "latency":
            return "mistralai/mistral-7b-instruct"
        elif optimize_for == "quality":
            return "openai/gpt-4"

    # Fallback to JSON sorting
    if models:
        sorted_models = sorted(models, key=lambda m: m[optimize_for])
        return sorted_models[0]["id"]

    # Final fallback
    return "openai/gpt-3.5-turbo"

# ------------------------
# FLASK ROUTES
# ------------------------
@app.route('/')
def llmUI():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_prompt():
    prompt = request.form.get('prompt')
    optimize_for = request.form.get('optimize_for')

    task_type = classify_prompt(prompt)
    models = load_models_for(task_type)
    model = select_best_model(models, optimize_for, task_type)

    response, latency_ms, usage = call_openrouter(model, prompt)
    model_display = model.split("/")[-1]

    # Get token count and estimate cost
    total_tokens = usage.get("total_tokens", 0)
    cost_per_token = 0
    for m in models:
        if m["id"] == model:
            cost_per_token = m["cost"] / 1000  # USD per token
            break

    estimated_cost = round(total_tokens * cost_per_token, 4)

    # Log the request
    log_request(prompt, task_type, optimize_for, model, latency_ms, estimated_cost, total_tokens)

    return render_template("response.html",
                           model=model_display,
                           response=response,
                           latency=latency_ms,
                           cost=f"${estimated_cost}")

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT timestamp, task_type, optimize_for, model, latency_ms, cost, total_tokens
        FROM logs
        ORDER BY id DESC
        LIMIT 100
    """)
    rows = cur.fetchall()

    total_requests = len(rows)
    total_cost = round(sum((r["cost"] or 0) for r in rows), 4) if rows else 0
    avg_latency = round(mean([r["latency_ms"] for r in rows]), 1) if rows else 0

    cur.execute("""
        SELECT model,
               COUNT(*) AS cnt,
               AVG(latency_ms) AS avg_latency,
               SUM(cost) AS sum_cost
        FROM logs
        GROUP BY model
        ORDER BY cnt DESC
        LIMIT 12
    """)
    per_model = cur.fetchall()

    cur.execute("""
        SELECT optimize_for, COUNT(*) AS cnt
        FROM logs
        GROUP BY optimize_for
        ORDER BY cnt DESC
    """)
    per_opt = cur.fetchall()

    # Pull latest A/B summaries (so dashboard shows them)
    cur.execute("""
        SELECT timestamp, task, model_a, model_b, wins_a, wins_b, ties,
               avg_score_a, avg_score_b, avg_latency_a, avg_latency_b,
               t_stat, p_value
        FROM ab_summaries
        ORDER BY id DESC
        LIMIT 20
    """)
    ab_rows = cur.fetchall()

    conn.close()

    times = [r["timestamp"] for r in rows][::-1]
    latencies = [r["latency_ms"] for r in rows][::-1]
    costs = [float(r["cost"] or 0) for r in rows][::-1]

    model_labels = [m["model"].split("/")[-1] for m in per_model]
    model_counts = [m["cnt"] for m in per_model]
    model_avg_lat = [round(m["avg_latency"] or 0, 1) for m in per_model]

    opt_labels = [o["optimize_for"] for o in per_opt]
    opt_counts = [o["cnt"] for o in per_opt]

    return render_template(
        "dashboard.html",
        total_requests=total_requests,
        total_cost=total_cost,
        avg_latency=avg_latency,
        times=times,
        latencies=latencies,
        costs=costs,
        model_labels=model_labels,
        model_counts=model_counts,
        model_avg_lat=model_avg_lat,
        opt_labels=opt_labels,
        opt_counts=opt_counts,
        rows=rows,
        ab_rows=ab_rows
    )

if __name__ == "__main__":
    app.run(debug=True)