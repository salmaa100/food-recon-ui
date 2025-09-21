# app.py
#one-file production-style app (edits: performance & fuzzy matching fixes)
# - FastAPI UI (light/dark) + Single Search + Batch + CSV Upload/Download
# - Live OpenFoodFacts fuzzy reconciliation
# - OpenRefine-compatible /reconcile API
# - CSV↔CSV comparison page

import io
import csv
import json
import math
import time
import uuid
import html
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.middleware.cors import CORSMiddleware

from rapidfuzz import fuzz, process
import textdistance
from functools import lru_cache

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_NAME = "Food Reconciliation — UI + API"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

# Matching knobs (tuned for typo tolerance + precision)
DEFAULT_TOP_N = 20          # UI default
API_PAGE_SIZE = 40          # how many candidates we fetch from OFF
MATCH_THRESHOLD = 60        # UI shows only >= this score
BRAND_BONUS = 10            # bonus if brand aligns

REQUEST_TIMEOUT = 10

# CSV constants
UPLOAD_EXPECTED_NAME_COLS = ["name", "product", "query", "item", "title", "label"]
UPLOAD_EXPECTED_BRAND_COLS = ["brand", "brands", "maker", "manufacturer"]

# In-memory stores
DOWNLOADS: Dict[str, bytes] = {}
STATE: Dict[str, Dict[str, Any]] = {}  # used for two-CSV compare workflow

# -----------------------------------------------------------------------------
# App init
# -----------------------------------------------------------------------------
app = FastAPI(title="Food Reconciliation API", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def normalize(s: str) -> str:
    s = _safe_str(s).lower()
    # keep letters/numbers and a few punctuation tokens; replace everything else with space
    s = "".join(ch if ch.isalnum() or ch in ("&", "-", "'", " ") else " " for ch in s)
    s = " ".join(s.split())
    return s

def keyboard_distance(a: str, b: str) -> float:
    """Damerau–Levenshtein normalized → 0..100 (higher = closer)"""
    a_n, b_n = normalize(a), normalize(b)
    if not a_n and not b_n:
        return 100.0
    # use normalized_distance (0..1) then convert to similarity 0..100
    dist = textdistance.damerau_levenshtein.normalized_distance(a_n, b_n)  # 0..1
    return round((1.0 - dist) * 100.0, 1)

def phonetic_score(a: str, b: str) -> float:
    """Metaphone / sound-based similarity -> 0..100.

    We try metaphone equality (cheap exact), then fallback to comparing Nysiis codes if metaphone fails.
    """
    a_n, b_n = normalize(a), normalize(b)
    if not a_n or not b_n:
        return 0.0
    try:
        ma = textdistance.metaphone(a_n)
        mb = textdistance.metaphone(b_n)
        if ma and mb and ma == mb:
            return 100.0
        # fallback: nysiis approximate similarity (0..1) -> 0..100
        na = textdistance.nysiis(a_n)
        nb = textdistance.nysiis(b_n)
        if na and nb:
            sim = 1.0 - textdistance.hamming.normalized_distance(na, nb) if len(na) == len(nb) else (1.0 - textdistance.levenshtein.normalized_distance(na, nb))
            return round(max(0.0, min(1.0, sim)) * 100.0, 1)
    except Exception:
        return 0.0
    return 0.0

def score_candidate(query_name: str, cand_name: str, query_brand: str = "", cand_brand: str = "", brand_weight: int = BRAND_BONUS) -> Tuple[float, Dict[str, Any]]:
    """Score a candidate using a weighted ensemble of fuzzy signals.

    Changes made to improve fuzzy tolerance & short-typo behavior:
    - Increased weight of keyboard_distance (Damerau–Levenshtein) so short typos like `mlk`→`milk` matter more.
    - Added a small short-edit bonus when two short strings differ by 1-2 edits.
    - Kept phonetic boost but added a nysiis fallback for small phonetic matches (e.g., cola vs kola).
    """
    qn = normalize(query_name)
    cn = normalize(cand_name)
    qb = normalize(query_brand)
    cb = normalize(cand_brand)

    # if either is empty, early return low score
    if not qn or not cn:
        return 0.0, {
            "WRatio": 0.0,
            "Partial": 0.0,
            "TokenSet": 0.0,
            "Keyboard": 0.0,
            "Phonetic": 0.0,
            "BrandBonus": 0,
            "ShortEditBonus": 0,
            "Final": 0.0,
        }

    # core fuzzy measures
    w = fuzz.WRatio(qn, cn)
    partial = fuzz.partial_ratio(qn, cn)
    token = fuzz.token_set_ratio(qn, cn)
    kd = keyboard_distance(qn, cn)
    ph = phonetic_score(qn, cn)

    # short-edit heuristic: if both strings short and edit distance small, give a boost
    short_edit_bonus = 0
    try:
        ed = textdistance.damerau_levenshtein.distance(qn, cn)
        # for very short queries, even distance 1 is usually a typo (mlk -> milk)
        if max(len(qn), len(cn)) <= 5:
            if ed == 0:
                short_edit_bonus = 0
            elif ed == 1:
                short_edit_bonus = 25
            elif ed == 2:
                short_edit_bonus = 12
        else:
            if ed == 1:
                short_edit_bonus = 8
    except Exception:
        short_edit_bonus = 0

    # changed weights: give more importance to keyboard (edit) and phonetic
    base = (
        0.30 * w +  # fuzzy whole-string (WRatio)
        0.15 * partial +
        0.15 * token +
        0.25 * kd +  # increased weight for Damerau-Levenshtein similarity
        0.15 * ph
    )

    brand_bonus = brand_weight if (qb and cb and (qb in cb or cb in qb)) else 0
    final = min(100.0, base + brand_bonus + short_edit_bonus)

    return round(final, 1), {
        "WRatio": round(w, 1),
        "Partial": round(partial, 1),
        "TokenSet": round(token, 1),
        "Keyboard": round(kd, 1),
        "Phonetic": round(ph, 1),
        "BrandBonus": brand_bonus,
        "ShortEditBonus": short_edit_bonus,
        "Final": round(final, 1),
    }

# ----------------------- Performance: OFF API cache ---------------------------
# Small LRU cache on OFF calls. Keyed by (query, brand, page_size)
@lru_cache(maxsize=2048)
def _off_cached(query: str, brand: Optional[str], page_size: int) -> str:
    params = {
        "search_simple": 1,
        "json": 1,
        "search_terms": query,
        "page_size": page_size,
    }
    if brand:
        params.update({
            "tagtype_0": "brands",
            "tag_contains_0": "contains",
            "tag_0": brand,
        })
    try:
        r = requests.get(OFF_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception:
        return json.dumps({"products": []})

def off_search_api(query: str, brand: Optional[str] = None, page_size: int = API_PAGE_SIZE) -> List[Dict[str, Any]]:
    raw = _off_cached(query, brand, page_size)
    try:
        data = json.loads(raw)
        products = data.get("products", [])
    except Exception:
        products = []

    out = []
    for p in products:
        out.append({
            "id": p.get("code", "") or p.get("_id", ""),
            "product_name": p.get("product_name") or p.get("generic_name") or "",
            "brands": p.get("brands", ""),
            "categories": p.get("categories", ""),
            "image_small_url": p.get("image_small_url", ""),
            "image_url": p.get("image_url", ""),
        })
    return [x for x in out if x["product_name"]]

def find_matches(query: str, brand: str = "", top_n: int = DEFAULT_TOP_N, page_size: int = API_PAGE_SIZE) -> List[Dict[str, Any]]:
    q = _safe_str(query)
    b = _safe_str(brand)
    if not q:
        return []

    # fetch candidates from OFF; use small page_size to reduce latency but allow caller to override
    cands = off_search_api(q, page_size=page_size)
    if b:
        # search also constrained by brand (may duplicate some results)
        cands += off_search_api(q, brand=b, page_size=page_size)

    # De-duplicate
    seen, uniq = set(), []
    for c in cands:
        key = (c.get("id", ""), c.get("product_name", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    scored = []
    for c in uniq:
        s, det = score_candidate(q, c.get("product_name", ""), b, c.get("brands", ""))
        scored.append({
            "id": c.get("id", ""),
            "name": c.get("product_name", ""),
            "brand": c.get("brands", ""),
            "score": s,
            "match": bool(s >= MATCH_THRESHOLD),
            "type": [{"id": "/product", "name": "Product"}],
            "_details": det,
            "image": c.get("image_small_url") or c.get("image_url") or "",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

# -----------------------------------------------------------------------------
# UI (No JSON shown to users) — theme + loading overlay + nav dropdown
# -----------------------------------------------------------------------------
BASE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0b1020; --panel:#111a35; --panel2:#0f1730; --text:#eaf0ff; --muted:#b9c5ff;
  --accent:#3b82f6; --accent2:#8b5cf6; --ok:#16a34a; --warn:#f59e0b; --border:#233160;
  --chip:#18224a; --input:#0d1633; --link:#93c5fd;
}
:root.light{
  --bg:#f7f9ff; --panel:#ffffff; --panel2:#ffffff; --text:#0b132b; --muted:#4b5563;
  --accent:#2563eb; --accent2:#7c3aed; --ok:#15803d; --warn:#b45309; --border:#e5e7eb;
  --chip:#f2f4f7; --input:#ffffff; --link:#1d4ed8;
}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);font-size:16px}
.container{max-width:1200px;margin:0 auto;padding:24px}
.header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px}
.brand{display:flex;align-items:center;gap:12px}
.brand h1{font-size:22px;margin:0}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{background:var(--chip);border:1px solid var(--border);color:var(--muted);padding:6px 12px;border-radius:999px;font-size:13px}
.theme{display:flex;align-items:center;gap:10px}
.toggle{appearance:none;width:48px;height:28px;border-radius:999px;background:var(--chip);border:1px solid var(--border);position:relative;cursor:pointer}
.toggle:checked::after{left:24px}
.toggle::after{content:"";position:absolute;top:3px;left:3px;width:22px;height:22px;border-radius:999px;background:linear-gradient(180deg,#fff,#ddd);box-shadow:0 2px 4px rgba(0,0,0,.25)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width: 980px){ .grid{grid-template-columns:1fr} }
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,.15)}
.card h2{margin:0 0 12px 0;font-size:22px}
.input, textarea, select{width:100%;padding:12px 14px;border:1px solid var(--border);border-radius:12px;background:var(--input);color:var(--text)}
.row{display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;margin-bottom:12px}
@media (max-width: 640px){ .row{grid-template-columns:1fr} .row>label{margin-bottom:2px} }
.btn{display:inline-block;background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;border-radius:12px;padding:12px 16px;color:#fff;font-weight:700;cursor:pointer;text-decoration:none}
.btn.secondary{background:var(--chip);color:var(--text);border:1px solid var(--border)}
.small{font-size:13px;color:var(--muted)}
.table{width:100%;border-collapse:collapse}
.table th,.table td{border-bottom:1px solid var(--border);padding:10px;text-align:left;vertical-align:top}
.table th{font-weight:700}
.score{font-weight:700}
.ok{color:var(--ok)}
.bad{color:#ef4444}
.kv{display:grid;grid-template-columns:200px 1fr;gap:10px}
.footer{opacity:.85;margin-top:18px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;background:var(--chip);border:1px solid var(--border);font-size:12px}
.img{width:36px;height:36px;object-fit:cover;border-radius:8px;border:1px solid var(--border)}
.nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.dropdown{position:relative;display:inline-block}
.dropdown button{border:1px solid var(--border);background:var(--chip);color:var(--text);padding:10px 12px;border-radius:10px;cursor:pointer}
.dropdown .menu{display:none;position:absolute;background:var(--panel2);border:1px solid var(--border);border-radius:12px;min-width:260px;z-index:10;padding:8px}
.dropdown:hover .menu{display:block}
.menu a{display:block;color:var(--text);text-decoration:none;padding:8px;border-radius:8px}
.menu a:hover{background:var(--chip)}
a{color:var(--link)}
/* Loading overlay */
#overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.35);backdrop-filter:blur(1px);z-index:1000;align-items:center;justify-content:center}
.spinner{border:4px solid rgba(255,255,255,.3);border-top:4px solid #fff;border-radius:50%;width:36px;height:36px;animation:spin 1s linear infinite;margin-right:12px}
@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
.overlay-box{display:flex;align-items:center;gap:12px;background:var(--panel);border:1px solid var(--border);padding:14px 18px;border-radius:12px;color:var(--text)}
</style>
<script>
(function(){
  const KEY='theme';
  const saved = localStorage.getItem(KEY);
  if(saved==='light'){ document.documentElement.classList.add('light'); }
  window.toggleTheme = function(){
    const el = document.documentElement;
    const isLight = el.classList.toggle('light');
    localStorage.setItem(KEY, isLight ? 'light' : 'dark');
  }
  // Loading overlay on all forms
  window.addEventListener('load', () => {
    const overlay = document.getElementById('overlay');
    document.querySelectorAll('form').forEach(f=>{
      f.addEventListener('submit', ()=>{
        if(overlay){ overlay.style.display='flex'; }
        f.querySelectorAll('button, input[type=submit]').forEach(b=>b.disabled=true);
      });
    });
  });
})();
</script>
"""

def nav(active: str="home") -> str:
    def act(k): return ' class="badge"' if k==active else ' class="badge" style="opacity:.7"'
    return f"""
    <div class="nav">
      <a href="/" {act("home")}>Home</a>
      <div class="dropdown">
        <button>Batch Tools ▾</button>
        <div class="menu">
          <a href="/ui/batch">Batch Reconcile</a>
          <a href="/ui/upload">Batch via CSV</a>
          <a href="/batch-compare">Compare Two CSVs</a>
        </div>
      </div>
    </div>
    """

def layout(body_html: str, active: str="home") -> HTMLResponse:
    top = f"""
    <!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{APP_NAME}</title>{BASE_CSS}</head>
    <body>
      <div id="overlay"><div class="overlay-box"><div class="spinner"></div><div><b>Running…</b><div class="small">Fetching candidates & scoring</div></div></div></div>
      <div class="container">
        <div class="header">
          <div class="brand">
            <div style="font-size:22px">🥗</div>
            <h1>{APP_NAME}</h1>
          </div>
          <div>{nav(active)}</div>
          <div class="theme">
            <span class="small">Theme</span>
            <input type="checkbox" class="toggle" onclick="toggleTheme()" aria-label="Toggle theme">
          </div>
        </div>
        {body_html}
        <div class="footer small">Live data from OpenFoodFacts • FastAPI • RapidFuzz • Accessibility-friendly UI</div>
      </div>
    </body></html>
    """
    return HTMLResponse(top)

# -----------------------------------------------------------------------------
# Home + Single search
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    body = f"""
    <div class="grid">
      <div class="card">
        <h2>🔎 Try Single Item</h2>
        <form method="post" action="/ui/search">
          <div class="row"><label>Product name</label><input class="input" name="query" placeholder="e.g., bread, milk, cola, chips" required></div>
          <div class="row"><label>Brand (optional)</label><input class="input" name="brand" placeholder="e.g., coca-cola, lays"></div>
          <div class="row"><label>Top results</label>
            <select name="top_n" class="input">
              {''.join(f'<option value="{n}" {"selected" if n==DEFAULT_TOP_N else ""}>{n}</option>' for n in [5,10,15,20,30])}
            </select>
          </div>
          <button class="btn" type="submit">Search</button>
        </form>
        <p class="small">Tip: Try misspellings — we’ll still find the closest real products😊. UI shows only matches ≥ {MATCH_THRESHOLD}%.</p>
      </div>

      <div class="card">
        <h2>📦 Batch Reconciliation</h2>
        <form method="post" action="/ui/reconcile">
          <div class="row"><label>Items</label>
            <textarea class="input" name="lines" rows="10" placeholder="milk\nbread\ncola, coca-cola\nchips, lays"></textarea>
          </div>
          <div class="row"><label>Top results</label>
            <select name="top_n" class="input">
              {''.join(f'<option value="{n}" {"selected" if n==DEFAULT_TOP_N else ""}>{n}</option>' for n in [5,10,15,20,30])}
            </select>
          </div>
          <button class="btn" type="submit">Reconcile</button>
        </form>
        <p class="small">Output shows Top-N per item with score, match flag and link. UI lists only ≥ {MATCH_THRESHOLD}%.</p>
      </div>
    </div>

    <br>

    <div class="card">
      <h2>🗂️ Batch via CSV (Upload & Download)</h2>
      <p class="small">Upload a CSV with columns like <span class="pill">name</span> and optional <span class="pill">brand</span>. We’ll clean, reconcile, and let you download results.</p>
      <form method="post" action="/ui/upload" enctype="multipart/form-data">
        <div class="row"><label>CSV file</label><input class="input" type="file" name="file" accept=".csv" required></div>
        <div class="row"><label>Top results</label>
          <select name="top_n" class="input">
            {''.join(f'<option value="{n}" {"selected" if n==DEFAULT_TOP_N else ""}>{n}</option>' for n in [5,10,15,20,30])}
          </select>
        </div>
        <button class="btn" type="submit">Upload & Reconcile</button>
      </form>
      <p class="small">We’ll show a cleaning log and a download link for your reconciled CSV.</p>
    </div>

    <br>

    <div class="card">
      <h2>🧩 OpenRefine / API</h2>
      <p class="small">Service manifest at <span class="pill">/reconcile</span>. Query via <span class="pill">POST /reconcile</span> with payload:
      <code>{{"queries":{{"q0":{{"query":"milk"}}}}}}</code></p>
      <a class="btn secondary" href="/docs">Open API Docs</a>
    </div>
    """
    return layout(body, active="home")

@app.post("/ui/search", response_class=HTMLResponse)
def ui_search(query: str = Form(...), brand: str = Form(""), top_n: int = Form(DEFAULT_TOP_N)):
    top_n = int(top_n) if str(top_n).isdigit() else DEFAULT_TOP_N
    results = [r for r in find_matches(query, brand, top_n=top_n, page_size=API_PAGE_SIZE) if r["score"] >= MATCH_THRESHOLD]

    rows = ""
    if not results:
        rows = "<tr><td colspan='6'>No results ≥ threshold.</td></tr>"
    else:
        for r in results:
            img = f"<img class='img' src='{r['image']}' alt='img'>" if r["image"] else ""
            rows += f"<tr><td>{img}</td><td>{html.escape(r['name'])}</td><td>{html.escape(r['brand'] or '-')}</td><td class='score ok'>{r['score']}</td><td>✅</td><td><a href='https://world.openfoodfacts.org/product/{r['id']}' target='_blank'>view</a></td></tr>"

    explain = ""
    if results:
        d = results[0].get("_details", {})
        explain = f"""
        <div class="kv small">
          <div>WRatio</div><div>{d.get('WRatio')}</div>
          <div>Partial</div><div>{d.get('Partial')}</div>
          <div>TokenSet</div><div>{d.get('TokenSet')}</div>
          <div>Keyboard</div><div>{d.get('Keyboard')}</div>
          <div>Phonetic</div><div>{d.get('Phonetic')}</div>
          <div>Brand Bonus</div><div>{d.get('BrandBonus')}</div>
          <div>ShortEdit</div><div>{d.get('ShortEditBonus')}</div>
          <div>Final</div><div><b>{d.get('Final')}</b></div>
        </div>
        """

    body = f"""
    <div class="card">
      <h2>Results for “{html.escape(query)}” {f"(brand: {html.escape(brand)})" if brand else ""}</h2>
      <table class="table">
        <thead><tr><th></th><th>Product</th><th>Brand</th><th>Score</th><th>Match?</th><th>Link</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <br><h3>Why the top match?</h3>{explain}
      <br><a class="btn" href="/">Back</a>
    </div>
    """
    return layout(body, active="home")

# -----------------------------------------------------------------------------
# Batch (textarea)
# -----------------------------------------------------------------------------
@app.post("/ui/reconcile", response_class=HTMLResponse)
def ui_reconcile(lines: str = Form(""), top_n: int = Form(DEFAULT_TOP_N)):
    top_n = int(top_n) if str(top_n).isdigit() else DEFAULT_TOP_N

    items: List[Tuple[str, str]] = []
    for raw in (lines or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if "," in raw:
            name, brand = raw.split(",", 1)
            items.append((name.strip(), brand.strip()))
        else:
            items.append((raw, ""))

    if not items:
        return layout("<div class='card'><h2>⚠️ No input</h2><p>Please paste at least one line.</p><a class='btn' href='/'>Back</a></div>", active="home")

    blocks = []
    for name, brand in items:
        results = [r for r in find_matches(name, brand, top_n=top_n, page_size=API_PAGE_SIZE) if r["score"] >= MATCH_THRESHOLD]
        rows = ""
        if not results:
            rows = "<tr><td colspan='6'>No results ≥ threshold.</td></tr>"
        else:
            for r in results:
                img = f"<img class='img' src='{r['image']}' alt='img'>" if r["image"] else ""
                rows += f"<tr><td>{img}</td><td>{html.escape(r['name'])}</td><td>{html.escape(r['brand'] or '-')}</td><td class='score ok'>{r['score']}</td><td>✅</td><td><a href='https://world.openfoodfacts.org/product/{r['id']}' target='_blank'>view</a></td></tr>"

        blocks.append(f"""
        <div class="card">
          <h3>🔗 {html.escape(name)} {f"(brand: {html.escape(brand)})" if brand else ""}</h3>
          <table class="table">
            <thead><tr><th></th><th>Product</th><th>Brand</th><th>Score</th><th>Match?</th><th>Link</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """)

    return layout("\n".join(blocks) + "<br><a class='btn' href='/'>Back</a>", active="home")

# -----------------------------------------------------------------------------
# Batch via CSV (Upload & Download)
# -----------------------------------------------------------------------------
def _detect_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    name_col = next((cols_lower[c] for c in cols_lower if c in UPLOAD_EXPECTED_NAME_COLS), None)
    brand_col = next((cols_lower[c] for c in cols_lower if c in UPLOAD_EXPECTED_BRAND_COLS), None)
    if not name_col:
        name_col = df.columns[0]
    if brand_col is None and len(df.columns) >= 2 and df.columns[1] != name_col:
        brand_col = df.columns[1]
    return name_col, brand_col

@app.get("/ui/upload", response_class=HTMLResponse)
def upload_page():
    body = f"""
    <div class="card">
      <h2>🗂️ Batch via CSV (Upload & Download)</h2>
      <form method="post" action="/ui/upload" enctype="multipart/form-data">
        <div class="row"><label>CSV file</label><input class="input" type="file" name="file" accept=".csv" required></div>
        <div class="row"><label>Top results</label>
          <select name="top_n" class="input">
            {''.join(f'<option value="{n}" {"selected" if n==DEFAULT_TOP_N else ""}>{n}</option>' for n in [5,10,15,20,30])}
          </select>
        </div>
        <button class="btn" type="submit">Upload & Reconcile</button>
      </form>
      <p class="small">We’ll clean, reconcile, and provide a downloadable CSV. UI shows only matches ≥ {MATCH_THRESHOLD}%.</p>
    </div>
    """
    return layout(body, active="home")

@app.post("/ui/upload", response_class=HTMLResponse)
async def ui_upload(file: UploadFile = File(...), top_n: int = Form(DEFAULT_TOP_N)):
    top_n = int(top_n) if str(top_n).isdigit() else DEFAULT_TOP_N

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
        except Exception:
            return layout("<div class='card'><h2>❌ Could not read CSV</h2><a class='btn' href='/'>Back</a></div>", active="home")

    name_col, brand_col = _detect_cols(df)

    # Cleaning log
    initial = len(df)
    df[name_col] = df[name_col].astype(str).map(_safe_str)
    if brand_col:
        df[brand_col] = df[brand_col].astype(str).map(_safe_str)

    df["__name_norm__"] = df[name_col].map(normalize)
    before_drop = len(df)
    df = df[df["__name_norm__"] != ""]
    dropped_blank = before_drop - len(df)
    df = df.drop_duplicates(subset=["__name_norm__", brand_col] if brand_col else ["__name_norm__"])
    deduped = before_drop - dropped_blank - len(df)

    # Reconcile
    results_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        q = _safe_str(row[name_col])
        b = _safe_str(row[brand_col]) if brand_col else ""
        matches = [m for m in find_matches(q, b, top_n=top_n, page_size=API_PAGE_SIZE) if m["score"] >= MATCH_THRESHOLD]
        top1 = matches[0] if matches else {}
        results_rows.append({
            "query": q,
            "brand": b,
            "match_id": top1.get("id", ""),
            "match_name": top1.get("name", ""),
            "match_brand": top1.get("brand", ""),
            "score": top1.get("score", ""),
            "is_match": top1.get("match", False),
            "topN_json": json.dumps([{ 
                "id": m["id"], "name": m["name"], "brand": m["brand"],
                "score": m["score"], "match": m["match"]
            } for m in matches], ensure_ascii=False),
        })

    out_df = pd.DataFrame(results_rows, columns=[
        "query", "brand", "match_id", "match_name", "match_brand", "score", "is_match", "topN_json"
    ])

    # Save downloadable CSV
    token = str(uuid.uuid4())
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    DOWNLOADS[token] = buf.getvalue().encode("utf-8")

    # Preview table
    head = out_df.head(20).to_dict(orient="records")
    rows_html = ""
    for r in head:
        link = f"<a href='https://world.openfoodfacts.org/product/{r.get('match_id')}' target='_blank'>view</a>" if r.get("match_id") else "—"
        rows_html += f"<tr><td>{html.escape(str(r['query']))}</td><td>{html.escape(str(r.get('brand','')))}</td><td class='score {'ok' if r.get('is_match') else 'bad'}'>{html.escape(str(r.get('score','')))}</td><td>{'✅' if r.get('is_match') else '—'}</td><td>{html.escape(str(r.get('match_name','')))}</td><td>{html.escape(str(r.get('match_brand','')))}</td><td>{link}</td></tr>"

    body = f"""
    <div class="card">
      <h2>✅ Upload processed</h2>
      <p class="small">Cleaning: start <b>{initial}</b>; removed blanks: <b>{dropped_blank}</b>; removed duplicates: <b>{deduped}</b>.</p>
      <a class="btn" href="/download?token={token}">Download CSV Results</a>
      <br><br>
      <table class="table">
        <thead><tr><th>Query</th><th>Brand</th><th>Score</th><th>Match?</th><th>Top match</th><th>Top brand</th><th>Link</th></tr></thead>
        <tbody>{rows_html or "<tr><td colspan='7'>No rows to preview.</td></tr>"}</tbody>
      </table>
      <br><a class="btn secondary" href="/">Back</a>
    </div>
    """
    return layout(body, active="home")

@app.get("/download")
def download(token: str):
    data = DOWNLOADS.get(token)
    if not data:
        return PlainTextResponse("Invalid or expired token", status_code=404)
    filename = f"reconciliation_results_{token[:8]}.csv"
    return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})

# -----------------------------------------------------------------------------
# Reconciliation API (OpenRefine-compatible)
# -----------------------------------------------------------------------------
@app.get("/reconcile")
def reconcile_meta():
    return {
        "name": "Food Reconciliation API",
        "identifierSpace": "https://world.openfoodfacts.org/",
        "schemaSpace": "http://schema.org/Product",
        "defaultTypes": [{"id": "/product", "name": "Product"}],
        "view": {"url": "https://world.openfoodfacts.org/product/{{id}}"},
    }

@app.post("/reconcile")
def reconcile_post(payload: Dict[str, Any]):
    """
    Accepts:
      {"queries": {"q0": {"query": "mlk", "brand": "..."}, "q1": {"query":"bred"}}}
    or directly:
      {"q0": {"query": "mlk"}}
    """
    queries = payload.get("queries")
    if queries is None:
        queries = payload
    if not isinstance(queries, dict):
        return JSONResponse({"error": "Invalid 'queries' format"}, status_code=400)

    out: Dict[str, Any] = {}
    for key, q in queries.items():
        if not isinstance(q, dict):
            continue
        query = _safe_str(q.get("query") or q.get("name") or q.get("q") or "")
        brand = _safe_str(q.get("brand") or "")
        top_n = int(q.get("limit") or DEFAULT_TOP_N)
        matches = find_matches(query, brand, top_n=top_n, page_size=API_PAGE_SIZE)
        clean = [{"id": m["id"], "name": m["name"], "score": m["score"], "match": m["match"], "type": m["type"]} for m in matches]
        out[key] = {"result": clean}
    return out

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

# ----------------------------------------------------------------------------
# CSV ↔ CSV Compare — separate page (dropdown in nav, same theme switcher)
# Workflow:
#   GET  /batch-compare         -> upload 2 CSVs
#   POST /batch-compare/upload  -> store bytes in STATE and show column dropdowns
#   POST /batch-compare/run     -> run fuzzy match (≥60), preview + download
# -----------------------------------------------------------------------------

@app.get("/batch-compare", response_class=HTMLResponse)
def compare_page():
    body = f"""
    <div class="card">
      <h2>🔁 Compare Two CSVs (Fuzzy)</h2>
      <p class="small">Upload two CSV files (Left vs Right). Choose the columns to match (name + optional brand). We’ll compute fuzzy matches (≥ {MATCH_THRESHOLD}%).</p>
      <form method="post" action="/batch-compare/upload" enctype="multipart/form-data">
        <div class="row"><label>Left CSV</label><input class="input" type="file" name="left" accept=".csv" required></div>
        <div class="row"><label>Right CSV</label><input class="input" type="file" name="right" accept=".csv" required></div>
        <div class="row"><label>Top candidates per item</label>
          <select name="top_k" class="input">
            {''.join(f'<option value="{n}" {"selected" if n==5 else ""}>{n}</option>' for n in [3,5,10,15,20])}
          </select>
        </div>
        <button class="btn" type="submit">Upload & Choose Columns</button>
      </form>
    </div>
    """
    return layout(body, active="home")

@app.post("/batch-compare/upload", response_class=HTMLResponse)
async def compare_upload(left: UploadFile = File(...), right: UploadFile = File(...), top_k: int = Form(5)):
    left_bytes = await left.read()
    right_bytes = await right.read()

    # Load CSVs
    def _load(b: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            return pd.read_csv(io.BytesIO(b), encoding="latin-1")

    try:
        dfL = _load(left_bytes)
        dfR = _load(right_bytes)
    except Exception:
        return layout("<div class='card'><h2>❌ Could not read one of the CSV files</h2><a class='btn' href='/batch-compare'>Back</a></div>", active="home")

    token = str(uuid.uuid4())
    STATE[token] = {
        "left": left_bytes,
        "right": right_bytes,
        "top_k": int(top_k)
    }

    # Build column dropdowns
    optsL = "".join(f"<option value='{html.escape(c)}'>{html.escape(c)}</option>" for c in dfL.columns)
    optsR = "".join(f"<option value='{html.escape(c)}'>{html.escape(c)}</option>" for c in dfR.columns)

    # Pre-select guess
    guessL_name, guessL_brand = _detect_cols(dfL)
    guessR_name, guessR_brand = _detect_cols(dfR)

    def sel(opts: str, guess: Optional[str]) -> str:
        if guess is None:
            return opts
        # mark selected
        return opts.replace(f"'>{html.escape(guess)}</option>", f"' selected>{html.escape(guess)}</option>")

    body = f"""
    <div class="card">
      <h2>🔧 Select Columns</h2>
      <form method="post" action="/batch-compare/run">
        <input type="hidden" name="token" value="{token}">
        <div class="grid">
          <div>
            <h3>Left CSV</h3>
            <div class="row"><label>Name column</label><select class="input" name="nameL">{sel(optsL, guessL_name)}</select></div>
            <div class="row"><label>Brand column (optional)</label><select class="input" name="brandL"><option value="">(none)</option>{sel(optsL, guessL_brand)}</select></div>
          </div>
          <div>
            <h3>Right CSV</h3>
            <div class="row"><label>Name column</label><select class="input" name="nameR">{sel(optsR, guessR_name)}</select></div>
            <div class="row"><label>Brand column (optional)</label><select class="input" name="brandR"><option value="">(none)</option>{sel(optsR, guessR_brand)}</select></div>
          </div>
        </div>
        <div class="row"><label>Top candidates per item</label>
          <select name="top_k" class="input">
            {''.join(f'<option value="{n}" {"selected" if n==STATE[token]["top_k"] else ""}>{n}</option>' for n in [3,5,10,15,20])}
          </select>
        </div>
        <button class="btn" type="submit">Run Compare</button>
      </form>
    </div>
    """
    return layout(body, active="home")

def _load_df_from_state(token: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st = STATE.get(token)
    if not st:
        raise ValueError("Invalid or expired session token")
    def _load(b: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            return pd.read_csv(io.BytesIO(b), encoding="latin-1")
    return _load(st["left"]), _load(st["right"])

def _compare_fuzzy(
    dfL: pd.DataFrame, dfR: pd.DataFrame,
    nameL: str, nameR: str,
    brandL: Optional[str], brandR: Optional[str],
    top_k: int = 5,
) -> pd.DataFrame:
    # Prepare right side index
    namesR = dfR[nameR].astype(str).map(_safe_str).tolist()
    brandsR = dfR[brandR].astype(str).map(_safe_str).tolist() if (brandR and brandR in dfR.columns) else [""] * len(dfR)
    # Precompute normalized choices once for performance (was inside loop before)
    norm_choices = [normalize(x) for x in namesR]

    rows = []
    for idx, row in dfL.iterrows():
        qn = _safe_str(row[nameL])
        qb = _safe_str(row[brandL]) if (brandL and brandL in dfL.columns) else ""
        if not qn:
            rows.append({
                "left_index": idx,
                "left_name": qn,
                "left_brand": qb,
                "right_index": "",
                "right_name": "",
                "right_brand": "",
                "score": "",
                "is_match": False
            })
            continue

        # shortlist by fuzzy name only using precomputed normalized choices
        shortlist = process.extract(
            normalize(qn),
            norm_choices,
            scorer=fuzz.WRatio,
            limit=max(10, top_k*4)  # generous shortlist but avoid too small limits
        )

        best = None
        best_score = -1.0
        best_j = -1
        for norm_nameR, wratio, j in shortlist:
            cand_name = namesR[j]
            cand_brand = brandsR[j]
            s, _ = score_candidate(qn, cand_name, qb, cand_brand)
            if s > best_score:
                best_score = s
                best = (cand_name, cand_brand)
                best_j = j

        is_match = bool(best is not None and best_score >= MATCH_THRESHOLD)
        rows.append({
            "left_index": idx,
            "left_name": qn,
            "left_brand": qb,
            "right_index": best_j if is_match else "",
            "right_name": best[0] if (best and is_match) else "",
            "right_brand": best[1] if (best and is_match) else "",
            "score": round(best_score,1) if best is not None else "",
            "is_match": is_match
        })

    out = pd.DataFrame(rows, columns=[
        "left_index","left_name","left_brand","right_index","right_name","right_brand","score","is_match"
    ])
    return out

@app.post("/batch-compare/run", response_class=HTMLResponse)
def compare_run(
    token: str = Form(...),
    nameL: str = Form(...),
    nameR: str = Form(...),
    brandL: str = Form(""),
    brandR: str = Form(""),
    top_k: int = Form(5),
):
    try:
        dfL, dfR = _load_df_from_state(token)
    except Exception as e:
        return layout(f"<div class='card'><h2>❌ {html.escape(str(e))}</h2><a class='btn' href='/batch-compare'>Back</a></div>", active="home")

    for col in [nameL, nameR]:
        if col not in dfL.columns and col not in dfR.columns:
            return layout("<div class='card'><h2>❌ Selected columns not found</h2><a class='btn' href='/batch-compare'>Back</a></div>", active="home")

    # ensure columns exist
    if nameL not in dfL.columns or nameR not in dfR.columns:
        return layout("<div class='card'><h2>❌ Selected columns not found</h2><a class='btn' href='/batch-compare'>Back</a></div>", active="home")
    if brandL and brandL not in dfL.columns:
        brandL = ""
    if brandR and brandR not in dfR.columns:
        brandR = ""

    top_k = int(top_k) if str(top_k).isdigit() else 5

    # Clean minimal for both dataframes
    dfL = dfL.copy()
    dfR = dfR.copy()
    dfL[nameL] = dfL[nameL].astype(str).map(_safe_str)
    dfR[nameR] = dfR[nameR].astype(str).map(_safe_str)
    if brandL: dfL[brandL] = dfL[brandL].astype(str).map(_safe_str)
    if brandR: dfR[brandR] = dfR[brandR].astype(str).map(_safe_str)

    t0 = time.time()
    out_df = _compare_fuzzy(dfL, dfR, nameL, nameR, brandL or None, brandR or None, top_k=top_k)
    dt = (time.time()-t0)*1000

    # Save downloadable CSV
    token_out = str(uuid.uuid4())
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    DOWNLOADS[token_out] = buf.getvalue().encode("utf-8")

    # Preview
    head = out_df.head(50).to_dict(orient="records")
    rows_html = ""
    for r in head:
        cls = "ok" if r.get("is_match") else "bad"
        rows_html += f"<tr><td>{r['left_index']}</td><td>{html.escape(str(r['left_name']))}</td><td>{html.escape(str(r.get('left_brand','')))}</td><td>{r.get('right_index','')}</td><td>{html.escape(str(r.get('right_name','')))}</td><td>{html.escape(str(r.get('right_brand','')))}</td><td class='score {cls}'>{html.escape(str(r.get('score','')))}</td><td>{'✅' if r.get('is_match') else '—'}</td></tr>"

    body = f"""
    <div class="card">
      <h2>✅ Compare Complete</h2>
      <p class="small">Matched with fuzzy threshold ≥ {MATCH_THRESHOLD}%. Top-k shortlist: {top_k}. Time: {dt:.1f} ms</p>
      <a class="btn" href="/download?token={token_out}">Download Comparison CSV</a>
      <br><br>
      <table class="table">
        <thead><tr><th>L-idx</th><th>L name</th><th>L brand</th><th>R-idx</th><th>R name</th><th>R brand</th><th>Score</th><th>Match?</th></tr></thead>
        <tbody>{rows_html or "<tr><td colspan='8'>No preview rows.</td></tr>"}</tbody>
      </table>
      <br><a class="btn secondary" href="/batch-compare">Back</a>
    </div>
    """
    return layout(body, active="home")
