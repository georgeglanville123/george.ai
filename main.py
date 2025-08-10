#AI

import os
import json
import time
import random
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import gspread
from google.oauth2.service_account import Credentials
import httpx
import trafilatura
import google.generativeai as genai
from flask import Flask, jsonify, request
from urllib.parse import urlparse
import google.auth
from google.auth import default as adc_default
from zoneinfo import ZoneInfo

# ────────────────────────── Config ──────────────────────────
TERM_GROUPS = [{
    "ai":  ["generative AI", "GenAI", "large language model", "LLM"],
    "act": ["deployment", "implementation", "pilot", "rollout"]
}]

# Refined list (170)
COMPANIES = [
    "1&1 Drillisch", "2degrees", "4iG", "AIS", "Almadar Aljaded (Al Madar)", "Altibox",
    "Altice", "Altice USA", "America Movil", "AT&T", "AXIAN Telecom", "Axiata", "Batelco",
    "Bell Canada", "Bezeq", "Bharti Airtel", "BICS", "Bite Group", "Bouygues Telecom",
    "BSNL", "BT", "Charter Communications", "China Broadnet", "China Mobile", "China Telecom",
    "China Unicom", "Chunghwa Telecom", "altafiber (CBTS)", "Cirion Technologies",
    "Citic Telecom CTC", "CK Hutchison", "COLT", "Comcast",
    "Companhia de Telecomunicacoes de Macau (CTM)", "Cox Communications",
    "Cyfrowy Polsat Group", "CYTA", "Deutsche Telekom", "Digital Nasional Berhad (DNB)",
    "DISH Wireless", "Dito Telecommunity", "du", "e& (formerly Etisalat Group)", "Elisa",
    "Entel", "Ethio Telecom", "FarEasTone", "Globe Telecom", "HKT", "Iliad", "Kazakhtelecom",
    "KDDI", "KPN", "KT", "Lao Telecommunications Company (LTC)", "LG U+",
    "Liberty Global", "Liberty Latin America", "Lumen Technologies (formerly CenturyLink)",
    "Mafab Communications", "MASMOVIL", "Mauritius Telecom", "Maxis", "MegaFon", "MegaPath",
    "Melita", "Millicom", "Mobile Communication Company of Iran (MCI)",
    "Mobile TeleSystems (MTS)", "MTN", "ngena", "NOS", "NOW Telecom", "NTT Group",
    "Omantel", "Ooredoo", "Orange", "Pakistan Telecommunication Co. Ltd (PTCL)", "PLDT",
    "Polkomtel", "PPF Telecom Group", "Proximus (formerly Belgacom)", "Rain", "Rakuten Mobile",
    "RCS&RDS", "Reliance Jio", "Retelit", "Rogers Communications", "Safaricom", "Sasktel",
    "stc", "SES", "Sify Technologies", "Silknet", "Singtel", "SK Telecom", "SLT-Mobitel",
    "SmarTone", "SoftBank", "Spark New Zealand", "Sprint", "StarHub", "Sunrise Communications",
    "Swisscom", "Taiwan Mobile", "TalkTalk", "Tata Communications", "Tcell", "TDC", "Tele2",
    "Telecom Argentina", "Telecom Egypt", "TIM", "Telefónica", "A1 Telekom Austria",
    "Telekom Malaysia", "Telekom Slovenije", "Telekom Srbija", "Telenet", "Telenor", "Telia",
    "Telkom (South Africa)", "Telkom Indonesia", "Telstra", "Telus", "TIME dotCom",
    "TPG Telecom", "TPx Communications", "True Corp.", "Turk Telekom", "Turkcell", "Ucell",
    "Unified National Networks (UNN)", "United Group", "Unitel", "US Cellular", "Uzbektelecom",
    "Veon (formerly VimpelCom)", "Verizon", "Videotron", "Viettel", "Virgin Media O2",
    "Vivacom", "Vocus Group", "Vodafone", "VodafoneZiggo", "Vonage", "Wind Tre", "Windstream",
    "WOM", "Yettel", "YTL Communications", "Zain", "Zayo", "T-Mobile US", "Vodacom",
    "Telkomsel", "XL Axiata", "Indosat Ooredoo Hutchison", "TIM Brasil",
    "Vivo (Telefônica Brasil)", "Claro", "Mobily", "Airtel Africa", "Jazz",
    "Grameenphone", "Robi Axiata"
]

BINARY_EXTS = (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx")

MAX_RESULTS_PER_QUERY = 1    # ← only 1 result per company
MAX_TOTAL_RESULTS     = 170
MAX_FETCH             = 25
CSE_TIMEOUT_SEC       = 30.0
FETCH_TIMEOUT_SEC     = 20.0
GEMINI_MODEL          = "gemini-1.5-flash"

# Time filtering knobs (adjust via env or edit here)
TIME_MODE = os.getenv("TIME_MODE", "calendar")  # "calendar" or "rolling"
TIME_DAYS = int(os.getenv("TIME_DAYS", "1"))    # N-day window
TIME_ZONE = "Europe/London"                     # for calendar mode

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

AI_HINTS  = tuple(w.lower() for w in TERM_GROUPS[0]["ai"])
ACT_HINTS = tuple(w.lower() for w in TERM_GROUPS[0]["act"])

ACTION_VERBS = (
    "announce","launch","pilot","rollout","roll-out","implement",
    "deploy","deployment","go-live","trial","proof of concept","poc"
)

BAD_HINTS = (
    "job","hiring","career","vacancy","investor","ir","earnings",
    "q1","q2","q3","q4","whitepaper","webinar","rfp","tender",
    "brochure","datasheet"
)

PRESS_PATHS = ("/press", "/news", "/media", "/newsroom")
DOC_EXTS = (".pdf",".ppt",".pptx",".doc",".docx",".xls",".xlsx")

# ────────────────────────── Helpers ──────────────────────────
def tail(v: str) -> str:
    return v[-6:] if v and v != "MISSING" else "MISSING"

def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def safe_json_extract(text: str) -> dict:
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

def _has_any(text: str, terms) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", t, re.I) for term in terms)

def score_item(it: dict) -> int:
    """Score a CSE item using title/snippet/url + company in it['company']."""
    title   = (it.get("title") or "")
    snippet = (it.get("snippet") or "")
    url     = (it.get("link")  or "")
    company = (it.get("company") or "")

    path = urlparse(url).path.lower()
    host = urlparse(url).netloc.lower()

    score = 0
    # Company mention (title > snippet)
    if _has_any(title, [company]):      score += 3
    elif _has_any(snippet, [company]):  score += 1

    # AI + Activity signals (title weighted more than snippet)
    if _has_any(title, AI_HINTS):       score += 3
    if _has_any(title, ACT_HINTS):      score += 3
    if _has_any(snippet, AI_HINTS):     score += 1
    if _has_any(snippet, ACT_HINTS):    score += 1

    # Action verbs (deploy/pilot/etc.)
    if _has_any(title, ACTION_VERBS):   score += 2
    if _has_any(snippet, ACTION_VERBS): score += 1

    # “Newsroom/press” path hint
    if any(p in path for p in PRESS_PATHS): score += 1

    # Penalties: downloads / junky intents / aggregator copies
    if path.endswith(DOC_EXTS):                      score -= 3
    if _has_any(title, BAD_HINTS) or _has_any(snippet, BAD_HINTS): score -= 3
    if host in ("news.google.com","news.yahoo.com"): score -= 2  # optional diversity

    return score

def cap_by_domain(items, per_domain=2):
    """Keep at most N items per hostname to avoid near-duplicates."""
    counts, out = {}, []
    for it in items:
        d = urlparse(it["link"]).netloc.lower()
        if counts.get(d, 0) < per_domain:
            out.append(it)
            counts[d] = counts.get(d, 0) + 1
    return out

# ───────────────────── Google CSE Search ─────────────────────
def build_queries() -> List[Tuple[str, str]]:
    g = TERM_GROUPS[0]
    ai_clause  = "(" + " OR ".join(g["ai"])  + ")"
    act_clause = "(" + " OR ".join(g["act"]) + ")"
    return [
        (f'{ai_clause} AND {act_clause} AND "{company}"', company)
        for company in COMPANIES
    ]

def google_cse_search(query: str, api_key: str, cx: str,
                      mode: str = TIME_MODE, days: int = TIME_DAYS) -> list:
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": MAX_RESULTS_PER_QUERY,
    }

    if mode == "calendar":
        # Previous N full calendar days (excluding today) in Europe/London
        ldn = ZoneInfo(TIME_ZONE)
        today_ldn = datetime.now(ldn).date()
        end = today_ldn - timedelta(days=1)          # yesterday
        start = end - timedelta(days=days - 1)       # go back N-1 days
        params["sort"] = f"date:r:{start:%Y%m%d}:{end:%Y%m%d}"
        logging.info("CSE calendar window: %s → %s", start.isoformat(), end.isoformat())
    else:
        # Rolling window (e.g., last 2 days ≈ 48h)
        params["dateRestrict"] = f"d{days}"
        logging.info("CSE rolling window: last %d day(s)", days)

    with httpx.Client(timeout=CSE_TIMEOUT_SEC) as client:
        resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)
        if resp.status_code == 429:
            time.sleep(1 + random.random())
            resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)
        resp.raise_for_status()
        return resp.json().get("items", []) or []

# ───────────────────── Fetch & Extract ─────────────────────
def fetch_and_extract(url: str, client: httpx.Client,
                      timeout: float = FETCH_TIMEOUT_SEC) -> dict:
    # 1) Skip obvious non-HTML files by extension (no request needed)
    if url.lower().endswith(BINARY_EXTS):
        return {"url": url, "text": None, "title": None, "date": None, "error": "skipped_binary"}

    # 2) Fetch the page; bail if not HTML
    try:
        r = client.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"HTTP:{e}"}

    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"non_html:{ctype}"}

    # 3) Extract main content
    extracted = trafilatura.extract(
        r.text,
        include_comments=False,
        include_tables=False,
        output_format="json",
        with_metadata=True,
        favor_precision=True
    )
    if not extracted:
        return {"url": url, "text": None, "title": None, "date": None, "error": "extract_failed"}

    try:
        data = json.loads(extracted)
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None, "error": f"parse_json:{e}"}

    return {
        "url":   url,
        "text":  data.get("text"),
        "title": data.get("title"),
        "date":  data.get("date"),
        "error": None,
    }

# ─────────────────── LLM Relevance Filter ───────────────────
def gemini_generate_with_backoff(model, prompt: str, config: dict, max_tries: int = 5):
    for attempt in range(1, max_tries + 1):
        try:
            return model.generate_content(prompt, generation_config=config)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status == 503 or "503" in str(e)) and attempt < max_tries:
                wait = (2**attempt) + random.random()
                logging.warning("503 from Gemini, retry %d/%d after %.1fs", attempt, max_tries, wait)
                time.sleep(wait)
                continue
            raise

def llm_relevance_filter(items: list, gemini_key: str,
                         model_name: str = GEMINI_MODEL
                        ) -> Tuple[List[dict], List[dict], List[dict]]:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    prompt_template = (
        "You are a strict filter for news about telecom operators (CSPs) "
        "using generative AI (GenAI/LLMs) for deployments, pilots, rollouts, or implementations.\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "{\"relevant\": true/false, \"reason\": \"<≤20 words>\"}\n\n"
        "Text to review (truncated):\n"
        "\"\"\"{article}\"\"\""
    )

    relevant, irrelevant, errors = [], [], []
    logging.info("Sending %d items to Gemini for relevance", len(items))

    for it in items:
        text = (it.get("extracted_text") or "").strip()
        if not text:
            irrelevant.append({"url": it["link"], "reason": "no text"})
            continue

        prompt = prompt_template.format(article=text[:3000])
        try:
            resp = gemini_generate_with_backoff(
                model, prompt, {"temperature": 0, "max_output_tokens": 60}
            )
        except Exception as e:
            errors.append({"url": it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text", "").strip()
        data = safe_json_extract(raw)
        if data.get("relevant"):
            it["relevance_reason"] = data.get("reason", "")
            relevant.append(it)
        else:
            irrelevant.append({"url": it["link"], "reason": data.get("reason", "not relevant")})

        time.sleep(0.3 + random.random()*0.2)

    return relevant, irrelevant, errors

# ─────────────────── LLM Structured Extraction ───────────────────
def llm_extract_structured(items: list, gemini_key: str,
                           model_name: str = GEMINI_MODEL
                          ) -> Tuple[List[List[str]], List[dict]]:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    prompt_schema = (
        "Extract these fields from the article text:\n"
        "- company: primary telecom operator or vendor\n"
        "- technology: GenAI/LLM used\n"
        "- activity: deployment, pilot, rollout, etc.\n"
        "- summary: <=40 words\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "{\"company\":\"…\",\"technology\":\"…\",\"activity\":\"…\",\"summary\":\"…\"}\n\n"
        "Text:\n"
        "\"\"\"{article}\"\"\""
    )

    rows, errors = [], []
    logging.info("Extracting structured data for %d items", len(items))

    for it in items:
        snippet = (it.get("extracted_text") or "")[:3500]
        prompt = prompt_schema.format(article=snippet)
        try:
            resp = gemini_generate_with_backoff(
                model, prompt, {"temperature": 0, "max_output_tokens": 120}
            )
        except Exception as e:
            errors.append({"url": it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text", "").strip()
        data = safe_json_extract(raw)
        if all(k in data for k in ("company","technology","activity","summary")):
            rows.append([
                data["company"], data["technology"], data["activity"], data["summary"],
                it["link"], it["extracted_date"] or today_iso()
            ])
        else:
            errors.append({"url": it["link"], "raw": raw})

        time.sleep(0.3 + random.random()*0.2)

    return rows, errors

# ------------------ Google Sheets Setup ------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if sa_path:
    # Running locally or in CI with a key file
    creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
else:
    # Running on Cloud Run: use Application Default Credentials
    creds, _ = adc_default(scopes=SCOPES)

gs_client = gspread.authorize(creds)

SPREADSHEET_ID = "1YRIXgBdft3PaJJrOLVjWudPTUXk8jfljNq7EkWtGtCY"
WORKSHEET_NAME = "NewsData"

def append_to_sheet(rows: list):
    """rows: list of [company, tech, activity, summary, url, date]"""
    sheet = gs_client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    logging.info("Appended %d rows to sheet '%s'", len(rows), WORKSHEET_NAME)

# ────────────────────────── Flask Routes ──────────────────────────
@app.get("/")
def health():
    return jsonify({"message": "Alive. POST /run to execute."})

@app.post("/run")
def run_pipeline():
    dry_run = request.args.get("dry_run","false").lower()=="true"
    step    = request.args.get("step","all").lower()

    cse_key    = os.getenv("CSE_API_KEY","MISSING")
    cse_cx     = os.getenv("CSE_CX","MISSING")
    gemini_key = os.getenv("GEMINI_API_KEY","MISSING")
    if "MISSING" in (cse_key, cse_cx):
        return jsonify({"error":"CSE secrets missing"}), 500

    # 1) Build & Search
    queries = build_queries()  # List[Tuple[query, company]]
    all_results, seen = [], set()
    if step in ("all","search","fetch","llm","extract"):
        for q, company in queries:
            if len(seen) >= MAX_TOTAL_RESULTS:
                break
            for it in google_cse_search(q, cse_key, cse_cx):  # uses TIME_MODE/TIME_DAYS
                link = it.get("link")
                if link and link not in seen:
                    seen.add(link)
                    all_results.append({
                        "link": link,
                        "title": it.get("title",""),
                        "snippet": it.get("snippet",""),
                        "company": company
                    })
            time.sleep(0.2 + random.random()*0.3)

    # 1.5) Rank and pick the best to fetch
    ranked = sorted(
        all_results,
        key=lambda it: (score_item(it), len(it.get("title",""))),
        reverse=True
    )
    candidates = cap_by_domain(ranked, per_domain=2)[:MAX_FETCH]

    # 2) Fetch (use the ranked candidates)
    fetched, fetch_errors = [], []
    if step in ("all","fetch","llm","extract") and not dry_run:
        client = httpx.Client(timeout=FETCH_TIMEOUT_SEC)
        for it in candidates:
            res = fetch_and_extract(it["link"], client)
            if res["error"]:
                fetch_errors.append(res)
            else:
                it.update({
                    "extracted_text":  res["text"],
                    "extracted_title": res["title"],
                    "extracted_date":  res["date"],
                })
                fetched.append(it)
        client.close()

    # 3) Relevance filter
    relevant, irrelevant, rel_errors = [], [], []
    if step in ("all","llm","extract") and not dry_run and fetched:
        if gemini_key == "MISSING":
            return jsonify({"error":"GEMINI_API_KEY missing"}), 500
        relevant, irrelevant, rel_errors = llm_relevance_filter(fetched, gemini_key)

    # 4) Structured extract
    structured, ext_errors = [], []
    if step in ("all","extract") and not dry_run and relevant:
        structured, ext_errors = llm_extract_structured(relevant, gemini_key)

    # 5) Append to Google Sheet
    if structured:
        try:
            append_to_sheet(structured)
        except Exception as e:
            logging.exception("Failed to write to Google Sheet: %s", e)

    # Summary
    return jsonify({
        "status": "ok",
        "date": today_iso(),
        "dry_run": dry_run,
        "step": step,
        "total_raw": len(all_results),
        "deduped": len(seen),
        "fetched": len(fetched),
        "fetch_errors": len(fetch_errors),
        "relevant": len(relevant),
        "irrelevant": len(irrelevant),
        "relevance_errors": len(rel_errors),
        "extract_rows": len(structured),
        "extract_errors": len(ext_errors),
        "secrets_tail": {
            "cse_key": tail(cse_key),
            "cse_cx":  tail(cse_cx),
            "gemini":  tail(gemini_key)
        }
    }), 200

if __name__ == "__main__":
    # Handy for local runs: Cloud Run ignores this because you use gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
