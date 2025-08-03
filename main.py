import os
import json
import time
import random
import re
import logging
from datetime import datetime, timezone
from typing import List, Tuple
import gspread
from google.oauth2.service_account import Credentials


import httpx
import trafilatura
import google.generativeai as genai
from flask import Flask, jsonify, request

# ────────────────────────── Config ──────────────────────────
TERM_GROUPS = [{
    "ai":  ["generative AI", "GenAI", "large language model", "LLM"],
    "act": ["deployment", "implementation", "pilot", "rollout"]
}]

COMPANIES = ["1&1 Drillisch", "2degrees", "4iG", "AIS", "Almadar Aljaded (Al Madar)", "Altibox", "Altice", "Altice USA", "America Movil", "ANTEL", "Antina", "Asia Pacific Telecom", "AT&T", "AXIAN Telecom", "Axiata", "Batelco", "Bell Canada", "Bezeq", "Bharti Airtel", "BICS", "Bite Group", "Bité Group", "Bouygues Telecom", "Brisanet", "BSNL", "BT", "C Spire", "Cable & Wireless Communications", "Cablevisión", "Cellcom", "Charter Communications", "China Broadnet", "China Mobile", "China Telecom", "China Unicom", "Chunghwa Telecom", "Cincinnati Bell (CBTS)", "Cirion Technologies", "Citic Telecom CTC", "CityFibre", "Citymesh", "CK Hutchison", "COLT", "Com Hem", "Comcast", "Companhia de Telecomunicacoes de Macau (CTM)", "Consolidated Communications", "Cox Communications", "Crnogorski Telekom (CT)", "Cyfrowy Polsat Group", "CYTA", "DCConnect", "Deutsche Telekom", "Dhiraagu", "Digicel", "Digital Nasional Berhad (DNB)", "DISH Wireless", "Dito Telecommunity", "du", "e& (formerly Etisalat Group)", "EarthLink", "Econet Global", "eir", "Elisa", "Emtel", "Entel", "Ethio Telecom", "FarEasTone", "Faroese Telecom", "Frontier Communications", "GCI", "Globe Telecom", "GTT Communications", "HGC (formerly Hutchison Global Communications)", "HKT", "HOT Mobile", "Hughes Network Systems", "Hutchison 3G", "ice", "Idea Cellular", "Iliad", "Interoute", "Kazakhtelecom", "KDDI", "KPN", "KT", "Lao Telecommunications Company (LTC)", "Level 3", "LG U+", "Liberty Global", "Liberty Latin America", "Lumen Technologies (formerly CenturyLink)", "Lyse", "M1", "Macquarie Telecom", "Mafab Communications", "Mascom", "Masergy Communications", "MASMOVIL", "Mauritius Telecom", "Maxis", "MegaFon", "MegaPath", "Melita", "MetTel", "Millicom", "Mobile Communication Company of Iran (MCI)", "Mobile TeleSystems (MTS)", "Monaco Telecom", "MTN", "ngena", "NOS", "Nova", "NOW Telecom", "NTT Group", "Omantel", "One Communications (OneComm)", "Ooredoo", "Orange", "Pakistan Telecommunication Co. Ltd (PTCL)", "Paradise Mobile", "Partner Communications", "PCCW", "PLDT", "Polkomtel", "Post Luxembourg", "PPF Telecom Group", "Proximus (formerly Belgacom)", "Qcell", "Rain", "Rakuten Mobile", "RCS&RDS", "Reliance Jio", "Retelit", "Rogers Communications", "Rubicon Wireless", "Safaricom", "Sasktel", "Saudi Telecom Company (STC)", "SES", "SETAR", "Shaw Communications", "Sify Technologies", "Silknet", "SingTel", "SK Telecom", "SLT-Mobitel", "SmarTone", "SoftBank", "Somtel", "Spark New Zealand", "Spectra", "Sprint", "StarHub", "Sunrise Communications", "Swisscom", "Taiwan Mobile", "Taiwan Star Telecom (TST)", "TalkTalk", "Tata Communications", "Tcell", "TDC", "TDS Telecommunications", "Tele2", "Telecom Argentina", "Telecom Egypt", "Telecom Italia", "Telefónica", "Telekom Austria", "Telekom Malaysia", "Telekom Slovenije", "Telekom Srbija", "Telemach", "Telenet", "Telenor", "Telesom", "Telesur", "Telia", "Telia Denmark (Norlys)", "Telkom", "Telkom Indonesia", "Telma", "Telstra", "Telus", "TIME dotCom", "Tiscali", "TogoCom", "TPG Telecom", "TPx Communications", "True Corp.", "TSTT", "TT-Netvaerket", "Turk Telekom", "Turkcell", "U Mobile", "Ucell", "Unified National Networks (UNN)", "United Group", "Unitel", "US Cellular", "Uzbektelecom", "Veon (formerly VimpelCom)", "Verizon", "Videotron", "Viettel", "Virgin Media O2", "Vivacom", "Vocus Group", "Vodafone", "VodafoneZiggo", "Vonage", "Wind Hellas", "Wind Tre", "Windstream", "WOM", "Yettel", "YTL Communications", "Zain", "Zayo"]


MAX_RESULTS_PER_QUERY = 1    # ← only 1 result per company
MAX_TOTAL_RESULTS     = 100
MAX_FETCH             = 10
CSE_TIMEOUT_SEC       = 30.0
FETCH_TIMEOUT_SEC     = 20.0
GEMINI_MODEL          = "gemini-1.5-flash"

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# ────────────────────────── Helpers ──────────────────────────
def tail(v: str) -> str:
    return v[-6:] if v and v != "MISSING" else "MISSING"

def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def safe_json_extract(text: str) -> dict:
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except:
        return {}

# ───────────────────── Google CSE Search ─────────────────────
def build_queries() -> List[str]:
    g = TERM_GROUPS[0]
    ai_clause  = "(" + " OR ".join(g["ai"])  + ")"
    act_clause = "(" + " OR ".join(g["act"]) + ")"
    return [
        f'{ai_clause} AND {act_clause} AND "{company}"'
        for company in COMPANIES
    ]

def google_cse_search(query: str, api_key: str, cx: str,
                      date_restrict_days: int = 7) -> list:
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": MAX_RESULTS_PER_QUERY,
        "dateRestrict": f"d{date_restrict_days}"
    }
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
    try:
        r = client.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None,
                "error": f"HTTP:{e}"}

    extracted = trafilatura.extract(
        r.text, include_comments=False, include_tables=False,
        output_format="json", with_metadata=True, favor_precision=True
    )
    if not extracted:
        return {"url": url, "text": None, "title": None, "date": None,
                "error": "extract_failed"}

    try:
        data = json.loads(extracted)
    except Exception as e:
        return {"url": url, "text": None, "title": None, "date": None,
                "error": f"parse_json:{e}"}

    return {
        "url":   url,
        "text":  data.get("text"),
        "title": data.get("title"),
        "date":  data.get("date"),
        "error": None,
    }

# ─────────────────── LLM Relevance Filter ───────────────────
def gemini_generate_with_backoff(model, prompt: str, config: dict,
                                 max_tries: int = 5):
    for attempt in range(1, max_tries + 1):
        try:
            return model.generate_content(prompt, generation_config=config)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status == 503 or "503" in str(e)) and attempt < max_tries:
                wait = (2**attempt) + random.random()
                logging.warning("503 from Gemini, retry %d/%d after %.1fs",
                                attempt, max_tries, wait)
                time.sleep(wait)
                continue
            raise

def llm_relevance_filter(items: list, gemini_key: str,
                         model_name: str = GEMINI_MODEL
                        ) -> Tuple[List[dict], List[dict], List[dict]]:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    # escape JSON‐schema braces by doubling
    prompt_template = (
        "You are a strict filter for news about telecom operators (CSPs) "
        "using generative AI (GenAI/LLMs) for deployments, pilots, rollouts, or implementations.\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "{{\"relevant\": true/false, \"reason\": \"<≤20 words>\"}}\n\n"
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
                model, prompt, {"temperature":0, "max_output_tokens":60}
            )
        except Exception as e:
            errors.append({"url": it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text", "").strip()
        data = safe_json_extract(raw)
        if data.get("relevant"):
            it["relevance_reason"] = data.get("reason","")
            relevant.append(it)
        else:
            irrelevant.append({
                "url": it["link"],
                "reason": data.get("reason","not relevant")
            })

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
        "{{\"company\":\"…\",\"technology\":\"…\",\"activity\":\"…\",\"summary\":\"…\"}}\n\n"
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
                model, prompt, {"temperature":0, "max_output_tokens":120}
            )
        except Exception as e:
            errors.append({"url":it["link"], "error": str(e)})
            continue

        raw = getattr(resp, "text","").strip()
        data = safe_json_extract(raw)
        if all(k in data for k in ("company","technology","activity","summary")):
            rows.append([
                data["company"], data["technology"],
                data["activity"], data["summary"],
                it["link"], it.get("extracted_date") or today_iso()
            ])
        else:
            errors.append({"url": it["link"], "raw": raw})

        time.sleep(0.3 + random.random()*0.2)

    return rows, errors

# ------------------ Google Sheets Setup ------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES
)
gs_client = gspread.authorize(creds)

# Replace with your actual spreadsheet ID:
SPREADSHEET_ID = "1YRIXgBdft3PaJJrOLVjWudPTUXk8jfljNq7EkWtGtCY"
# And the name of the worksheet/tab you want to write to:
WORKSHEET_NAME = "NewsData"

def append_to_sheet(rows: list):
    """
    rows: List of lists, each inner list is a row of [company, tech, activity, summary, url, date]
    """
    sheet = gs_client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    # Append all rows in one batched call:
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    logging.info("Appended %d rows to sheet '%s'", len(rows), WORKSHEET_NAME)


# ────────────────────────── Flask Routes ──────────────────────────
@app.get("/")
def health():
    return jsonify({"message":"Alive. POST /run to execute."})

@app.post("/run")
def run_pipeline():
    dry_run = request.args.get("dry_run","false").lower()=="true"
    step    = request.args.get("step","all").lower()

    cse_key    = os.getenv("CSE_API_KEY","MISSING")
    cse_cx     = os.getenv("CSE_CX","MISSING")
    gemini_key = os.getenv("GEMINI_API_KEY","MISSING")
    if "MISSING" in (cse_key,cse_cx):
        return jsonify({"error":"CSE secrets missing"}), 500

    # 1) Build & Search
    queries = build_queries()
    all_results, seen = [], set()
    if step in ("all","search","fetch","llm","extract"):
        for q in queries:
            if len(seen) >= MAX_TOTAL_RESULTS:
                break
            for it in google_cse_search(q,cse_key,cse_cx, date_restrict_days=7):
                link = it.get("link")
                if link and link not in seen:
                    seen.add(link)
                    all_results.append(it)
            time.sleep(0.2 + random.random()*0.3)

    # 2) Fetch
    fetched, fetch_errors = [], []
    if step in ("all","fetch","llm","extract") and not dry_run:
        client = httpx.Client(timeout=FETCH_TIMEOUT_SEC)
        for it in all_results[:MAX_FETCH]:
            res = fetch_and_extract(it["link"], client)
            if res["error"]:
                fetch_errors.append(res)
            else:
                it.update({
                    "extracted_text":  res["text"],
                    "extracted_title": res["title"],
                    "extracted_date": res["date"],
                })
                fetched.append(it)
        client.close()

    # 3) Relevance filter
    relevant, irrelevant, rel_errors = [], [], []
    if step in ("all","llm","extract") and not dry_run and fetched:
        if gemini_key=="MISSING":
            return jsonify({"error":"GEMINI_API_KEY missing"}),500
        relevant, irrelevant, rel_errors = llm_relevance_filter(fetched,gemini_key)

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
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=False
    )
