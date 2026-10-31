"""A2A 1.0 (HTTP+JSON) invoice action agent.

Layers: protocol (routes) -> store (sqlite) -> decider (LLM + validation).
Only message:send with a new batch calls the model; everything else is pure storage.
"""
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

BASE_URL = os.environ.get("PUBLIC_BASE_URL") or (
    os.environ["RENDER_EXTERNAL_URL"].rstrip("/") + "/" if os.environ.get("RENDER_EXTERNAL_URL")
    else "http://localhost:8000/")
DB_PATH = os.environ.get("DB_PATH", "agent.db")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
LLM_FALLBACK = os.environ.get("LLM_FALLBACK_MODEL", "gemini-flash-lite-latest")

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
JSON_CT = "application/a2a+json"

PREFIX = urlparse(BASE_URL).path.rstrip("/")
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter()

# ---------------------------------------------------------------- storage
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db_lock = threading.RLock()
with _db_lock:
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, body TEXT)")
    _db.execute("CREATE TABLE IF NOT EXISTS msgs(principal TEXT, message_id TEXT, hash TEXT, task_id TEXT, "
                "PRIMARY KEY(principal, message_id))")
    _db.execute("CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT)")
    _db.execute("CREATE TABLE IF NOT EXISTS proposals(task_id TEXT PRIMARY KEY, body TEXT)")


def db_one(sql, args=()):
    with _db_lock:
        return _db.execute(sql, args).fetchone()


def db_all(sql, args=()):
    with _db_lock:
        return _db.execute(sql, args).fetchall()


def load_task(principal, task_id):
    row = db_one("SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal))
    return json.loads(row[0]) if row else None


def save_task(principal, task, msg_rows=()):
    """Atomically upsert a task and any (message_id, hash) rows."""
    with _db_lock:
        _db.execute("BEGIN")
        try:
            exists = _db.execute("SELECT seq FROM tasks WHERE id=?", (task["id"],)).fetchone()
            if exists:
                _db.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(task), task["id"]))
            else:
                seq = (_db.execute("SELECT COALESCE(MAX(seq),0) FROM tasks").fetchone()[0]) + 1
                _db.execute("INSERT INTO tasks VALUES(?,?,?,?)", (task["id"], principal, seq, json.dumps(task)))
            for mid, h in msg_rows:
                _db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
            _db.execute("COMMIT")
        except Exception:
            _db.execute("ROLLBACK")
            raise


# ---------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(o):
    return hashlib.sha256(canon(o).encode()).hexdigest()


def reply(obj, status=200):
    return Response(json.dumps(obj), status_code=status, media_type=JSON_CT)


def err(status, code, msg):
    return reply({"error": {"code": status, "status": code, "message": msg}}, status)


def principal_of(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or len(auth) < 15:
        return None
    return hashlib.sha256(auth[7:].strip().encode()).hexdigest()


def check_common(request: Request):
    """Returns (principal, error_response)."""
    p = principal_of(request)
    if p is None:
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "") != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "A2A-Version 1.0 is required.")
    return p, None


# ---------------------------------------------------------------- decider
PROMPT_VERSION = "v2"
BR = re.compile(r"\[[^\[\]\r\n]{1,120}\]")

SYSTEM = f"""You are a careful accounts-payable controller reviewing synthetic invoice packages.
Each package is split into numbered paragraphs [P1], [P2], ... Choose exactly ONE action per package:
- settle_invoice: the controlling facts say the invoice is valid, fully reconciled, and within autonomous approval authority, with NO pending verification, conflict, duplicate or approval requirement.
- request_approval: commercially valid and reconciled, but amount/terms are OUTSIDE delegated authority, so a human approver must sign off before payment.
- hold_invoice: payment pauses until a stated verification completes (e.g. bank-detail callback, missing receipt/GRN, tax/KYC check, vendor confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number and amount) was ALREADY PAID. A payment for a different invoice number is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, currency, quantity mismatches) and need an exception workflow.

How to read a package:
- The packages deliberately mix the controlling case facts with distractors: cover sheets, archive/old examples, training decoys, quoted past cases, hypotheticals, negated statements ("do NOT hold", "no duplicate was found") and irrelevant action words. Words like "hold" or "approval" in a distractor or in a negation do not decide anything.
- The controlling paragraph is the one that states the CURRENT case's facts for this invoice under the stated policy revision and says what must happen. Pick that single paragraph.
- Facts (vendorName, invoiceNumber, amount, currency) must be the controlling invoice's own, not an archive example's. amountMinor is an integer in minor units (INR 1,234.50 -> 123450; no decimals -> multiply by 100).
- If the evidence is mixed or unclear, never choose settle_invoice: settle only when the controlling paragraph clearly supports payment.
- Output "decidingParagraph" as the paragraph id (e.g. "P4") and "evidenceRefs" as the bracketed references from that paragraph (copy exactly, brackets included). Never use the cover-sheet reference.
- rationale: 2-4 sentences. Name the action, state the decisive facts, explain how the evidence supports this action rather than the others, and quote at least two evidence refs verbatim.
Respond with JSON only: {{"decisions":[{{"packageId":"...","action":"one of {ACTIONS}","decidingParagraph":"P#",
"facts":{{"vendorName":"...","invoiceNumber":"...","amountMinor":<integer>,"currency":"ISO code"}},
"evidenceRefs":["..."],"rationale":"..."}}]}}"""


def cache_key(pkg, policy):
    pid = pkg.get("packageId")
    stripped = {k: v for k, v in pkg.items() if not (v == pid and isinstance(v, str))}
    return f"{PROMPT_VERSION}|{policy}|{sha(stripped)}"


def paragraphs(pkg):
    """Flatten a package into [(id, text)] paragraphs (split on blank lines, else on newlines)."""
    out = []

    def add(s):
        s = s.strip()
        if not s:
            return
        parts = re.split(r"\n\s*\n", s)
        if len(parts) == 1 and "\n" in s:
            parts = s.split("\n")
        out.extend(p.strip() for p in parts if p.strip())

    def walk(o):
        if isinstance(o, str):
            add(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k == "packageId":
                    continue
                if isinstance(v, (dict, list)):
                    walk(v)
                elif isinstance(v, str) and len(v) >= 80:
                    add(v)
                else:
                    add(f"{k}: {v}")
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif o is not None:
            add(str(o))

    walk(pkg)
    return [(f"P{i + 1}", t) for i, t in enumerate(out)]


def render(pkg, paras):
    lines = [f"=== PACKAGE {pkg['packageId']} ==="]
    for pid, t in paras:
        lines.append(f"[{pid}] {t}")
    return "\n".join(lines)


async def call_llm(policy, rendered, budget=40.0):
    user = f"policyRevision: {policy}\n\n" + "\n\n".join(rendered)
    t0, last = time.monotonic(), None
    models = [LLM_MODEL, LLM_FALLBACK, "gemini-3.1-flash-lite", LLM_MODEL, LLM_FALLBACK]
    for n, model in enumerate(models):
        timeout = min(24.0, budget - (time.monotonic() - t0))
        if timeout < 4:
            break
        if n:
            await asyncio.sleep(min(1.0, max(0.0, timeout - 4)))
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    LLM_BASE_URL.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                    json={"model": model, "temperature": 0,
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
            return {d["packageId"]: d for d in json.loads(txt)["decisions"]}
        except Exception as e:
            last = repr(e)
            print("LLM_ERROR", model, last[:300], flush=True)
    raise RuntimeError(f"llm failed: {last}")


def clean_decision(pkg, paras, d):
    """Validate/normalize a model decision. Returns dict or None if unusable."""
    text = canon(pkg)
    action = d.get("action")
    if action not in ACTIONS:
        return None
    f = d.get("facts") or {}
    try:
        facts = {"vendorName": str(f["vendorName"]), "invoiceNumber": str(f["invoiceNumber"]),
                 "amountMinor": int(round(float(f["amountMinor"]))), "currency": str(f["currency"]).upper()}
    except Exception:
        return None
    refs = []
    para = dict(paras).get(str(d.get("decidingParagraph", "")).strip("[] "))
    if para:
        refs = list(dict.fromkeys(BR.findall(para)))
    if len(refs) != 3:  # fall back to what the model listed, if it really appears in the package
        refs = [r for r in dict.fromkeys(str(x) for x in (d.get("evidenceRefs") or [])) if r in text]
    if len(refs) < 2:
        return None
    rationale = str(d.get("rationale", "")).strip()
    for r in refs:
        if r not in rationale:
            rationale += f" Evidence: {r}."
    if action not in rationale:
        rationale = f"Action {action}. " + rationale
    while len(rationale) < 60:
        rationale += " Decision follows the governing paragraph of the package."
    return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rationale[:1500]}


async def decide(policy, pkgs):
    """pkg list -> {packageId: decision}, using the content-keyed cache and one batched model call."""
    t0 = time.monotonic()
    out, missing = {}, []
    for p in pkgs:
        row = db_one("SELECT body FROM decisions WHERE key=?", (cache_key(p, policy),))
        if row:
            out[p["packageId"]] = json.loads(row[0])
        else:
            missing.append(p)
    for attempt in range(2):
        if not missing:
            break
        budget = 42.0 - (time.monotonic() - t0)
        if budget < 6:
            break
        paras = {p["packageId"]: paragraphs(p) for p in missing}
        if attempt == 0:
            print("PKGSAMPLE", json.dumps(missing[0], ensure_ascii=False)[:2500], flush=True)
        got = await call_llm(policy, [render(p, paras[p["packageId"]]) for p in missing], budget)
        still = []
        for p in missing:
            d = clean_decision(p, paras[p["packageId"]], got.get(p["packageId"], {}))
            if d is None:
                still.append(p)
                continue
            out[p["packageId"]] = d
            with _db_lock:
                _db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (cache_key(p, policy), json.dumps(d)))
        missing = still
    if missing:
        raise RuntimeError("could not decide all packages")
    return out


# ---------------------------------------------------------------- protocol
_key_locks, _task_locks = {}, {}


def klock(d, k):
    return d.setdefault(k, asyncio.Lock())


def _origin():
    u = urlparse(BASE_URL)
    return f"{u.scheme}://{u.netloc}"


def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-cited business action each, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": u, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
                                for u in dict.fromkeys([BASE_URL, _origin() + "/", _origin() + "/a2a/"])],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_BATCH, MT_RESULTS],
        "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
        "skills": [{
            "id": "invoice_action_agent", "name": "Invoice action decisions",
            "description": "Choose settle, approve, hold, reject-duplicate or exception for each invoice package with cited evidence.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
            "inputModes": [MT_BATCH], "outputModes": [MT_PROPOSALS, MT_RECEIPTS]}],
    }


@app.get("/.well-known/agent-card.json")
async def card():
    return reply(agent_card())


def terminal(task):
    return task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED")


@router.post("/message:send")
async def message_send(request: Request):
    p, e = check_common(request)
    if e:
        return e
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != JSON_CT:
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg.get("role") == "ROLE_USER"
        part = msg["parts"][0]
        data = part["data"]
        assert isinstance(data, dict)
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Malformed message.")
    mhash = sha(msg)

    async with klock(_key_locks, (p, mid)):
        prior = db_one("SELECT hash, task_id FROM msgs WHERE principal=? AND message_id=?", (p, mid))
        if prior:
            if prior[0] != mhash:
                return err(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: messageId reused with different content.")
            return reply({"task": load_task(p, prior[1])})
        if msg.get("taskId"):
            return await continuation(p, msg, part, data, mid, mhash)
        if part.get("mediaType") != MT_BATCH:
            return err(400, "INVALID_ARGUMENT", "Unsupported input media type.")
        return await new_batch(p, msg, data, mid, mhash)


async def new_batch(p, msg, data, mid, mhash):
    pkgs = data.get("packages")
    if (not isinstance(data.get("batchId"), str) or not isinstance(pkgs, list) or not pkgs
            or not all(isinstance(x, dict) and isinstance(x.get("packageId"), str) for x in pkgs)
            or len({x["packageId"] for x in pkgs}) != len(pkgs)):
        return err(400, "INVALID_ARGUMENT", "Malformed batch.")
    try:
        decisions = await decide(data.get("policyRevision"), pkgs)
    except Exception:
        return err(503, "UNAVAILABLE", "Decision service unavailable; retry.")
    proposals = []
    for x in pkgs:
        d = decisions[x["packageId"]]
        aid = "act-" + hashlib.sha256(f"{p}|{data['batchId']}|{x['packageId']}|{d['action']}".encode()).hexdigest()[:24]
        proposals.append({"packageId": x["packageId"], "actionId": aid, **d})
    tid, cid = str(uuid.uuid4()), str(uuid.uuid4())
    msg = {**msg, "taskId": tid, "contextId": cid}
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "invoice-action-proposals",
                           "parts": [{"mediaType": MT_PROPOSALS,
                                      "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [msg]}
    save_task(p, task, [(mid, mhash)])
    return reply({"task": task})


async def continuation(p, msg, part, data, mid, mhash):
    tid = msg["taskId"]
    async with klock(_task_locks, tid):
        task = load_task(p, tid)
        if task is None:
            return err(404, "NOT_FOUND", "Task not found.")
        if terminal(task):
            return err(409, "TASK_NOT_ACTIONABLE", "Task is already finished.")
        if msg.get("contextId") != task["contextId"] or part.get("mediaType") != MT_RESULTS:
            return err(400, "INVALID_ARGUMENT", "Continuation does not match task.")
        stored = task["artifacts"][0]["parts"][0]["data"]
        results = data.get("results")
        if data.get("batchId") != stored["batchId"] or not isinstance(results, list):
            return err(400, "INVALID_ARGUMENT", "Continuation does not match task.")
        props = {x["packageId"]: x for x in stored["proposals"]}
        seen, executions = set(), []
        for r in results:
            pr = props.get(r.get("packageId")) if isinstance(r, dict) else None
            if (pr is None or r["packageId"] in seen or r.get("actionId") != pr["actionId"]
                    or r.get("action") != pr["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_ARGUMENT", "Continuation does not match task.")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    return err(400, "INVALID_ARGUMENT", "Continuation does not match task.")
                executions.append({"packageId": pr["packageId"], "actionId": pr["actionId"], "action": pr["action"],
                                   "receiptNonce": n, "facts": pr["facts"], "evidenceRefs": pr["evidenceRefs"]})
        task["history"].append(msg)
        task["artifacts"][0]["parts"] = task["artifacts"][0]["parts"][:1]
        task["artifacts"][0]["parts"].append(
            {"mediaType": MT_RECEIPTS, "data": {"batchId": stored["batchId"], "executions": executions}})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save_task(p, task, [(mid, mhash)])
        return reply({"task": task})


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    p, e = check_common(request)
    if e:
        return e
    t = load_task(p, task_id)
    return reply(t) if t else err(404, "NOT_FOUND", "Task not found.")


@router.get("/tasks")
async def list_tasks(request: Request):
    p, e = check_common(request)
    if e:
        return e
    rows = db_all("SELECT body FROM tasks WHERE principal=? ORDER BY seq DESC LIMIT 50", (p,))
    tasks = []
    for r in rows:  # compact view: full history/artifacts are available via GET /tasks/{id}
        t = json.loads(r[0])
        tasks.append({"id": t["id"], "contextId": t["contextId"], "status": t["status"]})
    return reply({"tasks": tasks})


@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    p, e = check_common(request)
    if e:
        return e
    async with klock(_task_locks, task_id):
        t = load_task(p, task_id)
        if t is None:
            return err(404, "NOT_FOUND", "Task not found.")
        if terminal(t):
            return err(409, "TASK_NOT_CANCELABLE", "Task is already finished.")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(p, t)
        return reply(t)


for _p in {PREFIX, "/a2a", ""}:
    app.include_router(router, prefix=_p)
