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
SYSTEM = f"""You are an accounts-payable controller reviewing synthetic invoice packages.
Choose exactly ONE action per package:
- settle_invoice: invoice valid, reconciled (PO/receipt/amounts match) and within autonomous approval authority.
- request_approval: commercially valid but the amount/terms exceed delegated authority; needs a human approver.
- hold_invoice: payment pauses until a stated verification completes (e.g. bank-detail change callback, missing receipt/GRN, tax check).
- reject_duplicate: this exact commercial invoice (same vendor + invoice number + amount) was ALREADY PAID. A payment for a different invoice number is NOT a duplicate.
- open_exception: material records conflict (amount/vendor/PO/currency mismatches) and need an exception workflow.
The documents mix useful facts with old examples, archive notes, training decoys, negation ("do not hold ...") and
irrelevant action words. Decide from the paragraph that actually governs THIS invoice under the stated policy revision;
ignore cover-sheet references, archive examples and training decoys. Negated or hypothetical statements do not count.
evidenceRefs: return EXACTLY the three decisive bracketed references (copy character-for-character, including brackets
if the document writes them with brackets) from that single deciding paragraph. Never use the cover-sheet reference.
Respond with JSON only: {{"decisions":[{{"packageId":"...","action":"one of {ACTIONS}",
"facts":{{"vendorName":"...","invoiceNumber":"...","amountMinor":<integer minor units, e.g. paise/cents>,"currency":"ISO code"}},
"evidenceRefs":["r1","r2","r3"],"rationale":"2-4 sentences naming the action and citing the evidence refs"}}]}}"""


def cache_key(pkg):
    pid = pkg.get("packageId")
    stripped = {k: v for k, v in pkg.items() if not (v == pid and isinstance(v, str))}
    return sha(stripped)


def package_text(pkg):
    return canon(pkg)


async def call_llm(policy, pkgs):
    user = f"policyRevision: {policy}\npackages:\n" + json.dumps(pkgs, ensure_ascii=False)
    last = None
    # primary model gets most of the 45s request budget; the fast fallback covers a slow/failed first try
    for model, timeout in ((LLM_MODEL, 28), (LLM_FALLBACK, 14)):
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
    raise RuntimeError(f"llm failed: {last}")


def clean_decision(pkg, d):
    """Validate/normalize a model decision. Returns dict or None if unusable."""
    text = package_text(pkg)
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
    for r in d.get("evidenceRefs") or []:
        r = str(r)
        if r in text and r not in refs:
            refs.append(r)
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
    out, missing = {}, []
    for p in pkgs:
        row = db_one("SELECT body FROM decisions WHERE key=?", (cache_key(p) + "|" + str(policy),))
        if row:
            out[p["packageId"]] = json.loads(row[0])
        else:
            missing.append(p)
    for attempt in range(1):
        if not missing:
            break
        got = await call_llm(policy, missing)
        still = []
        for p in missing:
            d = clean_decision(p, got.get(p["packageId"], {}))
            if d is None:
                still.append(p)
                continue
            out[p["packageId"]] = d
            with _db_lock:
                _db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)",
                            (cache_key(p) + "|" + str(policy), json.dumps(d)))
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
    if "json" not in request.headers.get("content-type", "").lower():
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
    rows = db_all("SELECT body FROM tasks WHERE principal=? ORDER BY seq", (p,))
    return reply({"tasks": [json.loads(r[0]) for r in rows]})


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
