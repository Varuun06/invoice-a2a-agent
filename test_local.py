import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["PUBLIC_BASE_URL"] = "https://host.example/a2a/"

import app as A
from fastapi.testclient import TestClient

calls = []


async def fake_llm(policy, rendered, budget=40.0):
    calls.append(len(rendered))
    out = {}
    for block in rendered:
        pid = block.split("===")[1].replace("PACKAGE", "").strip()
        out[pid] = {"action": "hold_invoice", "decidingParagraph": "P1",
                    "facts": {"vendorName": "Acme", "invoiceNumber": "INV-1", "amountMinor": 1234, "currency": "inr"},
                    "evidenceRefs": [], "rationale": "short"}
    return out


A.call_llm = fake_llm
c = TestClient(A.app)
H = {"Authorization": "Bearer token-user-one-xxxx", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
H2 = {**H, "Authorization": "Bearer token-user-two-xxxx"}


def batch(mid, bid):
    pk = [{"packageId": f"{bid}-{i}", "text": f"doc {i} [REF-{i}-A] [REF-{i}-B] [REF-{i}-C]",
           "refs": [f"[REF-{i}-A]", f"[REF-{i}-B]", f"[REF-{i}-C]"]} for i in range(12)]
    return {"message": {"messageId": mid, "role": "ROLE_USER", "parts": [
        {"mediaType": A.MT_BATCH, "data": {"batchId": bid, "policyRevision": "r1", "packages": pk}}]},
        "configuration": {"returnImmediately": False}}


assert c.get("/.well-known/agent-card.json").status_code == 200
assert c.get("/a2a/tasks", headers={"A2A-Version": "1.0"}).status_code == 401
assert c.get("/a2a/tasks", headers={**H, "A2A-Version": "0.3"}).status_code == 400
assert c.post("/a2a/message:send", json=batch("mx", "bx"), headers={**H, "Content-Type": "application/json"}).status_code == 415

r = c.post("/a2a/message:send", json=batch("m1", "b1"), headers=H)
assert r.status_code == 200, r.text
t = r.json()["task"]
assert t["status"]["state"] == "TASK_STATE_INPUT_REQUIRED" and calls == [12]
props = t["artifacts"][0]["parts"][0]["data"]["proposals"]
assert len(props) == 12 and len(props[0]["rationale"]) >= 60 and len(props[0]["actionId"]) >= 12

# idempotent replay (changed config), then conflict, then second batch uses cache (same content, new ids)
b = batch("m1", "b1")
b["configuration"]["returnImmediately"] = True
assert c.post("/a2a/message:send", json=b, headers=H).json()["task"] == t and calls == [12]
assert c.post("/a2a/message:send", json=batch("m1", "other"), headers=H).status_code == 409

# isolation
assert c.get(f"/a2a/tasks/{t['id']}", headers=H2).status_code == 404
assert c.get("/a2a/tasks", headers=H2).json() == {"tasks": []}
assert c.post(f"/a2a/tasks/{t['id']}:cancel", headers=H2).status_code == 404

# continuation: mismatched then valid
def res(mid, mutate=None):
    rs = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
           "outcome": "ACCEPTED" if i % 2 == 0 else "REJECTED", "receiptNonce": f"n{i}"} for i, p in enumerate(props)]
    if mutate:
        mutate(rs)
    return {"message": {"messageId": mid, "taskId": t["id"], "contextId": t["contextId"], "role": "ROLE_USER",
                        "parts": [{"mediaType": A.MT_RESULTS, "data": {"batchId": "b1", "results": rs}}]}}


bad = c.post("/a2a/message:send", json=res("m2", lambda rs: rs[0].update(action="settle_invoice")), headers=H)
assert bad.status_code == 400
ok = c.post("/a2a/message:send", json=res("m3"), headers=H)
assert ok.status_code == 200
done = ok.json()["task"]
assert done["status"]["state"] == "TASK_STATE_COMPLETED"
parts = done["artifacts"][0]["parts"]
assert len(parts) == 2 and len(parts[1]["data"]["executions"]) == 6
assert c.post(f"/a2a/tasks/{t['id']}:cancel", headers=H).status_code == 409
assert c.post("/a2a/message:send", json=res("m3"), headers=H).json()["task"] == done
assert c.get(f"/a2a/tasks/{t['id']}", headers=H).json() == done

# cancel then result -> 409
t2 = c.post("/a2a/message:send", json=batch("m4", "b2"), headers=H).json()["task"]
assert calls == [12], calls  # cached by content despite new package ids
assert c.post(f"/a2a/tasks/{t2['id']}:cancel", headers=H).json()["status"]["state"] == "TASK_STATE_CANCELED"
print("all ok")
