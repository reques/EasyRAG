"""验证会话删除功能：DELETE /chat/conversations/{id} 级联删消息 + 归属检查。

前提：后端运行中（默认 :8000，可通过 BASE_URL 覆盖）。
用法：D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_conversation_delete.py
"""

import os, sys, json, urllib.request, urllib.error

BASE = os.environ.get("BASE_URL", "http://localhost:8000/api/v1")
USERNAME = os.environ.get("TEST_USER", "admin")
PASSWORD = os.environ.get("TEST_PASS", "admin123")

passed = 0
failed = 0


def req(method, path, body=None, token=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(r, timeout=15)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()}


def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if condition:
        passed += 1
    else:
        failed += 1


def main():
    print("=" * 60)
    print("Conversation Delete Feature Verification")
    print(f"  BASE: {BASE}")
    print("=" * 60)

    # 1. Login
    print("\n1. Authentication")
    login = req("POST", "/auth/login", {"username": USERNAME, "password": PASSWORD})
    token = login.get("access_token")
    check("login succeeded", bool(token), f"user={login.get('username')}")
    if not token:
        print("\n  Aborting: no token")
        return 1

    # 2. Create a conversation with messages (temp user → isolated sandbox)
    print("\n2. Setup: temp user with a conversation")
    import uuid as _uuid
    temp_user = f"test_convdel_{_uuid.uuid4().hex[:8]}"
    reg = req("POST", "/auth/register",
              {"username": temp_user, "password": "test123", "department": "test"})
    tok = reg.get("access_token")
    check("temp user registered", bool(tok))
    if not tok:
        return 1

    send = req("POST", "/chat/send", {"query": "你好，这是一条用于删除测试的消息"}, token=tok)
    conv_id = send.get("conversation_id")
    check("message sent, conversation created", bool(conv_id), f"conv={str(conv_id)[:8]}...")

    hist = req("GET", f"/chat/conversations/{conv_id}/history", token=tok)
    n_msgs = len(hist.get("messages", []))
    check("conversation has messages", n_msgs >= 2, f"{n_msgs} msgs")

    # 3. Cross-user delete must fail (404)
    print("\n3. Authorization (cross-user guard)")
    cross = req("DELETE", f"/chat/conversations/{conv_id}", token=token)
    check("cross-user delete rejected (404)", cross.get("error") == 404)

    # 4. Owner deletes conversation
    print("\n4. DELETE /chat/conversations/{id}")
    del_resp = req("DELETE", f"/chat/conversations/{conv_id}", token=tok)
    check("delete returns deleted=true", del_resp.get("deleted") is True)

    # 5. Verify gone from list
    convs = req("GET", "/chat/conversations", token=tok)
    gone = not any(c["id"] == conv_id for c in convs)
    check("conversation removed from list", gone, f"{len(convs)} convs remain")

    # 6. Verify history 404 (messages cascade-deleted with conversation)
    hist2 = req("GET", f"/chat/conversations/{conv_id}/history", token=tok)
    check("history endpoint returns 404 after delete", hist2.get("error") == 404)

    # 7. Deleting again → 404 (idempotent-ish semantics)
    del2 = req("DELETE", f"/chat/conversations/{conv_id}", token=tok)
    check("second delete returns 404", del2.get("error") == 404)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
