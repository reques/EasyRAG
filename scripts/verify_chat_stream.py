"""端到端验证 /chat/stream SSE 端点 + 引用 file_id 透出。"""
import sys, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.auth_service import create_access_token
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_user import User
from sqlalchemy import select
import asyncio

async def get_admin():
    async with get_session() as s:
        return (await s.execute(select(User))).scalars().first()

admin = asyncio.run(get_admin())
token = create_access_token(admin.id, admin.username)

req = urllib.request.Request(
    "http://localhost:8000/api/v1/chat/stream",
    data=json.dumps({"query": "SkelHCC 是什么？简要介绍一下", "conversation_id": None}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    method="POST",
)

deltas = 0
sources = None
done = False
conv_id = None
with urllib.request.urlopen(req, timeout=120) as resp:
    buf = b""
    for chunk in resp:
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            line = ev.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                t = payload.get("type")
                if t == "conversation_id":
                    conv_id = payload["conversation_id"]
                elif t == "delta":
                    deltas += 1
                    if deltas == 1:
                        print("[stream] first delta received")
                elif t == "done":
                    sources = payload.get("sources")
                    done = True
                    print(f"[stream] done, elapsed={payload.get('elapsed_seconds')}s")
                elif t == "error":
                    print("[stream] ERROR:", payload.get("detail"))

print("---")
print("conversation_id:", conv_id)
print("delta events:", deltas)
print("done:", done)
print("sources:", json.dumps(sources, ensure_ascii=False, indent=2))
