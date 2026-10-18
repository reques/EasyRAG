"""Manage the opt-in local stack without printing credentials or rotating them.

python scripts/local_langfuse.py init
python scripts/local_langfuse.py up
python scripts/local_langfuse.py connect
python scripts/local_langfuse.py app build backend frontend
python scripts/local_langfuse.py app up -d --no-deps backend frontend
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "deploy/langfuse/.env"


def read_env(path: Path) -> dict[str, str]:
    # This helper only reads the unquoted, generated deployment file.
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#") and "=" in line)


def init() -> None:
    if ENV.exists():
        print("Existing local configuration retained; no credentials rotated.")
        return
    config = {
        "LANGFUSE_WEB_PORT": "3000", "LANGFUSE_BLOB_PORT": "9190",
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "CLICKHOUSE_PASSWORD": secrets.token_hex(24),
        "REDIS_AUTH": secrets.token_hex(24),
        "MINIO_ROOT_PASSWORD": secrets.token_hex(24),
        "SALT": secrets.token_hex(32), "ENCRYPTION_KEY": secrets.token_hex(32),
        "NEXTAUTH_SECRET": secrets.token_hex(32),
        "LANGFUSE_INIT_ORG_ID": str(uuid.uuid4()),
        "LANGFUSE_INIT_PROJECT_ID": str(uuid.uuid4()),
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "pk-lf-" + str(uuid.uuid4()),
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": "sk-lf-" + str(uuid.uuid4()),
        "LANGFUSE_INIT_USER_EMAIL": "admin@easyrag.local",
        "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(30),
    }
    # Exclusive creation avoids replacing an existing deployment's keys.
    with ENV.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("# Local secrets. Do not commit or paste this file into logs.\n")
        handle.writelines(f"{k}={v}\n" for k, v in config.items())
    if os.name == "posix":
        ENV.chmod(0o600)
    print(f"Created local credentials at {ENV}; login email: {config['LANGFUSE_INIT_USER_EMAIL']}")


def connect() -> None:
    from urllib.request import Request, urlopen

    config = read_env(ENV)
    # Check the actual stack before turning on telemetry in the application.
    web = f"http://localhost:{config['LANGFUSE_WEB_PORT']}"
    with urlopen(web + "/api/public/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Local Langfuse is not healthy")
    auth = base64.b64encode((config["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"] + ":" +
                            config["LANGFUSE_INIT_PROJECT_SECRET_KEY"]).encode()).decode()
    with urlopen(Request(web + "/api/public/projects", headers={"Authorization": "Basic " + auth}), timeout=10) as response:
        projects = json.load(response).get("data", [])
    if config["LANGFUSE_INIT_PROJECT_ID"] not in {item["id"] for item in projects}:
        raise RuntimeError("Project initialization or credentials do not match this stack")
    updates = {
        "LANGFUSE_ENABLED": "true",
        "LANGFUSE_PUBLIC_KEY": config["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
        "LANGFUSE_SECRET_KEY": config["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
        "LANGFUSE_BASE_URL": "http://langfuse-web:3000",
        "LANGFUSE_TRACING_ENVIRONMENT": "local",
        "LANGFUSE_SAMPLE_RATE": "1.0",
        "EASYRAG_OBSERVABILITY_CONTENT_MODE": "metadata_only",
        "EASYRAG_OBSERVABILITY_WEB_BASE_URL": web,
        "EASYRAG_OBSERVABILITY_PROJECT_ID": config["LANGFUSE_INIT_PROJECT_ID"],
    }
    target = ROOT / ".env"
    original = target.read_bytes() if target.exists() else b""
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    pending = dict(updates)
    lines = []
    for line in original.splitlines():
        key = line.split(b"=", 1)[0].decode("utf-8", errors="replace").strip()
        if key in updates:
            if key in pending:
                lines.append(f"{key}={pending.pop(key)}".encode())
            # Remove only duplicate keys managed by this command.
        else:
            lines.append(line)
    if pending:
        lines.extend([b"", b"# Local Langfuse (managed by scripts/local_langfuse.py connect)"])
        lines.extend(f"{k}={v}".encode() for k, v in pending.items())
    target.write_bytes(newline.join(lines) + newline)
    print("Local observability enabled in .env. Recreate backend using the Langfuse overlay.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "up", "status", "stop", "connect", "app"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    if opts.command == "init":
        init()
        return
    if not ENV.exists():
        parser.error("Run init first")
    if opts.command == "connect":
        connect()
        return
    if opts.command == "app":
        if not opts.args:
            parser.error("app requires a Docker Compose command")
        command = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml")]
        local = ROOT / "docker-compose.override.yml"
        if local.exists():
            command += ["-f", str(local)]
        command += ["-f", str(ROOT / "docker-compose.langfuse.yml"), *opts.args]
    else:
        command = ["docker", "compose", "--env-file", str(ENV), "-f", str(ENV.parent / "compose.yml")]
        command += {"up": ["up", "-d", "--wait", "--wait-timeout", "300"], "status": ["ps"], "stop": ["stop"]}[opts.command]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
