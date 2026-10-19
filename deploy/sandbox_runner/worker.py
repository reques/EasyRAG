"""Protocol used inside a disposable Python execution container.

The runner supplies one JSON request on stdin and receives one JSON result on
stdout. User stdout is captured separately, so it cannot corrupt the protocol.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path('/work')
OUTPUT = Path('/output')
MAX_OUTPUT = 64 * 1024
MAX_FILE = 5 * 1024 * 1024
MAX_TOTAL = 10 * 1024 * 1024
MAX_REQUEST = 16 * 1024 * 1024
ALLOWED_SUFFIXES = {'.txt', '.md', '.csv', '.json', '.html', '.pdf', '.docx', '.xlsx'}


def main() -> None:
    request = json.loads(sys.stdin.buffer.readline(MAX_REQUEST + 1))
    code = request['code']
    if not isinstance(code, str) or not code or len(code.encode('utf-8')) > MAX_FILE:
        raise ValueError('Invalid code size')
    WORK.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    inputs = WORK / 'inputs'
    inputs.mkdir(exist_ok=True)
    input_total = 0
    for item in request.get('inputs', []):
        filename = item['filename']
        if (not isinstance(filename, str) or filename in {'.', '..'}
                or '/' in filename or '\\' in filename or filename.startswith('.')):
            raise ValueError('Invalid input filename')
        data = base64.b64decode(item['content_b64'], validate=True)
        input_total += len(data)
        if len(data) > MAX_FILE or input_total > MAX_TOTAL:
            raise ValueError('Input files exceed size limit')
        (inputs / filename).write_bytes(data)
    script = WORK / 'main.py'
    script.write_text(code, encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-I', '-S', '-B', str(script)],
        cwd=WORK, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=45, check=False,
        env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/work',
             'PYTHONDONTWRITEBYTECODE': '1'},
    )
    files = []
    total = 0
    for item in sorted(OUTPUT.iterdir()):
        if item.is_symlink() or not item.is_file() or item.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        size = item.stat().st_size
        if size > MAX_FILE or total + size > MAX_TOTAL:
            continue
        data = item.read_bytes()
        total += len(data)
        files.append({'filename': item.name, 'content_b64': base64.b64encode(data).decode('ascii')})
    print(json.dumps({
        'exit_code': result.returncode,
        'stdout': result.stdout[:MAX_OUTPUT].decode('utf-8', 'replace'),
        'stderr': result.stderr[:MAX_OUTPUT].decode('utf-8', 'replace'),
        'stdout_truncated': len(result.stdout) > MAX_OUTPUT,
        'stderr_truncated': len(result.stderr) > MAX_OUTPUT,
        'files': files,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
