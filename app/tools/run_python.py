"""Execute Python through the separate, disposable sandbox runner."""
from __future__ import annotations

import base64
import binascii
import json
from uuid import UUID

from app.core.config import get_settings
from app.tools.registry import ToolDefinition
from app.tools.sandbox.context import get_sandbox_context
from app.tools.sandbox.policy import get_policy_engine
from app.tools.sandbox.runner_client import execute_python


def _available() -> bool:
    settings = get_settings()
    return bool(settings.SANDBOX_EXEC_ENABLED and settings.SANDBOX_RUNNER_URL
                and settings.SANDBOX_RUNNER_TOKEN and get_policy_engine().mode == 'enforce')


def run_python(code: str, artifact_ids: str = '') -> str:
    if not _available():
        raise RuntimeError('Python sandbox requires an enabled runner and enforce policy')
    context = get_sandbox_context()
    if not context.user_id or not context.session_id:
        raise RuntimeError('Authenticated conversation is required for Python sandbox')
    if len(code.encode('utf-8')) > 262144:
        raise ValueError('Python code exceeds 256 KiB')
    from backend.services.artifact_service import create_generated_file, read_file

    ids = [part.strip() for part in artifact_ids.split(',') if part.strip()]
    if len(ids) > 3:
        raise ValueError('At most three input artifacts are allowed')
    inputs = []
    total = 0
    for artifact_id in ids:
        info, data = read_file(context.user_id, context.session_id, UUID(artifact_id))
        total += len(data)
        if total > 10 * 1024 * 1024:
            raise ValueError('Input artifacts exceed 10 MiB')
        inputs.append({'filename': info['filename'],
                       'content_b64': base64.b64encode(data).decode('ascii')})
    result = execute_python(code, inputs)

    artifacts = []
    for item in result['files']:
        try:
            data = base64.b64decode(item['content_b64'], validate=True)
            artifact = create_generated_file(
                context.user_id, context.session_id, item['filename'], data,
            )
            artifacts.append(artifact)
            from app.agents.events import emit
            emit('file_operation', 'file_created', '沙箱文件已生成', item['filename'],
                 status='completed', artifact=artifact)
        except (KeyError, TypeError, ValueError, binascii.Error):
            continue
    return json.dumps({
        'exit_code': result.get('exit_code'),
        'stdout': str(result.get('stdout', ''))[:12000],
        'stderr': str(result.get('stderr', ''))[:12000],
        'stdout_truncated': bool(result.get('stdout_truncated')),
        'stderr_truncated': bool(result.get('stderr_truncated')),
        'artifacts': artifacts,
    }, ensure_ascii=False)


TOOL = ToolDefinition(
    name='run_python',
    description='在隔离容器中运行 Python 3.11。会话附件 ID 可用逗号分隔，文件位于 /work/inputs；生成文件写入 /output。默认无网络，运行上限 45 秒。',
    fn=run_python,
    arg_schema={
        'code': ('string', '要执行的完整 Python 代码', True),
        'artifact_ids': ('string', '可选：当前会话附件 ID，最多三个，用逗号分隔', False),
    },
    check_fn=_available,
    timeout_s=0,
    metadata={'public': True, 'capabilities': ['proc.exec'],
              'tags': ['python', 'code', 'sandbox'],
              'scenarios': ['运行 Python', '计算数据', '生成文件']},
)
