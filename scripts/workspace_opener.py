"""Loopback-only Windows Explorer bridge. No arbitrary paths or shell commands over HTTP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid

import httpx


class OpenerError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


def loopback_url(value, *, origin=False):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or (origin and parsed.path not in ('', '/'))):
        raise ValueError('Only explicit HTTP loopback addresses are supported')
    return value.rstrip('/')


def local_directory(value):
    if not isinstance(value, str) or any(c in value for c in ('\x00', '\n', '\r')):
        raise OpenerError(400, '项目目录格式不正确')
    candidate = PureWindowsPath(value)
    if not candidate.is_absolute() or len(candidate.drive) != 2 or candidate.drive[1] != ':':
        raise OpenerError(400, '仅支持本机磁盘的绝对目录，不支持网络共享或设备路径')
    resolved = Path(value).resolve()
    drive = PureWindowsPath(str(resolved)).drive
    if len(drive) != 2 or drive[1] != ':':
        raise OpenerError(400, '目录不能指向网络共享或设备路径')
    if not resolved.is_dir():
        raise OpenerError(404, '本机项目目录不存在，请在“编辑项目”中检查路径')
    return resolved


class WorkspaceOpener:
    def __init__(self, backend='http://127.0.0.1:8000/api/v1', *, client=None, launcher=None):
        self.backend = loopback_url(backend)
        self.client = client or httpx.Client(timeout=5, trust_env=False, follow_redirects=False)
        self.launcher = launcher or (lambda path: os.startfile(str(path), 'explore'))

    def open(self, payload, authorization):
        if not isinstance(payload, dict) or set(payload) != {'workspace_id'}:
            raise OpenerError(400, '只接受 workspace_id，不接受客户端提供的目录或命令')
        try:
            identifier = uuid.UUID(payload['workspace_id'])
        except (ValueError, TypeError, AttributeError):
            raise OpenerError(400, 'workspace_id 格式不正确') from None
        if not authorization or not authorization.startswith('Bearer ') or len(authorization) > 8192:
            raise OpenerError(401, '请先登录 EasyRAG')
        try:
            response = self.client.get(f'{self.backend}/chat/workspaces/{identifier}', headers={'Authorization': authorization})
        except (httpx.HTTPError, ValueError):
            raise OpenerError(502, '无法连接本机 EasyRAG 后端，请检查服务状态') from None
        if response.status_code in (401, 403, 404):
            raise OpenerError(response.status_code, '登录已失效或无权访问该项目')
        if response.status_code != 200:
            raise OpenerError(502, '本机 EasyRAG 后端暂时不可用')
        try:
            path = response.json().get('local_path')
        except (ValueError, AttributeError):
            raise OpenerError(502, '项目配置响应无效') from None
        if not path:
            raise OpenerError(409, '请先在“编辑项目”中关联本机目录')
        directory = local_directory(path)
        try:
            self.launcher(directory)
        except OSError:
            raise OpenerError(500, 'Windows 无法打开该目录，请检查本机权限') from None
        return {'opened': True, 'workspace_id': str(identifier)}


def make_server(opener, *, port=18765, origins=('http://localhost:5173', 'http://127.0.0.1:5173')):
    allowed = {loopback_url(origin, origin=True) for origin in origins}

    class Handler(BaseHTTPRequestHandler):
        server_version = 'EasyRAGWorkspaceOpener/1'

        def log_message(self, fmt, *args):
            # Never log request bodies, authorization headers or project directories.
            pass

        def reply(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            origin = self.headers.get('Origin')
            if origin in allowed:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def permitted(self, *, require_origin=True):
            actual_port = self.server.server_port
            if self.headers.get('Host') not in (f'127.0.0.1:{actual_port}', f'localhost:{actual_port}'):
                self.reply(403, {'detail': '不接受该 Host'})
                return False
            origin = self.headers.get('Origin')
            if origin not in allowed and (require_origin or origin):
                self.reply(403, {'detail': '不接受该网页来源'})
                return False
            return True

        def do_GET(self):
            if not self.permitted(require_origin=False):
                return
            self.reply(200 if self.path == '/health' else 404,
                       {'service': 'easyrag-workspace-opener', 'status': 'ok'} if self.path == '/health' else {'detail': 'Not found'})

        def do_OPTIONS(self):
            if not self.permitted():
                return
            requested = {h.strip().lower() for h in self.headers.get('Access-Control-Request-Headers', '').split(',') if h.strip()}
            if self.path != '/open-workspace' or self.headers.get('Access-Control-Request-Method') != 'POST' or not requested <= {'authorization', 'content-type'}:
                self.reply(403, {'detail': '不接受该预检请求'})
                return
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
            self.end_headers()

        def do_POST(self):
            if not self.permitted():
                return
            if self.path != '/open-workspace':
                self.reply(404, {'detail': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise OpenerError(400, '需要小于 4KB 的 JSON 请求')
                payload = json.loads(self.rfile.read(length))
                self.reply(200, opener.open(payload, self.headers.get('Authorization')))
            except OpenerError as exc:
                self.reply(exc.status, {'detail': exc.detail})
            except (ValueError, UnicodeDecodeError):
                self.reply(400, {'detail': '无效 JSON 请求'})
            except Exception:
                self.reply(500, {'detail': '目录打开服务发生错误'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8000/api/v1')
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--origin', action='append')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('Run this service on the Windows browser host, outside Docker')
    opener = WorkspaceOpener(args.backend)
    server = make_server(opener, port=args.port, origins=args.origin or ('http://localhost:5173', 'http://127.0.0.1:5173'))
    print(f'EasyRAG workspace opener listening on 127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        opener.client.close()


if __name__ == '__main__':
    main()
