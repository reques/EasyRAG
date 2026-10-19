"""Small CONNECT-only egress proxy for the pinned 12306 MCP server."""
from __future__ import annotations

import ipaddress
import select
import socket
import socketserver

ALLOWED_HOSTS = frozenset({'kyfw.12306.cn', 'search.12306.cn', 'www.12306.cn'})
MAX_TUNNEL_BYTES = 20 * 1024 * 1024


def public_addresses(host: str) -> list[tuple[int, tuple]]:
    addresses = []
    for family, _, _, _, address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if ipaddress.ip_address(address[0]).is_global:
            addresses.append((family, address))
    return addresses


class ProxyHandler(socketserver.StreamRequestHandler):
    rbufsize = 0  # Do not prefetch TLS bytes beyond the CONNECT headers.

    def handle(self) -> None:
        self.connection.settimeout(10)
        line = self.rfile.readline(4097)
        if len(line) > 4096:
            return
        try:
            method, authority, version = line.decode('ascii').strip().split(' ')
            host, port = authority.rsplit(':', 1)
        except (UnicodeDecodeError, ValueError):
            return
        host = host.lower().rstrip('.')
        if (method != 'CONNECT' or version not in {'HTTP/1.0', 'HTTP/1.1'}
                or host not in ALLOWED_HOSTS or port != '443'):
            self.wfile.write(b'HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n')
            return
        header_size = 0
        while True:
            header = self.rfile.readline(4097)
            header_size += len(header)
            if not header or header in {b'\r\n', b'\n'}:
                break
            if len(header) > 4096 or header_size > 8192:
                return
        try:
            addresses = public_addresses(host)
            if not addresses:
                raise OSError('No public address')
            remote = None
            for family, address in addresses:
                try:
                    remote = socket.socket(family, socket.SOCK_STREAM)
                    remote.settimeout(5)
                    remote.connect(address)
                    break
                except OSError:
                    if remote is not None:
                        remote.close()
                    remote = None
            if remote is None:
                raise OSError('Connection failed')
        except OSError:
            self.wfile.write(b'HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n')
            return
        with remote:
            self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            self.wfile.flush()
            remote.settimeout(None)
            self.connection.settimeout(None)
            total = 0
            while total < MAX_TUNNEL_BYTES:
                readable, _, _ = select.select([self.connection, remote], [], [], 30)
                if not readable:
                    break
                for source in readable:
                    data = source.recv(min(65536, MAX_TUNNEL_BYTES - total))
                    if not data:
                        return
                    (remote if source is self.connection else self.connection).sendall(data)
                    total += len(data)


class ProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    with ProxyServer(('0.0.0.0', 8080), ProxyHandler) as server:
        server.serve_forever()
