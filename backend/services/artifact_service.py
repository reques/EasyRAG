"""Private generated files, with durable metadata stored on the same MinIO object."""
import io
import json
import re
from pathlib import PurePosixPath
from urllib.parse import quote, unquote
from uuid import UUID, uuid4

from app.core.config import get_settings
from backend.storage.minio.client import get_minio_client, ensure_bucket

MAX_BYTES = 5 * 1024 * 1024
MIME_TYPES = {
    '.md': 'text/markdown', '.txt': 'text/plain', '.csv': 'text/csv',
    '.json': 'application/json', '.html': 'text/html',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.pdf': 'application/pdf',
}


def normalize_filename(filename):
    if isinstance(filename, str):
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix in {'.doc', '.xls'}:
            return filename[:-len(suffix)] + {'.doc': '.docx', '.xls': '.xlsx'}[suffix]
    return filename


def prefix(user_id, conversation_id):
    return f"artifacts/{UUID(str(user_id))}/{UUID(str(conversation_id))}/"


def prepare_file(filename, content):
    filename = normalize_filename(filename)
    if not isinstance(filename, str) or not filename or len(filename.encode('utf-8')) > 180:
        raise ValueError('Filename must be 1–180 UTF-8 bytes')
    if re.search(r'[\\/:*?"<>|\x00-\x1f\x7f]', filename) or filename.startswith('.') or filename.endswith((' ', '.')):
        raise ValueError('Use a filename without directories or control characters')
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in MIME_TYPES:
        raise ValueError('Supported formats: ' + ', '.join(MIME_TYPES))
    if not isinstance(content, str) or not content.strip():
        raise ValueError('File content must not be empty')
    data = content.encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError('File exceeds the 5 MiB limit')
    if extension == '.json':
        json.loads(content)
    if extension in {'.docx', '.xlsx', '.pdf'}:
        from backend.services.artifact_formats import write_docx, write_xlsx, write_pdf
        data = {'.docx': write_docx, '.xlsx': write_xlsx, '.pdf': write_pdf}[extension](content)
        if len(data) > MAX_BYTES:
            raise ValueError('Generated file exceeds the 5 MiB limit')
    return data, MIME_TYPES[extension]


def describe(item, conversation_id):
    metadata = {k.lower(): v for k, v in (item.metadata or {}).items()}
    filename = unquote(metadata.get('x-amz-meta-filename') or metadata.get('filename') or 'download.txt')
    artifact_id = item.object_name.rsplit('/', 1)[-1]
    return {
        'id': artifact_id, 'kind': 'file', 'title': filename, 'filename': filename,
        'size': item.size, 'content_type': getattr(item, 'content_type', None) or MIME_TYPES.get(PurePosixPath(filename).suffix.lower(), 'application/octet-stream'),
        'download_url': f'/artifacts/{conversation_id}/{artifact_id}',
    }


def create_file(user_id, conversation_id, filename, content):
    filename = normalize_filename(filename)
    data, mime = prepare_file(filename, content)
    object_name = prefix(user_id, conversation_id) + str(uuid4())
    bucket = get_settings().MINIO_BUCKET
    ensure_bucket(bucket)
    client = get_minio_client()
    client.put_object(bucket, object_name, io.BytesIO(data), len(data), content_type=mime,
                      metadata={'filename': quote(filename, safe='')})
    return {'id': object_name.rsplit('/', 1)[-1], 'kind': 'file', 'title': filename,
            'filename': filename, 'size': len(data), 'content_type': mime,
            'download_url': f'/artifacts/{conversation_id}/{object_name.rsplit("/", 1)[-1]}'}


def list_files(user_id, conversation_id):
    return [describe(item, conversation_id) for item in get_minio_client().list_objects(
        get_settings().MINIO_BUCKET, prefix=prefix(user_id, conversation_id),
        recursive=True, include_user_meta=True)]


def read_file(user_id, conversation_id, artifact_id):
    client = get_minio_client()
    bucket = get_settings().MINIO_BUCKET
    name = prefix(user_id, conversation_id) + str(UUID(str(artifact_id)))
    info = client.stat_object(bucket, name)
    if info.size > MAX_BYTES:
        raise ValueError('Stored file exceeds download limit')
    response = client.get_object(bucket, name)
    try:
        return describe(info, conversation_id), response.read(MAX_BYTES + 1)
    finally:
        response.close()
        response.release_conn()
