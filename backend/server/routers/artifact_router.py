"""Authenticated artifact listing and attachment downloads."""
import asyncio
from urllib.parse import quote
from uuid import UUID
from zipfile import BadZipFile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, JSONResponse
from minio.error import S3Error

from backend.server.utils.auth_middleware import get_current_user
from backend.services.chat_service import get_conversation
from backend.services.artifact_service import list_files, read_file
from backend.services.artifact_formats import preview_file
from backend.storage.postgres.manager import get_session

router = APIRouter(prefix='/artifacts', tags=['artifacts'])


async def check_owner(conversation_id, user):
    async with get_session() as session:
        conversation = await get_conversation(session, conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(404, 'Conversation not found')


@router.get('/{conversation_id}')
async def list_artifacts(conversation_id: UUID, user=Depends(get_current_user)):
    await check_owner(conversation_id, user)
    try:
        return {'artifacts': await asyncio.to_thread(list_files, user.id, conversation_id)}
    except S3Error as exc:
        if exc.code == 'NoSuchBucket':
            return {'artifacts': []}
        raise HTTPException(503, 'File storage unavailable') from exc


@router.get('/{conversation_id}/{artifact_id}')
async def download_artifact(conversation_id: UUID, artifact_id: UUID, user=Depends(get_current_user)):
    info, data = await load_owned_file(conversation_id, artifact_id, user)
    return Response(data, media_type=info['content_type'], headers={
        'Content-Disposition': "attachment; filename=download; filename*=UTF-8''" + quote(info['filename'], safe=''),
        'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'private, no-store',
        'Content-Security-Policy': "sandbox; default-src 'none'",
    })


async def load_owned_file(conversation_id, artifact_id, user):
    await check_owner(conversation_id, user)
    try:
        info, data = await asyncio.to_thread(read_file, user.id, conversation_id, artifact_id)
    except S3Error as exc:
        if exc.code in {'NoSuchKey', 'NoSuchObject', 'NoSuchBucket'}:
            raise HTTPException(404, 'File not found') from exc
        raise HTTPException(503, 'File storage unavailable') from exc
    except ValueError as exc:
        raise HTTPException(413, 'File exceeds the size limit') from exc
    return info, data


@router.get('/{conversation_id}/{artifact_id}/preview')
async def preview_artifact(conversation_id: UUID, artifact_id: UUID, user=Depends(get_current_user)):
    info, data = await load_owned_file(conversation_id, artifact_id, user)
    if info['content_type'] == 'application/pdf':
        return Response(data, media_type='application/pdf', headers={
            'Content-Disposition': 'inline', 'Cache-Control': 'private, no-store',
            'X-Content-Type-Options': 'nosniff',
        })
    try:
        preview = await asyncio.to_thread(preview_file, info['filename'], data)
    except (ValueError, KeyError, OSError, BadZipFile) as exc:
        raise HTTPException(422, '无法预览此文件，请下载查看') from exc
    return JSONResponse(preview, headers={'Cache-Control': 'private, no-store'})
