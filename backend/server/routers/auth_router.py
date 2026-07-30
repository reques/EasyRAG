"""认证路由 — 注册 / 登录 / Token 刷新。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.auth_service import (
    AuthError,
    register_user,
    authenticate_user,
    create_access_token,
)
from backend.storage.postgres.manager import get_session

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request / Response ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    email: str | None = None
    display_name: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """注册新账户并返回 Token。"""
    async with get_session() as session:
        try:
            user = await register_user(
                session,
                username=req.username,
                password=req.password,
                email=req.email,
                display_name=req.display_name,
            )
            await session.commit()
            token = create_access_token(user.id, user.username)
            return TokenResponse(
                access_token=token,
                username=user.username,
                user_id=str(user.id),
            )
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """登录并获取 JWT Token。"""
    async with get_session() as session:
        try:
            user = await authenticate_user(session, req.username, req.password)
            token = create_access_token(user.id, user.username)
            return TokenResponse(
                access_token=token,
                username=user.username,
                user_id=str(user.id),
            )
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
