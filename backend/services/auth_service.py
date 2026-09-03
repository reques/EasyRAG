"""认证服务 — 注册/登录/Token 签发。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import AgentError
from backend.repositories.user_repository import UserRepository
from backend.storage.postgres.models_user import User

cfg = get_settings()


class AuthError(AgentError):
    """认证相关错误。"""
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: uuid.UUID, username: str) -> str:
    """生成 JWT Token。"""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=cfg.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, cfg.JWT_SECRET_KEY, algorithm=cfg.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """校验并解码 JWT Token，返回 payload。"""
    try:
        return jwt.decode(token, cfg.JWT_SECRET_KEY, algorithms=[cfg.JWT_ALGORITHM])
    except JWTError as exc:
        raise AuthError(f"Invalid or expired token: {exc}")


async def register_user(
    session: AsyncSession,
    username: str,
    password: str,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
) -> User:
    """注册新用户。"""
    repo = UserRepository(session)
    existing = await repo.get_by_username(username)
    if existing:
        raise AuthError(f"Username '{username}' already exists")
    if email:
        existing_email = await repo.get_by_email(email)
        if existing_email:
            raise AuthError(f"Email '{email}' already registered")

    user = User(
        username=username,
        email=email,
        display_name=display_name or username,
        hashed_password=hash_password(password),
    )
    await repo.add(user)
    return user


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User:
    """用户认证，返回 User 或抛出 AuthError。"""
    repo = UserRepository(session)
    user = await repo.get_by_username(username)
    if not user:
        raise AuthError("Invalid username or password")
    if not user.is_active:
        raise AuthError("Account is disabled")
    if not verify_password(password, user.hashed_password):
        raise AuthError("Invalid username or password")

    # 更新最后登录时间
    user.last_login_at = datetime.now(timezone.utc)
    await session.flush()
    return user
