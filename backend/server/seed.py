"""阶段 1 种子脚本 — 创建初始管理员账户和默认部门。

用法::

    python -m backend.server.seed
"""

from __future__ import annotations

import asyncio
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.storage.postgres.manager import init_db, get_session
from backend.storage.postgres.models_user import Department, User
from backend.repositories.user_repository import UserRepository
from backend.services.auth_service import hash_password


async def seed():
    """创建默认管理员和默认部门。"""
    print("🔧 初始化数据库表...")
    await init_db()

    async with get_session() as session:
        # 创建默认部门
        dept_repo = None
        stmt = None

        # 检查是否已有部门
        from sqlalchemy import select as sa_select, func
        result = await session.execute(sa_select(func.count()).select_from(Department))
        dept_count = result.scalar_one()

        if dept_count == 0:
            default_dept = Department(
                name="默认部门",
                description="系统默认部门",
            )
            session.add(default_dept)
            await session.flush()
            print(f"✅ 创建默认部门: {default_dept.name}")
        else:
            result = await session.execute(
                sa_select(Department).limit(1)
            )
            default_dept = result.scalar_one()
            print(f"ℹ️  部门已存在: {default_dept.name}")

        # 创建管理员
        user_repo = UserRepository(session)
        admin = await user_repo.get_by_username("admin")
        if not admin:
            admin = User(
                username="admin",
                email="admin@easyrag.local",
                display_name="管理员",
                hashed_password=hash_password("admin123"),
                role="admin",
                is_superuser=True,
                department_id=default_dept.id,
            )
            session.add(admin)
            await session.flush()
            print("✅ 创建管理员账户: admin / admin123")
        else:
            print("ℹ️  管理员账户已存在")

        await session.commit()
        print("✅ 种子数据初始化完成")
        print()
        print("📋 登录信息:")
        print("   用户名: admin")
        print("   密码:   admin123")


if __name__ == "__main__":
    asyncio.run(seed())
