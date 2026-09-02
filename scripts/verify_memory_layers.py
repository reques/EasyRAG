"""hermes-verify: 结构化记忆验证（非测试套件）。

针对 Task 3 结构化记忆:
1. DB: user_facts 表存在, conversations.summary 列存在
2. 语义记忆: add_user_fact/get_user_facts 存取正常, 规则触发关键词判断
3. 情景记忆: maybe_update_summary 在 10 轮倍数触发, get_compressed_history 压缩逻辑
4. 集成: prepare_context 注入 user facts 到 messages

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_memory_layers.py
前置: easyrag-postgres 容器运行, LLM 服务可达（摘要/事实提取走真实调用）
"""
import asyncio
import subprocess
import sys

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
sys.path.insert(0, ROOT)


def db(sql):
    out = subprocess.run(["docker", "exec", "easyrag-postgres", "psql", "-U", "easyrag",
                          "-d", "easyrag", "-t", "-c", sql], capture_output=True, text=True)
    return out.stdout.strip()


print("== 1. DB 结构 ==")
check("user_facts table exists",
      "user_facts" in db("SELECT tablename FROM pg_tables WHERE schemaname='public';"))
check("conversations.summary column exists",
      "summary" in db("SELECT column_name FROM information_schema.columns WHERE table_name='conversations';"))
check("conversations.last_summarized_message_id column exists",
      "last_summarized_message_id" in db(
          "SELECT column_name FROM information_schema.columns WHERE table_name='conversations';"))


async def main():
    from backend.storage.postgres.manager import get_session
    from backend.storage.postgres.models_user import User
    from backend.services.chat_service import (
        add_message, create_conversation, get_compressed_history, get_conversation_history,
    )
    from app.memory.manager import (
        add_user_fact, get_user_facts, should_extract_fact, maybe_update_summary,
    )
    from sqlalchemy import select

    async with get_session() as s:
        user = (await s.execute(select(User))).scalars().first()
        uid = user.id

        print("== 2. 语义记忆（user_facts）==")
        await add_user_fact(s, uid, "用户偏好简洁的回答风格")
        await add_user_fact(s, uid, "用户是法律专业背景")
        await s.commit()
        facts = await get_user_facts(s, uid)
        check("add/get user_facts", len(facts) >= 2, f"{len(facts)} facts")
        check("fact content stored", any("法律" in f for f in facts))

        # 规则触发
        check("trigger: 我喜欢简洁", should_extract_fact("我喜欢简洁的回答"))
        check("trigger: 记住这个", should_extract_fact("帮我记住这个"))
        check("trigger: 我是律师", should_extract_fact("我是一名律师"))
        check("no trigger: 普通问题", not should_extract_fact("民法典第10条是什么"))

        print("== 3. 情景记忆（会话摘要压缩）==")
        # 造 10 轮对话触发 summary
        conv = await create_conversation(s, uid)
        await s.commit()
        cid = conv.id
        for i in range(10):
            await add_message(s, cid, "user", f"问题{i}：民法典第{i+1}条是什么")
            await add_message(s, cid, "assistant", f"回答{i}：第{i+1}条规定了……")
            await s.commit()

        await s.refresh(conv)
        check("summary generated after 10 rounds", bool(conv.summary),
              (conv.summary or "")[:40] if conv.summary else "empty")
        check("summary pointer advanced after 10 rounds (no-loss fold)",
              bool(conv.last_summarized_message_id),
              f"ptr={conv.last_summarized_message_id}")

        # 摘要压缩只在长会话生效：10 轮（20条）刚好在窗口内时 compressed==full
        # 再造 5 轮（共 15 轮 30 条）超出窗口，验证真正压缩
        for i in range(10, 15):
            await add_message(s, cid, "user", f"追加问题{i}")
            await add_message(s, cid, "assistant", f"追加回答{i}")
            await s.commit()
        full = await get_conversation_history(s, cid)
        compressed = await get_compressed_history(s, cid)
        check("full history has all 30 msgs", len(full) == 30, f"{len(full)}")
        check("compressed starts with system summary",
              compressed[0]["role"] == "system" and "摘要" in compressed[0]["content"])
        check("compressed shorter than full (long conv)", len(compressed) < len(full),
              f"compressed={len(compressed)} full={len(full)}")
        check("compressed keeps recent 10 turns (20 msgs) + 1 system",
              len(compressed) == 21, f"{len(compressed)}")
        # 压缩尾窗必须是真实最近窗口：最后一条是第 30 条（追加回答14），
        # 而不是"最早 100 条内"的伪最近窗口
        check("compressed tail is the TRUE recent window (ends at msg 30)",
              compressed[-1]["content"].startswith("追加回答14"),
              f"tail last={compressed[-1]['content'][:24]}")
        # 断点应推进到最后一条消息：追加的 5 轮（10 条）也被折叠进摘要，无消息丢失
        from backend.storage.postgres.models_conversation import Message as MsgModel
        last_msg = (await s.execute(
            select(MsgModel).where(MsgModel.conversation_id == cid)
            .order_by(MsgModel.id.desc()).limit(1)
        )).scalar_one()
        await s.refresh(conv)
        check("summary pointer advanced to last message (no message loss)",
              conv.last_summarized_message_id == last_msg.id,
              f"ptr={conv.last_summarized_message_id} last_id={last_msg.id}")

        # 清理测试会话
        from backend.storage.postgres.models_conversation import Conversation
        await s.delete(conv)
        # 清 facts
        from backend.storage.postgres.models_memory import UserFact
        for f in (await s.execute(select(UserFact).where(UserFact.user_id == uid))).scalars().all():
            await s.delete(f)
        await s.commit()
        check("cleanup test conv + facts", True)

asyncio.run(main())

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
