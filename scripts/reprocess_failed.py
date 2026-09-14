import asyncio
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select

from backend.services.ingestion_service import fetch_raw_from_minio, run_ingestion
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_knowledge import KnowledgeFile

DOCX_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


async def main():
    async with get_session() as s:
        rows = (
            await s.execute(
                select(KnowledgeFile).where(
                    KnowledgeFile.status == "failed",
                    KnowledgeFile.minio_object.isnot(None),
                )
            )
        ).scalars().all()
        items = [
            (f.id, f.knowledge_base_id, f.filename, f.minio_bucket, f.minio_object)
            for f in rows
        ]
    print("to reprocess:", len(items), flush=True)

    sem = asyncio.Semaphore(3)

    async def one(fid, kb, fn, bucket, obj):
        raw = await fetch_raw_from_minio(bucket, obj)
        if raw is None:
            print("MISSING RAW", fn, flush=True)
            return
        async with sem:
            try:
                await run_ingestion(fid, kb, raw, fn, None, DOCX_CT, "auto")
                print("done", fn, flush=True)
            except Exception as exc:
                print("FAIL", fn, repr(exc), flush=True)

    await asyncio.gather(*(one(*i) for i in items))
    print("ALL FINISHED", flush=True)


asyncio.run(main())
