"""Durable score delivery, isolated from chat and evaluation transactions."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from backend.storage.postgres.models_observability import ObservabilityScore

logger = logging.getLogger(__name__)


async def enqueue_score(session, *, score_id, user_id, payload):
    # Serialize against delivery acknowledgements; refresh identity-map state
    # so a concurrent accepted->pending transition is always written.
    row = await session.get(ObservabilityScore, score_id, with_for_update=True, populate_existing=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if row is None:
        row = ObservabilityScore(id=score_id, user_id=user_id, payload_json=encoded,
                                 revision=1, state="pending", attempts=0,
                                 next_attempt_at=datetime.now(timezone.utc))
        session.add(row)
    elif row.payload_json != encoded:
        row.payload_json = encoded
        row.revision += 1
        row.state = "pending"
        row.attempts = 0
        row.error_type = None
        row.next_attempt_at = datetime.now(timezone.utc)
    return row


def submit_score(client, score_id, payload):
    """Use the acknowledged API, not enqueue+flush (which cannot prove acceptance)."""
    from langfuse.api.scores.types.create_score_request import CreateScoreRequest
    from app.observability.redaction import content
    selected = {key: payload[key] for key in ("trace_id", "observation_id", "name", "value", "data_type", "environment") if key in payload}
    if payload.get("comment"):
        selected["comment"] = content(payload["comment"])
    response = client.api.scores.create(request=[CreateScoreRequest(id=score_id, **selected)],
                                        request_options={"timeout_in_seconds": 3, "max_retries": 0})
    if getattr(response, "rejected", 0) or getattr(response, "errors", None):
        raise RuntimeError("ScoreRejected")
    if getattr(response, "accepted", None) != 1 and getattr(response, "message", None) != "Accepted":
        raise RuntimeError("ScoreAcceptanceUnknown")


async def deliver_pending_scores(session_factory, *, client=None, limit=20):
    if client is None:
        from app.observability.langfuse_client import get_client
        client = get_client()
    if client is None:
        return 0
    delivered = 0
    for _ in range(limit):
        now = datetime.now(timezone.utc)
        token = uuid.uuid4().hex
        async with session_factory() as session:
            row = (await session.execute(select(ObservabilityScore).where(
                ObservabilityScore.state == "pending", ObservabilityScore.next_attempt_at <= now,
                or_(ObservabilityScore.lease_until.is_(None), ObservabilityScore.lease_until < now),
            ).order_by(ObservabilityScore.next_attempt_at).limit(1).with_for_update(skip_locked=True))).scalar_one_or_none()
            if row is None:
                break
            row.lease_token = token
            row.lease_until = now + timedelta(seconds=60)
            score_id, revision, attempts = row.id, row.revision, row.attempts
            payload = json.loads(row.payload_json)
            await session.commit()
        error = None
        try:
            await asyncio.to_thread(submit_score, client, score_id, payload)
            delivered += 1
        except Exception as exc:
            error = type(exc).__name__
        async with session_factory() as session:
            row = await session.get(ObservabilityScore, score_id, with_for_update=True)
            if row is None or row.lease_token != token:
                continue
            row.lease_until = row.lease_token = None
            if row.revision == revision:
                row.attempts = attempts + 1
                row.state = "accepted" if error is None else "pending"
                row.error_type = error
                row.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=min(300, 2 ** min(row.attempts, 8)))
            await session.commit()
    return delivered


async def score_delivery_worker():
    from backend.storage.postgres.manager import get_session
    while True:
        try:
            await deliver_pending_scores(get_session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Score delivery will retry (%s)", type(exc).__name__)
        await asyncio.sleep(10)
