"""Trusted artifact ownership propagated to tools, never supplied by the model."""
from contextlib import contextmanager
from contextvars import ContextVar

_owner = ContextVar("artifact_owner", default=None)


@contextmanager
def use_artifact_owner(user_id, conversation_id):
    token = _owner.set((user_id, conversation_id))
    try:
        yield
    finally:
        _owner.reset(token)


def artifact_owner():
    owner = _owner.get()
    if not owner or not all(owner):
        raise ValueError("File generation requires an authenticated conversation")
    return owner
