"""Risky-action gate.

Read-only tools (search, get_open_issues, get_recent_commits) execute
immediately. Anything that would *write* to the outside world (closing an
issue, posting a comment, merging a PR) is modeled here as a RiskyAction:
the agent can PROPOSE one, but it is never auto-executed. A human has to
call `.confirm()` on it. This is the "prepare the plan but wait before
execution" pattern from the video, implemented as an actual code boundary
rather than a prompt instruction (which an injected issue body could try to
talk the model out of).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class ActionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class RiskyAction:
    kind: str  # e.g. "close_issue", "post_comment", "merge_pr"
    target: str  # e.g. "issue:482"
    payload: dict
    reason: str
    status: ActionStatus = ActionStatus.PENDING
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def confirm(self) -> None:
        if self.status != ActionStatus.PENDING:
            raise ValueError(f"Cannot confirm action in status {self.status}")
        self.status = ActionStatus.CONFIRMED

    def reject(self) -> None:
        if self.status != ActionStatus.PENDING:
            raise ValueError(f"Cannot reject action in status {self.status}")
        self.status = ActionStatus.REJECTED

    def mark_executed(self) -> None:
        if self.status != ActionStatus.CONFIRMED:
            raise ValueError("Only a confirmed action may be executed")
        self.status = ActionStatus.EXECUTED

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "payload": self.payload,
            "reason": self.reason,
            "status": self.status.value,
        }


class RiskyActionRegistry:
    """Holds proposed actions for a session so a human (or the eval harness,
    simulating one) can review and confirm/reject them out-of-band from the
    agent's own generation loop.
    """

    def __init__(self) -> None:
        self._actions: dict[str, RiskyAction] = {}

    def propose(self, kind: str, target: str, payload: dict, reason: str) -> RiskyAction:
        action = RiskyAction(kind=kind, target=target, payload=payload, reason=reason)
        self._actions[action.id] = action
        return action

    def get(self, action_id: str) -> RiskyAction | None:
        return self._actions.get(action_id)

    def pending(self) -> list[RiskyAction]:
        return [a for a in self._actions.values() if a.status == ActionStatus.PENDING]

    def all(self) -> list[RiskyAction]:
        return list(self._actions.values())
