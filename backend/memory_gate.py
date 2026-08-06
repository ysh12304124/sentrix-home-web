"""Semantic routing gate for the Thin Agent — R9 thin wrapper.

R9 moved the routing decision into :class:`Router` (backend/router.py).  This
module keeps the ``GateDecision`` contract and a thin compatibility shell so
legacy callers and tests keep working: ``fast_path`` delegates to
:class:`ExplicitOperationDetector`, ``classify`` delegates to ``Router.route``.

No word list or semantic classifier lives here anymore — all structural rules
are single-sourced in ``backend/routing_rules.py`` and decided in
``backend/router.py``.
"""

from __future__ import annotations

from .router import ExplicitOperationDetector, GateDecision, Router


class MemoryGate:
    """Thin compatibility shell over the R9 Router (no model call)."""

    def __init__(self, router=None, entity_resolver=None):
        self._router = router or Router(entity_resolver=entity_resolver)
        self._detector = ExplicitOperationDetector()

    def fast_path(self, message, *, api_signals=None):
        return self._detector.detect(message, api_signals=api_signals)

    def classify(self, message, conversation="", *, draft=None, api_signals=None,
                 proactive_enabled=False):
        decision = self._router.route(message, draft, api_signals=api_signals,
                                      conversation=conversation)
        return decision.as_gate_decision()


__all__ = ["MemoryGate", "GateDecision", "ExplicitOperationDetector", "Router"]
