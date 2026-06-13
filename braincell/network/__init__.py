"""First-class network declarations and runtime."""

from .core import Connection, NetworkRunResult, Population
from .edges import EdgeMethod, EdgeSet, all_pairs, dense, pairs, probability
from .engine import Network
from .lowering import ConnectionBlock, lower_connections
from .projections import (
    ContactMethod,
    ContactTable,
    Projection,
    ProjectionContactContext,
    ProjectionEdgeContext,
    by_post,
    explicit_contacts,
    per_edge,
)

__all__ = [
    "Connection",
    "ConnectionBlock",
    "ContactMethod",
    "ContactTable",
    "EdgeMethod",
    "EdgeSet",
    "Network",
    "NetworkRunResult",
    "Population",
    "Projection",
    "ProjectionContactContext",
    "ProjectionEdgeContext",
    "all_pairs",
    "by_post",
    "dense",
    "explicit_contacts",
    "lower_connections",
    "pairs",
    "per_edge",
    "probability",
]
