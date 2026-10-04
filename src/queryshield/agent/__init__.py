"""Server-owned contracts for the W02 model proposal boundary."""

from queryshield.agent.proposals import (
    CallIdentityError,
    CallAttemptKind,
    DenyAction,
    ExecutionContext,
    FactRef,
    FinalAnswerAction,
    MetricBinding,
    ModelCallIdentity,
    ModelCallStore,
    ParallelReadonlyAction,
    ProposalAction,
    ProposalParseError,
    QueryProposal,
    ResultEvidence,
    AskUserAction,
    ToolCallAction,
    parse_query_proposal,
)
from queryshield.agent.call_store import DurableModelCallStore
from queryshield.agent.config import (
    ACTION_SCHEMA_VERSION,
    DEFAULT_ADAPTER_VERSION,
    DEFAULT_CATALOG_VERSION,
    DEFAULT_KNOWLEDGE_SNAPSHOT_ID,
    DEFAULT_MODEL_VERSION,
    DEFAULT_PROFILE,
    DEFAULT_RUN_CONFIG,
    RUN_CONFIG_VERSION,
    SYSTEM_PROMPT_VERSION,
    TOOL_DESCRIPTION_VERSION,
    RunConfig,
)
from queryshield.agent.context import (
    CONTEXT_VERSION,
    MAX_CONTEXT_BYTES,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    ContextBuildError,
    ContextBudgetError,
    ContextBuildResult,
    ContextMessage,
    build_context,
)

# graph.py imports the semantic tools, and semantic.py imports db.guarded.
# Keep these heavier runtime exports lazy so importing db.readonly/check_db.py
# cannot re-enter a partially initialized db.guarded module.  The public names
# remain unchanged: ``from queryshield.agent import BoundedAgent`` still works
# through module-level ``__getattr__`` below.
_GRAPH_EXPORTS = frozenset(
    {
        "AgentRunResult",
        "BoundedAgent",
        "GraphLimits",
        "MAX_QUERY_REPAIRS",
        "MAX_MODEL_CALLS",
        "MAX_TOOL_CALLS",
        "MAX_WALL_CLOCK_SECONDS",
        "RunResumeError",
    }
)
_PARALLEL_EXPORTS = frozenset(
    {
        "BranchExecution",
        "MAX_ACTIVE_BRANCHES",
        "MAX_PARALLEL_BRANCHES",
        "PARALLEL_VERSION",
        "ParallelBranchResult",
        "ParallelGroupStore",
        "ParallelPlan",
        "ParallelPlanConflict",
        "ParallelRunResult",
        "ParallelScheduler",
        "ParallelValidationError",
    }
)


def __getattr__(name: str):
    if name in _GRAPH_EXPORTS:
        from queryshield.agent import graph

        value = getattr(graph, name)
        globals()[name] = value
        return value
    if name in _PARALLEL_EXPORTS:
        from queryshield.agent import parallel

        value = getattr(parallel, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AskUserAction",
    "AgentRunResult",
    "ACTION_SCHEMA_VERSION",
    "BoundedAgent",
    "BranchExecution",
    "CallAttemptKind",
    "CallIdentityError",
    "CONTEXT_VERSION",
    "ContextBuildError",
    "ContextBudgetError",
    "ContextBuildResult",
    "ContextMessage",
    "DEFAULT_ADAPTER_VERSION",
    "DEFAULT_CATALOG_VERSION",
    "DEFAULT_KNOWLEDGE_SNAPSHOT_ID",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_PROFILE",
    "DEFAULT_RUN_CONFIG",
    "DenyAction",
    "DurableModelCallStore",
    "ExecutionContext",
    "FactRef",
    "FinalAnswerAction",
    "GraphLimits",
    "MAX_QUERY_REPAIRS",
    "MAX_ACTIVE_BRANCHES",
    "MAX_PARALLEL_BRANCHES",
    "MetricBinding",
    "ModelCallIdentity",
    "ModelCallStore",
    "ParallelReadonlyAction",
    "ProposalAction",
    "ProposalParseError",
    "PARALLEL_VERSION",
    "ParallelBranchResult",
    "ParallelGroupStore",
    "ParallelPlan",
    "ParallelPlanConflict",
    "ParallelRunResult",
    "ParallelScheduler",
    "ParallelValidationError",
    "QueryProposal",
    "RUN_CONFIG_VERSION",
    "ResultEvidence",
    "RunConfig",
    "SYSTEM_PROMPT_VERSION",
    "ToolCallAction",
    "TOOL_DESCRIPTION_VERSION",
    "MAX_CONTEXT_BYTES",
    "MAX_MESSAGE_CHARS",
    "MAX_MESSAGES",
    "MAX_MODEL_CALLS",
    "MAX_TOOL_CALLS",
    "MAX_WALL_CLOCK_SECONDS",
    "RunResumeError",
    "build_context",
    "parse_query_proposal",
]
