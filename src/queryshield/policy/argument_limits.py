"""Argument limits of the model's tool and action calls.

The proposal validator (agent.proposals) enforces them, and the native function
schemas (agent.context) and the MCP tool schemas (mcp_metadata.schemas) declare
them, so the three cannot drift.  This module imports nothing from the
package: both sides can depend on it without an import cycle.
"""

SEARCH_QUERY_MAX_CHARS = 200
TOP_K_RANGE = (1, 5)
TOP_K_DEFAULT = 3
DESCRIBE_TABLES_RANGE = (1, 3)
SQL_MAX_CHARS = 4000
ANSWER_MAX_CHARS = 4000
LIST_ITEM_MAX_CHARS = 4000
QUESTION_MAX_CHARS = 1000
REASON_MAX_CHARS = 1000
CLARIFICATION_ID_MAX_CHARS = 100
MAX_SOURCE_IDS = 32
MAX_FACT_REFS = 10
