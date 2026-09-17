"""Anthropic tool-use schemas for the tools in sentinelrag.tools.repo_tools."""

TOOLS = [
    {
        "name": "search_repo",
        "description": (
            "Semantic-ish (BM25 lexical) search over the repo's issues, pull requests and "
            "commits. Use this to find evidence relevant to a specific sub-question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "source_type": {
                    "type": "string",
                    "enum": ["issue", "pull_request", "commit"],
                    "description": "Optionally restrict to one evidence type.",
                },
                "top_k": {"type": "integer", "description": "Max results, default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_recent_commits",
        "description": "List the N most recently updated commits, most recent first.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "How many, default 5."}},
        },
    },
    {
        "name": "list_open_issues",
        "description": "List currently open issues, optionally filtered by label (e.g. 'bug', 'blocker').",
        "input_schema": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
        },
    },
    {
        "name": "recall_memory",
        "description": "Look up a previously remembered durable fact by key (long-term memory across sessions).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "Persist a durable fact to long-term memory for future sessions. Only use this for "
            "genuinely durable facts (e.g. 'project X's current priority is Y'), never transient "
            "details of this single question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
    {
        "name": "propose_risky_action",
        "description": (
            "Propose (never directly execute) a write action against the repo, e.g. closing an "
            "issue or merging a PR. This only queues the action for human confirmation -- it is "
            "never auto-executed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "e.g. 'close_issue', 'post_comment', 'merge_pr'"},
                "target": {"type": "string", "description": "e.g. 'issue:482'"},
                "reason": {"type": "string"},
            },
            "required": ["kind", "target", "reason"],
        },
    },
]
