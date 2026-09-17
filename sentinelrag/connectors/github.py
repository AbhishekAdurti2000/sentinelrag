"""Live data connector: pulls issues, pull requests and commits from a GitHub
repo via the REST API.

This is deliberately "live" rather than a static snapshot: every call to
``refresh()`` re-fetches from the API, so questions like "what changed this
week" reflect the actual current state of the repo, not a fixture baked into
the repo at build time. Results are cached to disk with a fetch timestamp so
repeated runs (and the eval suite) don't hammer the API or require network
access to demo the rest of the pipeline offline.

Design trade-off documented in the README: we use the unauthenticated REST
API (60 req/hr) by default so anyone can clone this repo and run it with zero
setup; supplying GITHUB_TOKEN raises that to 5000 req/hr.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from sentinelrag.config import settings
from sentinelrag.schema import Document, now_iso

API_ROOT = "https://api.github.com"


class GitHubRateLimitError(RuntimeError):
    """Raised when GitHub's API rate limit is hit (used to test failure recovery)."""


class GitHubConnector:
    def __init__(self, repo: str | None = None, token: str | None = None, cache_dir: Path | None = None):
        self.repo = repo or settings.repo
        self.token = token if token is not None else settings.github_token
        self.cache_dir = cache_dir or (settings.data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{self.repo.replace('/', '__')}.json"

    # -- HTTP -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> list[dict] | dict:
        url = f"{API_ROOT}{path}"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise GitHubRateLimitError(f"GitHub rate limit hit for {url}")
        resp.raise_for_status()
        return resp.json()

    # -- Fetchers ---------------------------------------------------------
    def _fetch_issues_and_prs(self, per_page: int = 30) -> list[Document]:
        # GitHub's /issues endpoint returns both issues and PRs; PRs carry a
        # "pull_request" key.
        raw = self._get(
            f"/repos/{self.repo}/issues",
            params={"state": "all", "per_page": per_page, "sort": "updated", "direction": "desc"},
        )
        docs: list[Document] = []
        for item in raw:
            is_pr = "pull_request" in item
            docs.append(
                Document(
                    id=str(item["number"]),
                    source_type="pull_request" if is_pr else "issue",
                    title=item.get("title", ""),
                    body=(item.get("body") or "").strip(),
                    url=item.get("html_url", ""),
                    updated_at=item.get("updated_at", ""),
                    labels=[l["name"] for l in item.get("labels", []) if isinstance(l, dict)],
                    state=item.get("state"),
                    author=(item.get("user") or {}).get("login"),
                )
            )
        return docs

    def _fetch_commits(self, per_page: int = 30) -> list[Document]:
        raw = self._get(f"/repos/{self.repo}/commits", params={"per_page": per_page})
        docs = []
        for item in raw:
            sha = item["sha"][:7]
            commit = item.get("commit", {})
            docs.append(
                Document(
                    id=sha,
                    source_type="commit",
                    title=(commit.get("message", "").splitlines() or [""])[0],
                    body=commit.get("message", ""),
                    url=item.get("html_url", ""),
                    updated_at=(commit.get("author") or {}).get("date", ""),
                    author=(item.get("author") or {}).get("login") or (commit.get("author") or {}).get("name"),
                )
            )
        return docs

    # -- Public API ---------------------------------------------------------
    def refresh(self) -> list[Document]:
        """Hit the live API and overwrite the local cache. Raises on failure."""
        docs = self._fetch_issues_and_prs() + self._fetch_commits()
        payload = {
            "repo": self.repo,
            "fetched_at": now_iso(),
            "documents": [doc.__dict__ for doc in docs],
        }
        self.cache_file.write_text(json.dumps(payload, indent=2))
        return docs

    def load(self, max_age_seconds: int | None = None) -> list[Document]:
        """Load from cache, refreshing first if the cache is stale/missing.

        If a live refresh fails (offline, rate-limited) but a cache exists,
        falls back to the cache rather than crashing the whole pipeline --
        this is the "recover from tool failure" behavior the eval suite
        exercises.
        """
        cached = self._read_cache()
        if cached is not None and max_age_seconds is not None:
            age = time.time() - cached["fetched_at_epoch"]
            if age < max_age_seconds:
                return cached["documents"]

        try:
            return self.refresh()
        except (requests.RequestException, GitHubRateLimitError):
            if cached is not None:
                return cached["documents"]
            raise

    def _read_cache(self) -> dict[str, Any] | None:
        if not self.cache_file.exists():
            return None
        payload = json.loads(self.cache_file.read_text())
        from datetime import datetime

        fetched_at_epoch = datetime.fromisoformat(payload["fetched_at"]).timestamp()
        documents = [Document(**d) for d in payload["documents"]]
        return {"documents": documents, "fetched_at_epoch": fetched_at_epoch}
