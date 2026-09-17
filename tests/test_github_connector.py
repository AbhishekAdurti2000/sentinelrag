"""Tests the GitHub connector's caching / fallback behavior without ever
hitting the real network -- `requests.get` is monkeypatched.
"""
from pathlib import Path

import pytest
import requests

from sentinelrag.connectors.github import GitHubConnector, GitHubRateLimitError


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


ISSUES_PAYLOAD = [
    {
        "number": 1,
        "title": "Bug A",
        "body": "broke",
        "html_url": "https://x/1",
        "updated_at": "2026-01-01T00:00:00Z",
        "labels": [{"name": "bug"}],
        "state": "open",
        "user": {"login": "alice"},
    },
    {
        "number": 2,
        "title": "PR A",
        "body": "fixes it",
        "html_url": "https://x/2",
        "updated_at": "2026-01-02T00:00:00Z",
        "labels": [],
        "state": "open",
        "user": {"login": "bob"},
        "pull_request": {},
    },
]

COMMITS_PAYLOAD = [
    {
        "sha": "abcdef1234567",
        "commit": {"message": "Fix bug A\n\nCloses #1.", "author": {"date": "2026-01-03T00:00:00Z"}},
        "html_url": "https://x/commit/abcdef1",
        "author": {"login": "carol"},
    }
]


@pytest.fixture()
def connector(tmp_path: Path) -> GitHubConnector:
    return GitHubConnector(repo="acme/repo", token=None, cache_dir=tmp_path)


def test_refresh_parses_issues_prs_and_commits(monkeypatch, connector: GitHubConnector):
    def fake_get(self, path, params=None):
        if "issues" in path:
            return ISSUES_PAYLOAD
        return COMMITS_PAYLOAD

    monkeypatch.setattr(GitHubConnector, "_get", fake_get)
    docs = connector.refresh()
    by_type = {d.source_type for d in docs}
    assert by_type == {"issue", "pull_request", "commit"}
    issue = next(d for d in docs if d.source_type == "issue")
    assert issue.id == "1"
    assert issue.labels == ["bug"]
    pr = next(d for d in docs if d.source_type == "pull_request")
    assert pr.id == "2"


def test_load_falls_back_to_cache_on_rate_limit(monkeypatch, connector: GitHubConnector):
    def fake_get_ok(self, path, params=None):
        return ISSUES_PAYLOAD if "issues" in path else COMMITS_PAYLOAD

    monkeypatch.setattr(GitHubConnector, "_get", fake_get_ok)
    connector.refresh()  # populate cache

    def fake_get_fail(self, path, params=None):
        raise GitHubRateLimitError("rate limited")

    monkeypatch.setattr(GitHubConnector, "_get", fake_get_fail)
    docs = connector.load()  # no cached max_age -> tries refresh, fails, falls back
    assert len(docs) == 3


def test_load_raises_if_no_cache_and_refresh_fails(monkeypatch, connector: GitHubConnector):
    def fake_get_fail(self, path, params=None):
        raise GitHubRateLimitError("rate limited")

    monkeypatch.setattr(GitHubConnector, "_get", fake_get_fail)
    with pytest.raises(GitHubRateLimitError):
        connector.load()
