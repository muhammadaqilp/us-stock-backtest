import pytest

from swing_screener.events import recent_headlines


@pytest.fixture(autouse=True)
def _no_live_news(monkeypatch):
    """CLI tests must never hit Yahoo's news feed; the stub is a SYNTHETIC empty feed."""
    monkeypatch.setattr("swing_screener.cli.recent_headlines", lambda t: recent_headlines(t, fetch=lambda x: []))
