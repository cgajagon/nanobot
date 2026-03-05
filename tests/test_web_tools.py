import pytest

from nanobot.agent.tools.web import WebSearchTool


@pytest.mark.asyncio
async def test_web_search_uses_configured_api_key(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "web": {
                    "results": [
                        {
                            "title": "Result",
                            "url": "https://example.com",
                            "description": "Snippet",
                        }
                    ]
                }
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, _url: str, *, params: dict, headers: dict, timeout: float):
            seen["params"] = params
            seen["headers"] = headers
            seen["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", lambda: FakeClient())

    tool = WebSearchTool(api_key="test-token")
    output = await tool.execute("nanobot", count=1)

    assert output.success
    assert "Results for: nanobot" in output.output
    assert seen["params"] == {"q": "nanobot", "count": 1}
    assert seen["headers"] == {
        "Accept": "application/json",
        "X-Subscription-Token": "test-token",
    }
