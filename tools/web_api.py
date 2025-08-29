from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import json

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class WebSearchError(Exception):
    pass


class WebSearchClient:
    """Production-leaning web search client with provider abstraction.

    Supported providers: serpapi, tavily. Uses requests with timeouts and retries.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        endpoint: Optional[str] = None,
        timeout_s: float = 8.0,
        retries: int = 2,
        backoff: float = 0.5,
        cache_ttl_s: Optional[float] = None,
        cache_path: Optional[str] = None,
    ):
        if requests is None:
            raise ImportError("'requests' package is required for web search.")
        self.provider = provider.lower()
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff = backoff
        self.cache_ttl_s = cache_ttl_s
        self.cache_path = cache_path
        self._cache = None
        if cache_path:
            try:
                import json, os
                if os.path.exists(cache_path):
                    self._cache = json.loads(open(cache_path, "r", encoding="utf-8").read())
                else:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    self._cache = {}
            except Exception:
                self._cache = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Cache lookup
        cache_key = f"{self.provider}:{query}:{top_k}"
        if self._cache is not None and cache_key in self._cache:
            entry = self._cache[cache_key]
            ts = entry.get("ts", 0)
            if self.cache_ttl_s and (time.time() - ts) < self.cache_ttl_s:
                return entry.get("results", [])
        if self.provider == "serpapi":
            results = self._search_serpapi(query, top_k)
        if self.provider == "tavily":
            results = self._search_tavily(query, top_k)
        else:
            raise WebSearchError(f"Unsupported provider: {self.provider}")
        if self._cache is not None:
            self._cache[cache_key] = {"ts": time.time(), "results": results}
            try:
                import json
                open(self.cache_path, "w", encoding="utf-8").write(json.dumps(self._cache))
            except Exception:
                pass
        return results

    def _request(self, method: str, url: str, params: Dict[str, Any] | None = None, data: Dict[str, Any] | None = None) -> Any:
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                if method.lower() == "get":
                    r = requests.get(url, params=params, timeout=self.timeout_s)
                else:
                    headers = {"Content-Type": "application/json"}
                    r = requests.post(url, data=json.dumps(data or {}), headers=headers, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.json()
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise WebSearchError(f"HTTP {r.status_code}: {r.text}")
                last_err = WebSearchError(f"HTTP {r.status_code}: {r.text}")
            except Exception as e:  # pragma: no cover - network specific
                last_err = e
            time.sleep(self.backoff * (2 ** attempt))
        assert last_err is not None
        raise WebSearchError(str(last_err))

    def _search_serpapi(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        url = self.endpoint or "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "num": min(top_k, 10),
            "api_key": self.api_key,
        }
        data = self._request("get", url, params=params)
        results = []
        for item in (data.get("organic_results") or [])[:top_k]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "source": "serpapi",
            })
        return results

    def _search_tavily(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        url = self.endpoint or "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
        }
        data = self._request("post", url, data=payload)
        results = []
        for item in (data.get("results") or [])[:top_k]:
            results.append({
                "title": item.get("title"),
                "link": item.get("url"),
                "snippet": item.get("content"),
                "source": "tavily",
            })
        return results


def build_client_from_config(cfg: dict) -> Optional[WebSearchClient]:
    web = cfg.get("web_config") if cfg else None
    if not web:
        return None
    provider = str(web.get("provider", "serpapi"))
    api_key_env = str(web.get("api_key_env", "SERPAPI_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        return None
    endpoint = web.get("endpoint")
    timeout_s = float(web.get("timeout_s", 8))
    cache_ttl_s = float(web.get("cache_ttl_s", 0)) or None
    cache_path = web.get("cache_path")
    return WebSearchClient(provider=provider, api_key=api_key, endpoint=endpoint, timeout_s=timeout_s, cache_ttl_s=cache_ttl_s, cache_path=cache_path)
