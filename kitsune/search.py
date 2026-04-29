"""
Kitsune Web Search — Search the internet for real-time information.
Supports DuckDuckGo (free, no API key) and Google Custom Search.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("kitsune.search")

# Windows SSL fix: create unverified context for scraping
_SSL_UNVERIFIED = ssl._create_unverified_context()


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "unknown"


class WebSearch:
    """Web search interface with multiple provider support."""

    def __init__(
        self,
        ollama_api_key: str = "",
        ollama_api_base: str = "https://ollama.com",
        google_api_key: str = "",
        google_cx: str = "",
        max_results: int = 5,
        timeout: int = 15,
    ):
        self.ollama_api_key = ollama_api_key
        self.ollama_api_base = ollama_api_base.rstrip("/")
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        self.max_results = max(max_results, 1)
        self.timeout = timeout

    def search(self, query: str) -> list[SearchResult]:
        """Search the web and return results."""
        if not query or not query.strip():
            return []

        query = query.strip()
        logger.info("🔍 Searching web: %s", query[:100])

        # 1. Try Ollama Web Search API first (most reliable, uses Ollama API key)
        try:
            results = self._search_ollama_web_search(query)
            if results:
                logger.info("✅ Ollama web search returned %d results", len(results))
                return results[: self.max_results]
        except Exception as e:
            logger.warning("Ollama web search failed: %s", e)

        # 2. Fallback: DuckDuckGo HTML scraping
        try:
            results = self._search_duckduckgo(query)
            if results:
                logger.info("✅ DuckDuckGo returned %d results", len(results))
                return results[: self.max_results]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)

        # 3. Fallback: DuckDuckGo Lite
        try:
            results = self._search_duckduckgo_lite(query)
            if results:
                logger.info("✅ DuckDuckGo Lite returned %d results", len(results))
                return results[: self.max_results]
        except Exception as e:
            logger.warning("DuckDuckGo Lite search failed: %s", e)

        # 4. Fallback: Google Custom Search API (if configured)
        if self.google_api_key and self.google_cx:
            try:
                results = self._search_google(query)
                if results:
                    logger.info("✅ Google returned %d results", len(results))
                    return results[: self.max_results]
            except Exception as e:
                logger.warning("Google search failed: %s", e)

        return []

    def _search_ollama_web_search(self, query: str) -> list[SearchResult]:
        """Search using Ollama Web Search API (requires OLLAMA_API_KEY)."""
        if not self.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY is required for Ollama web search")

        url = f"{self.ollama_api_base}/api/web_search"
        body = json.dumps({
            "query": query,
            "max_results": min(self.max_results, 10),
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.ollama_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Kitsune-Agent/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_UNVERIFIED) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="ollama",
                )
            )

        return results

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        """Search using DuckDuckGo HTML (no API key required)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_UNVERIFIED) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_duckduckgo_html(html)

    def _search_duckduckgo_lite(self, query: str) -> list[SearchResult]:
        """Fallback: DuckDuckGo Lite endpoint (simpler HTML)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://lite.duckduckgo.com/",
            },
        )

        with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_UNVERIFIED) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_duckduckgo_lite_html(html)

    @staticmethod
    def _parse_duckduckgo_lite_html(html: str) -> list[SearchResult]:
        """Parse DuckDuckGo Lite HTML results."""
        results = []

        # DDG Lite format: <a class="result-link" href="...">title</a>
        #                   <td class="result-snippet">snippet</td>
        rows = re.findall(
            r'<tr[^>]*>.*?<a[^>]*class="result-link"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?(?:<td[^>]*class="result-snippet"[^>]*>(.*?)</td>)?.*?</tr>',
            html,
            re.DOTALL | re.IGNORECASE,
        )

        for row in rows:
            raw_url = row[0]
            title = _strip_html(row[1])
            snippet = _strip_html(row[2]) if len(row) > 2 and row[2] else ""

            # Clean DDG redirect URLs
            if raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            elif "/uddg=" in raw_url:
                redirect_match = re.search(r"uddg=([^&]+)", raw_url)
                if redirect_match:
                    raw_url = urllib.parse.unquote(redirect_match.group(1))

            if title and raw_url:
                results.append(
                    SearchResult(
                        title=title,
                        url=raw_url,
                        snippet=snippet,
                        source="duckduckgo",
                    )
                )

        return results

    @staticmethod
    def _parse_duckduckgo_html(html: str) -> list[SearchResult]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # DuckDuckGo lite uses class="result" for each result
        # Each result has: <a class="result__a">title</a>
        #                  <a class="result__url">url</a>
        #                  <a class="result__snippet">snippet</a>

        result_blocks = re.findall(
            r'<div class="result[^"]*"[^>]*>.*?(?=<div class="result|<div class="no-results|</body>)',
            html,
            re.DOTALL | re.IGNORECASE,
        )

        for block in result_blocks:
            # Extract title and URL
            title_match = re.search(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if not title_match:
                # Try alternative format
                title_match = re.search(
                    r'<a[^>]*href="([^"]*)"[^>]*class="result__a"[^>]*>(.*?)</a>',
                    block,
                    re.DOTALL | re.IGNORECASE,
                )

            if title_match:
                raw_url = title_match.group(1)
                title = _strip_html(title_match.group(2))

                # DuckDuckGo redirects through their domain
                if raw_url.startswith("//"):
                    raw_url = "https:" + raw_url
                elif raw_url.startswith("/"):
                    # Extract actual URL from DDG redirect
                    redirect_match = re.search(r"uddg=([^&]+)", raw_url)
                    if redirect_match:
                        raw_url = urllib.parse.unquote(redirect_match.group(1))

                # Extract snippet
                snippet_match = re.search(
                    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                if not snippet_match:
                    snippet_match = re.search(
                        r'<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
                        block,
                        re.DOTALL | re.IGNORECASE,
                    )

                snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""

                if title and raw_url:
                    results.append(
                        SearchResult(
                            title=title,
                            url=raw_url,
                            snippet=snippet,
                            source="duckduckgo",
                        )
                    )

        return results

    def _search_google(self, query: str) -> list[SearchResult]:
        """Search using Google Custom Search API."""
        encoded = urllib.parse.quote_plus(query)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={self.google_api_key}"
            f"&cx={self.google_cx}"
            f"&q={encoded}"
            f"&num={self.max_results}"
        )

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_UNVERIFIED) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        for item in data.get("items", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                )
            )

        return results

    @staticmethod
    def format_for_prompt(results: list[SearchResult]) -> str:
        """Format search results for injection into LLM prompt."""
        if not results:
            return ""

        lines = ["[Web search results]:"]
        for idx, result in enumerate(results, 1):
            lines.append(f"{idx}. {result.title}")
            lines.append(f"   URL: {result.url}")
            if result.snippet:
                lines.append(f"   Snippet: {result.snippet}")
            lines.append("")

        return "\n".join(lines)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common entities
    replacements = {
        "&quot;": '"',
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&#39;": "'",
        "&nbsp;": " ",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    # Collapse whitespace
    return " ".join(text.split())


def needs_web_search(text: str) -> bool:
    """Heuristic to detect if a query likely needs web search."""
    text_lower = text.lower()

    # Explicit search triggers
    search_prefixes = (
        "cari ",
        "cariin ",
        "search ",
        "google ",
        "lookup ",
        "find ",
        "info ",
        "informasi ",
        "berita ",
        "news ",
        "latest ",
        "terbaru ",
        "update ",
    )
    if any(text_lower.startswith(prefix) for prefix in search_prefixes):
        return True

    # Question patterns that suggest real-time info need
    real_time_indicators = (
        r"\b(berita|news|update|terbaru|latest|current|hari ini|today|sekarang|now)\b",
        r"\b(cuaca|weather|saham|stock|harga bitcoin|btc price|harga emas)\b",
        r"\b(jadwal|schedule|pertandingan|match|skor|score)\b",
        r"\b(berapa|how much|what is the price of)\b.*\b(harga|price)\b",
        r"\b(kapan|when)\b.*\b(rilis|release|launch|rilis)\b",
    )
    for pattern in real_time_indicators:
        if re.search(pattern, text_lower):
            return True

    return False


class SearchTool:
    """Lightweight search tool for /search command and explicit use."""

    def __init__(self, web_search: WebSearch | None = None):
        self.web_search = web_search

    def run(self, query: str) -> str:
        if not self.web_search:
            return "Web search belum dikonfigurasi."
        results = self.web_search.search(query)
        if not results:
            return "Tidak ada hasil pencarian."
        return WebSearch.format_for_prompt(results)
