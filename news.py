# news.py
from __future__ import annotations
import re, time
import requests, feedparser

USER_AGENT = "Mozilla/5.0 (autostock-bot)"

def fetch_news_headlines(query: str, max_items: int = 3, lang="ko") -> list[dict]:
    """
    Google News RSS로 최근 헤드라인 n개 반환.
    return: [{'title':..., 'link':..., 'published':...}, ...]
    """
    try:
        q = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl={lang}"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = []
        for e in feed.entries[:max_items]:
            items.append({
                "title": _clean(e.get("title")),
                "link": e.get("link"),
                "published": e.get("published", ""),
            })
        return items
    except Exception:
        return []

def _clean(s: str | None) -> str:
    if not s: return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s
