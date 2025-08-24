# tools.py
"""
Custom tools for the Smart UAE Agent.

Includes:
- search_knowledge: query `uae_knowledge.json` for facts/attractions/cultural tips
- prayer_times: get daily prayer times via Aladhan API (with offline fallback)
- estimate_budget: simple per-day AED trip cost
- set_preferences / get_preferences: demo user-preferences store
- web_search: optional DuckDuckGo search (if package available)

All tools return JSON strings so the agent can parse easily.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import requests

from langchain_core.tools import tool

# Try to import DuckDuckGo search tool if installed
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _HAS_DDG = True
except Exception:
    _HAS_DDG = False

# ---------------------- Preferences ----------------------
class PreferencesStore:
    """A tiny in-memory store to demonstrate memoryful tools."""
    def __init__(self):
        self._store: Dict[str, Any] = {
            "budget_level": "standard",
            "home_city": None,
            "language": "en",
        }

    def set_many(self, **kwargs) -> Dict[str, Any]:
        for k, v in kwargs.items():
            self._store[k] = v
        return self._store

    def get_all(self) -> Dict[str, Any]:
        return dict(self._store)

PREFS = PreferencesStore()

# ---------------------- Knowledge Loader ----------------------
def _load_knowledge(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _search_json_kb(kb: Dict[str, Any], query: str) -> Dict[str, Any]:
    """A simple, deterministic search over the JSON knowledge base."""
    q = (query or "").lower().strip()
    cities: Dict[str, Any] = kb.get("cities", {})
    activities: Dict[str, List[str]] = kb.get("activities", {})
    general: Dict[str, Any] = kb.get("general_info", {})

    def detect_city() -> Optional[str]:
        for c in cities.keys():
            if c.lower() in q:
                return c
        if "ras al khaimah" in q or "rak" in q:
            return "Ras Al Khaimah"
        return None

    city = detect_city()

    # Cultural tips
    if "cultural tip" in q or "culture tip" in q or "dress" in q or "ramadan" in q:
        tips = []
        if city and city in cities and "cultural_tips" in cities[city]:
            tips = cities[city]["cultural_tips"]
        else:
            # combine all known tips (dedup)
            for c, v in cities.items():
                tips.extend(v.get("cultural_tips", []))
            tips = list(dict.fromkeys(tips))
        return {"city": city or "general", "cultural_tips": tips}

    # Top attractions / things to do
    if any(k in q for k in ["attraction", "thing to do", "what can i do", "places to visit", "itinerary"]):
        if city == "Ras Al Khaimah":
            rak_recs = [x for x in activities.get("relaxation", []) if "Ras Al Khaimah" in x]
            return {"city": "Ras Al Khaimah", "top_attractions": rak_recs or ["Beach resorts in Ras Al Khaimah"]}
        if city and city in cities:
            return {"city": city, "top_attractions": cities[city].get("top_attractions", [])}

    # General info
    if any(k in q for k in ["currency", "language", "best time", "transport", "metro"]):
        return {"general_info": general}

    # Categories
    for bucket in ["adventure", "culture", "relaxation"]:
        if bucket in q:
            return {bucket: activities.get(bucket, [])}

    # Default: return whole KB (small) so the agent can compose an answer safely.
    return kb

# ---------------------- Tools Factory ----------------------
def make_tools(knowledge_path: str) -> List[Any]:
    """Return a list of LangChain tools bound to the given knowledge file."""
    kb = _load_knowledge(knowledge_path)

    @tool("search_knowledge", return_direct=False)
    def search_knowledge(query: str) -> str:
        """Search local UAE knowledge base for facts, attractions, cultural tips. Args: query:str"""
        result = _search_json_kb(kb, query)
        return json.dumps(result, ensure_ascii=False)

    @tool("web_search", return_direct=False)
    def web_search(query: str, max_results: int = 5) -> str:
        """Lightweight web search via DuckDuckGo. Use only if a fact is missing from the knowledge base."""
        if not _HAS_DDG:
            return json.dumps({"error": "DuckDuckGoSearchRun not installed"}, ensure_ascii=False)
        ddg = DuckDuckGoSearchRun()
        try:
            hits = ddg.run(query)
            return json.dumps({"results": hits}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @tool("prayer_times", return_direct=False)
    def prayer_times(city: str, date: Optional[str] = None) -> str:
        """Get prayer times for a UAE city on a date (YYYY-MM-DD). Returns Fajr, Dhuhr, Asr, Maghrib, Isha."""
        city_norm = (city or "").strip().title() or "Dubai"
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        try:
            yyyy, mm, dd = date.split("-")
        except Exception:
            dt = datetime.now()
            yyyy, mm, dd = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
        # API
        try:
            url = "https://api.aladhan.com/v1/timingsByCity"
            resp = requests.get(url, params={
                "city": city_norm,
                "country": "United Arab Emirates",
                "method": 2,
                "date": f"{dd}-{mm}-{yyyy}"
            }, timeout=4)
            timings = resp.json()["data"]["timings"]
            out = {k: timings[k] for k in ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"] if k in timings}
            if out:
                return json.dumps({"city": city_norm, "date": f"{yyyy}-{mm}-{dd}", "prayer_times": out}, ensure_ascii=False)
        except Exception:
            pass
        # Fallback
        FALLBACK = {
            "Dubai": {"Fajr": "04:25", "Dhuhr": "12:25", "Asr": "15:55", "Maghrib": "18:49", "Isha": "20:19"},
            "Abu Dhabi": {"Fajr": "04:30", "Dhuhr": "12:30", "Asr": "16:00", "Maghrib": "18:53", "Isha": "20:23"},
            "Sharjah": {"Fajr": "04:26", "Dhuhr": "12:25", "Asr": "15:56", "Maghrib": "18:49", "Isha": "20:19"},
        }
        return json.dumps({"city": city_norm, "date": f"{yyyy}-{mm}-{dd}", "prayer_times": FALLBACK.get(city_norm, FALLBACK["Dubai"]), "note":"Static fallback"}, ensure_ascii=False)

    @tool("estimate_budget", return_direct=False)
    def estimate_budget(city: str, days: int, style: str = "standard") -> str:
        """Estimate trip budget for a city. Args: city:str, days:int (>0), style:str in budget|standard|luxury"""
        style = (style or "standard").lower().strip()
        days = int(days)
        if style not in {"budget", "standard", "luxury"}:
            return json.dumps({"error": "style must be budget|standard|luxury"}, ensure_ascii=False)
        if days <= 0:
            return json.dumps({"error": "days must be > 0"}, ensure_ascii=False)
        rates = {"budget": 150, "standard": 400, "luxury": 1000}
        total = rates[style] * days
        return json.dumps({
            "city": city, "days": days, "style": style,
            "rate_per_day_aed": rates[style], "estimated_total_aed": total
        }, ensure_ascii=False)

    @tool("set_preferences", return_direct=False)
    def set_preferences(budget_level: Optional[str] = None, home_city: Optional[str] = None, language: Optional[str] = None) -> str:
        """Store user preferences like budget_level (budget/standard/luxury), home_city, language code."""
        updates = {}
        if budget_level: updates["budget_level"] = budget_level
        if home_city: updates["home_city"] = home_city
        if language: updates["language"] = language
        PREFS.set_many(**updates)
        return json.dumps({"preferences": PREFS.get_all()}, ensure_ascii=False)

    @tool("get_preferences", return_direct=False)
    def get_preferences() -> str:
        """Return stored user preferences."""
        return json.dumps({"preferences": PREFS.get_all()}, ensure_ascii=False)

    return [search_knowledge, prayer_times, estimate_budget, set_preferences, get_preferences, web_search]
