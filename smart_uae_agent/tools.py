import json
import os
import math
import datetime as dt
import requests
import numexpr as ne
from typing import Optional, Dict, Any, List

from langchain_core.tools import tool
from langchain_community.vectorstores import chroma, Chroma
from langchain_community.tools import DuckDuckGoSearchRun

class PreferencesStore:
    def __init__(self):
        self._store: Dict[str, Any] = {
            "budget_level": "mid",
            "interests": [],
            "food": None
        }

    def set(self, key: str, value: Any):
        self._store[key] = value

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def update(self, d: Dict[str, Any]):
        self._store.update(d)

    def dump(self):
        return dict(self._store)

def make_tools(retriever, prefs: PreferencesStore):
    ddg = DuckDuckGoSearchRun()

    @tool("search_knowledge", return_direct=False)
    def search_knowledge(query: str, k: int = 5) -> str:
        """Search local uae knowledge base for facts, attractions, tips. args: query: str, k:int"""
        docs = retriever.get_relevant_documants(query)[:k]
        results = []
        for d in docs:
            results.append({
                "snippet": d.page_content,
                "meta": d.metadata,
            })
        return json.dumps({"results": results}, ensure_ascii=False)

    @tool("web_search", return_direct=False)
    def web_search(query: str, max_results: int = 5) -> str:
        """Lightweight search via duckduckgo search. return list of results strings"""
        try:
            res = ddg.run(query)
            if isinstance(res, list):
                res = res[:max_results]
            return json.dumps({"results": res})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool("prayer_times", return_direct=False)
    def prayer_times(city: str, date: Optional[str] = None) -> str:
        """Get real-time prayer times via aladhan for uae city. date=YYYY-MM-DD (Optional)"""
        try:
            if date is None:
                date = dt.date.today().isoformat()
                url = "https://api.aladhan.com/v1/timingsByCity?date={}"
                params = {"city": city, "country":"UAE", "method": 2, "date": date}
                r = requests.get(url.format(city), params=params, timeout=15)
                data = r.json()
                if data.get("code") == 200:
                    return json.dumps({
                        "city": city,
                        "date": date,
                        "results": data["date"]["timings"]
                    })
                else:
                    return json.dumps({"error": data.get("data")})
        except Exception as e:
            return json.dumps({"error": str(e)})

    CITY_COSTS = {
        "Dubai" : {"budget": 100, "mid": 150, "luxe": 250},
        "Abu Dhabi" : {"budget": 50, "mid": 120, "luxe": 260},
        "Sharjah" : {"budget": 50, "mid": 120, "luxe": 260},
        "General" : {"budget": 50, "mid": 120, "luxe": 260},
    }
    TRANSPORT_DAY = {
        "budget": 12, "mid": 25, "luxe": 60
    }
    MEALS_DAY = {"budget": 18, "mid": 35, "luxe": 80}
    ATTRACTIONS = {"budget": 10, "mid": 25, "luxe": 50}

    @tool("estimate_budget", return_direct=False)
    def estimate_budget(city: str, days: int, travellers: int =1, budget_level: Optional[str] = None) -> str:
        """Estimate a sample trip budget (USD). arges : city: str, days: int, travellers: int =1, budget_level: Optional[str] = None"""
        lvl = (budget_level or prefs.get("budget_level", "mid")).lower()
        if lvl not in ("budget", "mid", "luxe"):
            lvl = "mid"
        base = CITY_COSTS.get(city, CITY_COSTS["General"])
        hotel = base[lvl] * days * travellers
        transport = TRANSPORT_DAY[lvl] * days * travellers
        meals = MEALS_DAY[lvl] * days * travellers
        attractions = ATTRACTIONS[lvl] * days * travellers
        subtotal = hotel + transport + meals + attractions
        total = math.ceil(subtotal * 1.10)
        breakdown = {
            "assumptions_note": "Rough averages; adjust to season and availability.",
            "inputs": {"city": city, "days": days, "travellers": travellers, "budget_level": lvl},
            "usd_breakdown": {
                "accommodation": round(hotel, 2),
                "transport": round(transport, 2),
                "meals": round(meals, 2),
                "attractions": round(attractions, 2),
                "buffer_10pct": round(total - subtotal, 2),
            },
            "total_usd": total
        }
        return json.dumps(breakdown)

    @tool("set_preferences", return_direct=False)
    def set_preferences(preferences_json: str) -> str:
        """update traveller preferences"""
        try:
            d = json.loads(preferences_json)
            prefs.update(d)
            return json.dumps({"ok": True, "prefs": prefs.dump()})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool("get_preferences", return_direct=False)
    def get_preferences(_: str = "") -> str:
        """return current traveller preferences"""
        return json.dumps(prefs.dump())

    @tool("calculator", return_direct=False)
    def calculator(expression: str) -> str:
        try:
            val = float(ne.evaluate(expression))
            return json.dumps({"result": val})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [search_knowledge, web_search, prayer_times, estimate_budget, set_preferences, get_preferences]

def make_retriever(embedding_fn):
    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    vs = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embedding_fn,
        collection_name="uae_kb"
    )
    return vs.as_retriever(search_kwargs={"k":5})
