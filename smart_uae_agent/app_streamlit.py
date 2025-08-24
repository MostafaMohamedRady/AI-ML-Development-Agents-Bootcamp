#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for the UAE Tourism Agent (course project)
Uses your existing module layout exactly:
  - llm_provider: load_llm, load_embeddings
  - tools: make_tools, PreferencesStore, (optional) make_retriever
  - smart_uae_agent: build_agent

Run:
  streamlit run app_streamlit.py
"""

import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# --- Import from your codebase (exact module layout) ---
from llm_provider import load_llm, load_embeddings  # noqa: F401 (embeddings optional)
from tools import make_tools, PreferencesStore  # noqa: F401
from smart_uae_agent import build_agent  # noqa: F401


# -------------------------- Helpers --------------------------

def _safe_agent_invoke(agent: Any, text: str) -> str:
    """
    Calls a LangChain-style agent regardless of whether it expects .invoke({...})
    or .run(text). Also handles callables for maximum flexibility.
    """
    # AgentExecutor from LangChain
    try:
        return agent.invoke({"input": text})["output"]
    except Exception:
        pass

    # Older agents or chains
    try:
        return agent.run(text)
    except Exception:
        pass

    # Callable fallback
    if callable(agent):
        return agent(text)

    # If none of the above work, raise a clean error
    raise RuntimeError("Agent does not support .invoke({'input': ...}), .run(...), or callable(text).")


def _ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {"role": "user"|"assistant", "content": str}
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = PreferencesStore()


def _build_tools_safely(knowledge_path: str, prefs: PreferencesStore) -> List[Any]:
    """
    Build tools with best-effort signatures.
    Required by the assignment: knowledge search, prayer times, budget planner (and optional search).
    """
    # Try common signatures for make_tools in user repos.
    # 1) (knowledge_path, prefs)
    try:
        return make_tools(knowledge_path=knowledge_path, prefs=prefs)
    except TypeError:
        pass
    try:
        return make_tools(knowledge_path, prefs)
    except TypeError:
        pass

    # 2) (knowledge_path) only
    try:
        return make_tools(knowledge_path=knowledge_path)
    except TypeError:
        pass
    try:
        return make_tools(knowledge_path)
    except TypeError:
        pass

    # 3) No args
    return make_tools()


def _build_agent_safely(llm: Any, tools: List[Any], system_message: Optional[str] = None, verbose: bool = False) -> Any:
    """
    Build the agent using the user's build_agent() with flexible signatures.
    """
    # Most likely signature
    try:
        return build_agent(llm=llm, tools=tools, system_message=system_message, verbose=verbose)
    except TypeError:
        pass

    # Some repos use (llm, tools, **kwargs)
    try:
        return build_agent(llm, tools, system_message=system_message, verbose=verbose)
    except TypeError:
        pass

    # Minimal
    try:
        return build_agent(llm, tools)
    except TypeError:
        pass

    # Fallback no llm (if build_agent wires its own LLM internally)
    return build_agent(tools=tools)


# -------------------------- UI --------------------------

st.set_page_config(page_title="UAE Tourism & City Helper", page_icon="🧭", layout="wide")
load_dotenv()  # load .env

_ensure_session_state()

with st.sidebar:
    st.header("⚙️ Settings")

    # Show provider label from env (matches your screenshot intent)
    provider = os.getenv("PROVIDER", "gemini").strip()
    st.caption(f"LLM Provider: `{provider}` (change in .env)")

    # Model + temperature knobs (these do not override your build_agent unless it reads envs)
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    os.environ.setdefault("MODEL_NAME", model_name)
    os.environ["MODEL_NAME"] = model_name
    os.environ.setdefault("MODEL_TEMPERATURE", str(temperature))
    os.environ["MODEL_TEMPERATURE"] = str(temperature)

    # Knowledge path (required by the assignment)
    default_knowledge = os.getenv("KNOWLEDGE_PATH", "uae_knowledge.json")
    knowledge_path = st.text_input("Knowledge JSON", value=default_knowledge, help="Path to uae_knowledge.json")

    # Default budget preference (we store it in PreferencesStore)
    default_budget = st.selectbox("Default budget level", ["budget", "mid", "luxe"], index=1)
    st.session_state["prefs"].set("default_budget", default_budget)

    # Clear chat button
    if st.button("🗑️ Clear chat"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


st.title("🇦🇪 UAE Tourism & City Helper")
st.write("Smart, tool-using assistant for attractions, itineraries, budgets, and prayer times across the UAE.")
with st.expander("Tips", expanded=False):
    st.write("- Start with your **city** and **days** (e.g., *2 days in Dubai*).")
    st.write("- Tell me your **interests** (history, food, outdoors).")
    st.write("- Ask *prayer times in Abu Dhabi today* or *budget for Dubai, 3 days*.")

# -------------------------- Boot the agent --------------------------

if st.session_state["agent"] is None:
    try:
        llm = load_llm()  # your llm_provider.py decides based on env PROVIDER/API key
    except Exception as e:
        st.error(f"Failed to load LLM. Check your keys and provider. Details: {e}")
        st.stop()

    try:
        tools = _build_tools_safely(knowledge_path, st.session_state["prefs"])
    except Exception as e:
        st.error(f"Failed to build tools. Ensure uae_knowledge.json exists. Details: {e}")
        st.stop()

    # System message enforces no-hallucinations, tool usage, and itinerary rule
    system_message = (
        "You are a careful UAE tourism assistant.\n"
        "Rules:\n"
        "1) Base factual answers ONLY on the uae_knowledge.json or tool outputs.\n"
        "2) For itineraries, suggest places that are present in the knowledge JSON. If unsure, ask or use the knowledge tool.\n"
        "3) Use the budget tool for costs; use the prayer tool for prayer times.\n"
        "4) Be concise and friendly. Remember prior city/duration mentioned by the user."
    )

    try:
        agent = _build_agent_safely(llm=llm, tools=tools, system_message=system_message, verbose=False)
        st.session_state["agent"] = agent
    except Exception as e:
        st.error(f"Failed to build agent. Details: {e}")
        st.stop()


# -------------------------- Chat loop --------------------------

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_text = st.chat_input("Ask about visas, itineraries, prayer times, budgets, or attractions…")
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Call the agent
    try:
        reply = _safe_agent_invoke(st.session_state["agent"], user_text)
    except Exception as e:
        reply = f"Sorry, I hit an error while answering: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
