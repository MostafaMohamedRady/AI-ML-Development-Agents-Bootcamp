# smart_uae_agent.py
"""
Smart UAE Agent (LangChain)

- Agent type: chat-zero-shot-react-description
- Tools: knowledge search (JSON), prayer times, trip budget, preferences, optional web search
- Memory: ConversationBufferMemory
- LLM: Gemini (or OpenAI) via llm_provider.load_llm()

Run:
  export GOOGLE_API_KEY=...
  python smart_uae_agent.py --knowledge ./uae_knowledge.json --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

from llm_provider import load_llm
from tools import make_tools

SYSTEM_RULES = """
You are SmartUAE, a careful UAE tourism assistant.

RULES:
1) Do NOT hallucinate: Base factual answers ONLY on the UAE knowledge tool results.
2) For itineraries or "things to do" questions, you MAY answer directly without tool calls,
   but ONLY propose places found in the knowledge base (search_knowledge). If unsure, call the tool.
3) Use tools when appropriate:
   - search_knowledge for UAE facts/attractions/cultural tips
   - prayer_times for daily timings
   - estimate_budget for trip costs
   - get/set_preferences for user settings
4) Be concise, friendly, and remember prior context via chat history.
"""

def build_agent(knowledge_path: str, verbose: bool = False) -> AgentExecutor:
    llm = load_llm()
    tools = make_tools(knowledge_path)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        agent_kwargs={"system_message": SystemMessage(content=SYSTEM_RULES)},
        memory=memory,
    )
    return agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", type=str, default="./uae_knowledge.json", help="Path to uae_knowledge.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.knowledge):
        raise SystemExit(f"Knowledge file not found at {args.knowledge}")

    agent = build_agent(args.knowledge, verbose=args.verbose)
    print("✅ SmartUAE agent ready. Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        t0 = time.time()
        resp = agent.invoke({"input": user})
        elapsed = time.time() - t0
        print(f"SmartUAE: {resp['output']}")
        if elapsed > 5:
            print(f"(Note: reply took {elapsed:.1f}s)")
        print()

if __name__ == "__main__":
    main()
