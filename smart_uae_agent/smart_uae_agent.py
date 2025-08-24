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
from typing import List
from langchain.agents import initialize_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

SYSTEM = """ You are SmartUAEAgent, a careful UAE tourism assistant (Dubai, Abu Dhabi, Sharjah, etc... ).
Be concise, friendly and proactive. think step-by-step with tools
Always:-
- prefer local knowledge via search knowledge first for attractions/tips/facts.
- Uae set_preference/get_preference to remember budget level , intersets and food etc...
- For itineraries, tailors to city, days, season (if mentioned), and preferences.
- For Prayer times, call prayer_times. for quick math, use calculator.
- For rough budget, call estimate_budget and explain assumption briefly.
- If you browse, city that results may vary; summarize, don't dump links.
- Keep context over turns, don't lose the traveler's choices.
- If something is unknown or unavailable, say so and offer a nearby alternative.
"""

def build_agent(llm, tools: List, memory=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)
    return executor
