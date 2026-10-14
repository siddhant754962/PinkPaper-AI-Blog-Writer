
from langchain_core.messages import SystemMessage, HumanMessage
from ..models import get_llm
from ..schemas import RouterDecision
from ..state import State


ROUTER_SYSTEM = """
You route topics for a technical blog writer.
closed_book = evergreen concept; no web research.
hybrid = evergreen topic needing current tools/examples; research is useful.
open_book = latest/news/pricing/policy; research is required.
If research is needed, return 3–6 focused search queries. Do not invent evidence.
Return only the requested structured decision.
"""


def _fallback_decision(topic: str) -> RouterDecision:
    """Fallback for models that fail the structured tool-call response."""
    text = topic.lower()
    current_signals = (
        "latest", "today", "current", "recent", "2026", "2025", "news",
        "price", "pricing", "policy", "regulation", "update", "market",
        "election", "contract", "breaking",
    )
    hybrid_signals = (
        "best tools", "tools for", "comparison", " vs ", "versus",
        "frameworks", "libraries", "examples", "industry", "trends",
        "data analytics", "use cases",
    )

    if any(term in text for term in current_signals):
        mode, needs_research = "open_book", True
        reason = "The topic appears to require current information."
    elif any(term in text for term in hybrid_signals):
        mode, needs_research = "hybrid", True
        reason = "Research may improve current tools and practical examples."
    else:
        mode, needs_research = "closed_book", False
        reason = "The topic appears evergreen and can be covered from model knowledge."

    return RouterDecision(
        needs_research=needs_research,
        mode=mode,
        reason=reason,
        queries=[topic.strip()] if needs_research and topic.strip() else [],
        max_results_per_query=5,
    )


def router_node(state: State) -> dict:
    topic = state["topic"]
    try:
        decision = get_llm().with_structured_output(RouterDecision).invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}\nAs-of date: {state['as_of']}"),
        ])
    except Exception as exc:
        # Groq may return ordinary prose instead of the tool call required by
        # with_structured_output. Keep the graph running using a safe fallback.
        print(f"[router_node] Structured routing failed; using fallback. Reason: {exc}")
        decision = _fallback_decision(topic)

    # Respect an explicit request from the UI to enable web research.
    if state.get("needs_research"):
        decision.needs_research = True
        if decision.mode == "closed_book":
            decision.mode = "hybrid"
        if not decision.queries:
            decision.queries = [topic.strip()]

    days = 7 if decision.mode == "open_book" else 45 if decision.mode == "hybrid" else 3650
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"
