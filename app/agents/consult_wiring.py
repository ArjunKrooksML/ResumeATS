from app.agents.advisor_agent import advisor_agent
from app.agents.critic_agent import critic_agent
from app.agents.gap_analyst_agent import gap_analyst_agent
from app.agents.parser_agent import parser_agent
from app.agents.scorer_agent import scorer_agent
from app.agents.strategist_agent import strategist_agent
from app.tools.consult import make_consult_tool

# Only give agents consultation tools they can actually use mid-task.
# Parser, Scorer, Gap Analyst and Strategist have everything they need in their
# task inputs — giving them cross-agent tools just adds unnecessary LLM calls.
# The Advisor is the only agent that benefits: it may want to clarify a gap
# or pre-validate a rewrite before the Critic sees it.
CONSULT_MAP = {
    advisor_agent: [gap_analyst_agent, critic_agent],
}

_wired = False


def wire_consult_tools() -> None:
    global _wired
    if _wired:
        return
    for agent, peers in CONSULT_MAP.items():
        agent.tools = list(agent.tools) + [make_consult_tool(peer) for peer in peers]
    _wired = True
