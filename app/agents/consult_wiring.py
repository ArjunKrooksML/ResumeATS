from app.agents.advisor_agent import advisor_agent
from app.agents.critic_agent import critic_agent
from app.agents.gap_analyst_agent import gap_analyst_agent
from app.agents.parser_agent import parser_agent
from app.agents.scorer_agent import scorer_agent
from app.agents.strategist_agent import strategist_agent
from app.tools.consult import make_consult_tool

ALL_AGENTS = [parser_agent, scorer_agent, gap_analyst_agent, advisor_agent, critic_agent, strategist_agent]

_wired = False


def wire_consult_tools() -> None:
    global _wired
    if _wired:
        return
    for agent in ALL_AGENTS:
        others = [other for other in ALL_AGENTS if other is not agent]
        agent.tools = list(agent.tools) + [make_consult_tool(other) for other in others]
    _wired = True
