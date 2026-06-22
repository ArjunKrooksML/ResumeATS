from contextlib import contextmanager
from contextvars import ContextVar

from crewai import Crew, Process, Task
from crewai.tools import tool

from app.debug_log import log_step

# tracks which roles are already mid-consultation in the current call stack,
# so an agent can't be asked to consult someone who is already waiting on it
_chain: ContextVar[tuple] = ContextVar("consult_chain", default=())

_registered_tools = []


def reset_consult_tools() -> None:
    for consult_tool in _registered_tools:
        consult_tool.reset_usage_count()


@contextmanager
def track_active_agents(*roles: str):
    """Seed the consultation chain with the agent(s) about to run directly.

    Without this, an agent invoked directly (not via a consult tool) is
    invisible to the cycle check, so a consulted agent could call back into
    it while its own execution is still on the call stack — a real
    re-entrant call CrewAI itself rejects with "Executor is already running".
    """
    chain = _chain.get()
    token = _chain.set(chain + tuple(roles))
    try:
        yield
    finally:
        _chain.reset(token)


def ask_agent(agent, question: str) -> str:
    consult_task = Task(
        description=f"Answer this question briefly and directly, using only your own expertise: {question}",
        expected_output="A brief, direct answer.",
        agent=agent,
    )
    Crew(agents=[agent], tasks=[consult_task], process=Process.sequential).kickoff()
    return consult_task.output.raw


def make_consult_tool(agent):
    role_name = agent.role

    def consult(question: str) -> str:
        chain = _chain.get()
        if role_name in chain:
            return f"Cannot consult {role_name}: already part of this consultation chain, would create a loop."
        log_step(f"Consulting {role_name}: {question}")
        token = _chain.set(chain + (role_name,))
        try:
            return ask_agent(agent, question)
        finally:
            _chain.reset(token)

    consult.__doc__ = f"Ask the {role_name} a clarifying question when their judgment would help, and get their direct answer."
    consult_tool = tool(f"Consult {role_name}", max_usage_count=1)(consult)
    _registered_tools.append(consult_tool)
    return consult_tool
