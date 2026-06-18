from datetime import date

from crewai.tools import tool


@tool("Get Current Date")
def get_current_date() -> str:
    """Returns today's date in YYYY-MM-DD format, for calculating experience durations from resume dates."""
    return date.today().isoformat()
