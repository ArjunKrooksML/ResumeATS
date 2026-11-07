import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Optional

from loguru import logger

LOGS_DIR = Path("logs")
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHF]')

logger.remove()
logger.add(sys.stdout, format="[ATS] {message}", level="INFO", colorize=False)

_log_file: Optional[IO] = None


class _StripAnsi:
    # wraps a file and strips ANSI escape codes on write
    def __init__(self, f: IO):
        self._f = f

    def write(self, s: str) -> int:
        return self._f.write(_ANSI_RE.sub("", s))

    def flush(self):
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


def log_step(message: str) -> None:
    logger.info(message)


def log_verbose(content: str) -> None:
    if _log_file is not None:
        _log_file.write(_ANSI_RE.sub("", content) + "\n")
        _log_file.flush()


@contextmanager
def capture_crew_output():
    if _log_file is None:
        yield
        return
    old_stdout = sys.stdout
    sys.stdout = _StripAnsi(_log_file)
    try:
        yield
    finally:
        sys.stdout = old_stdout


@contextmanager
def run_log_file(run_id: str):
    global _log_file
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"run_{run_id}.log"
    _log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    handler_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        level="INFO",
        encoding="utf-8",
    )
    try:
        yield
    finally:
        logger.remove(handler_id)
        _log_file.close()
        _log_file = None
