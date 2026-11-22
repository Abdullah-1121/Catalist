"""LLM plumbing: a Postgres cache for LangChain, and a retry that waits as long as Google asks."""

import hashlib
import re
import time

import openai
from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from catalist.store import LLMCacheRow

# Google's 429 has no Retry-After header; the delay is only in the text: "Please retry in 48.0s."
RETRY_IN = re.compile(r"retry in ([\d.]+)s")


class PostgresCache(BaseCache):
    """Same prompt + same model settings -> stored answer, no API call."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\0{prompt}".encode()).hexdigest()

    def lookup(self, prompt, llm_string):
        with Session(self.engine) as session:
            row = session.get(LLMCacheRow, self._key(prompt, llm_string))
            # allowlist: rebuild only these classes from stored JSON, nothing else
            allowed = [ChatGeneration, AIMessage]
            return loads(row.response, allowed_objects=allowed) if row else None

    def update(self, prompt, llm_string, return_val):
        row = LLMCacheRow(
            key=self._key(prompt, llm_string), response=dumps(list(return_val))
        )
        with Session(self.engine) as session, session.begin():
            session.merge(row)

    def clear(self, **kwargs):
        with Session(self.engine) as session, session.begin():
            session.execute(delete(LLMCacheRow))


def call_with_retry(fn, max_attempts=5, sleep=time.sleep):
    """Call fn() and retry temporary failures.

    429: wait the delay Google names in the message (or 60 s, the quota window).
    5xx / timeout / network error: wait 10 s (usually back within seconds).
    """
    temporary = (
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APIConnectionError,  # includes timeouts
    )
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except temporary as error:
            if attempt == max_attempts:
                raise
            if not isinstance(error, openai.RateLimitError):
                sleep(10)
                continue
            delay = RETRY_IN.search(str(error))
            sleep(float(delay.group(1)) + 1 if delay else 60)
