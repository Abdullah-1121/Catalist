import httpx
import openai
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_openai.chat_models.base import (
    OpenAIAPIError,
    OpenAIRateLimitError,
    OpenAITimeoutError,
)

from catalist.llm import PostgresCache, call_with_retry

# ---------- cache ----------


def test_cache_miss_returns_none(engine):
    assert PostgresCache(engine).lookup("hello", "model-a") is None


def test_cache_returns_what_was_stored(engine):
    cache = PostgresCache(engine)
    cache.update("hello", "model-a", [ChatGeneration(message=AIMessage(content="hi!"))])
    [gen] = cache.lookup("hello", "model-a")
    assert gen.message.content == "hi!"


def test_cache_key_includes_the_model(engine):
    cache = PostgresCache(engine)
    cache.update("hello", "model-a", [ChatGeneration(message=AIMessage(content="hi!"))])
    assert cache.lookup("hello", "model-b") is None


def test_cache_accepts_a_whole_meeting_as_prompt(engine):
    # LangChain's SQLAlchemyCache crashes here: Postgres index entries max 8191 bytes
    prompt = "[#1 A] we decided on a yellow banana-shaped remote. " * 2000  # ~100 KB
    cache = PostgresCache(engine)
    cache.update(prompt, "model-a", [ChatGeneration(message=AIMessage(content="ok"))])
    assert cache.lookup(prompt, "model-a")[0].message.content == "ok"


def test_second_identical_call_comes_from_cache_not_the_model(engine):
    # the fake model would answer "second" on a real 2nd call; the cache must answer "first"
    model = FakeListChatModel(
        responses=["first", "second"], cache=PostgresCache(engine)
    )
    assert model.invoke("same question").content == "first"
    assert model.invoke("same question").content == "first"


# ---------- retry on 429 ----------


# The exact error classes ChatOpenAI raises (it wraps the openai SDK's errors).
def rate_limit_error(message):
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://example.test")
    )
    return OpenAIRateLimitError(message, response=response, body=None)


def overloaded_error():
    response = httpx.Response(
        503, request=httpx.Request("POST", "https://example.test")
    )
    return OpenAIAPIError("Error code: 503", response=response, body=None)


def flaky(errors, result="ok"):
    """A function that raises the given errors one by one, then returns result."""
    errors = list(errors)

    def fn():
        if errors:
            raise errors.pop(0)
        return result

    return fn


def test_retry_waits_as_long_as_google_says():
    waits = []
    fn = flaky([rate_limit_error("Quota exceeded ... Please retry in 48.006513235s.")])
    assert call_with_retry(fn, sleep=waits.append) == "ok"
    assert waits == [
        pytest.approx(49, abs=1)
    ]  # the 48 s from the message, plus a margin


def test_retry_waits_a_full_minute_when_no_delay_is_given():
    waits = []
    assert (
        call_with_retry(
            flaky([rate_limit_error("Too many requests")]), sleep=waits.append
        )
        == "ok"
    )
    assert waits == [60]


def test_retry_gives_up_after_max_attempts():
    errors = [rate_limit_error("retry in 1s")] * 3
    with pytest.raises(openai.RateLimitError):
        call_with_retry(flaky(errors), max_attempts=3, sleep=lambda s: None)


def test_retry_on_server_overload_waits_30_seconds():
    # Google returned 503 during our first real call
    waits = []
    assert call_with_retry(flaky([overloaded_error()]), sleep=waits.append) == "ok"
    assert waits == [30]


def test_retry_on_timeout_waits_30_seconds():
    # a real call once hung for 5+ minutes; with a timeout set, the hang becomes this error
    waits = []
    timeout = OpenAITimeoutError(httpx.Request("POST", "https://example.test"))
    assert call_with_retry(flaky([timeout]), sleep=waits.append) == "ok"
    assert waits == [30]
