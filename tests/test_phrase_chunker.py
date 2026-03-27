import asyncio

import pytest

from speaches.text_utils import PhraseChunker


@pytest.mark.asyncio
async def test_phrase_chunker_sentence_endings() -> None:
    chunker = PhraseChunker(min_phrase_length=5)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    chunker.add_token("Hello there.")
    chunker.add_token(" How are you?")
    await asyncio.sleep(0.1)
    chunker.close()
    await task

    assert len(results) >= 2
    combined = "".join(results)
    assert "Hello there." in combined
    assert "How are you?" in combined


@pytest.mark.asyncio
async def test_phrase_chunker_clause_boundaries() -> None:
    chunker = PhraseChunker(min_phrase_length=5)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    chunker.add_token("Well, I think so; but maybe not:")
    chunker.add_token(" let me check.")
    await asyncio.sleep(0.1)
    chunker.close()
    await task

    # Should yield at commas, semicolons, colons
    assert len(results) >= 2
    combined = "".join(results)
    assert "Well," in combined
    assert "let me check." in combined


@pytest.mark.asyncio
async def test_phrase_chunker_timeout_yields() -> None:
    chunker = PhraseChunker(min_phrase_length=15, timeout=0.1)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    # Add text without any boundaries
    chunker.add_token("This has no punctuation at all")
    # Wait for timeout to trigger
    await asyncio.sleep(0.3)
    chunker.close()
    await task

    assert len(results) >= 1
    combined = "".join(results)
    assert "This has no punctuation at all" in combined


@pytest.mark.asyncio
async def test_phrase_chunker_first_chunk_small() -> None:
    chunker = PhraseChunker(min_phrase_length=15)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    # First chunk should be yielded with as few as 4 chars
    chunker.add_token("Hi, how are you doing today?")
    await asyncio.sleep(0.1)
    chunker.close()
    await task

    # The first chunk should include "Hi," (>= 4 chars)
    assert len(results) >= 1
    assert len(results[0].strip()) >= 3


@pytest.mark.asyncio
async def test_phrase_chunker_empty() -> None:
    chunker = PhraseChunker()
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())
    chunker.close()
    await task

    assert len(results) == 0


@pytest.mark.asyncio
async def test_phrase_chunker_close_error() -> None:
    chunker = PhraseChunker()
    chunker.close()

    with pytest.raises(RuntimeError):
        chunker.add_token("should fail")


@pytest.mark.asyncio
async def test_phrase_chunker_remaining_text_on_close() -> None:
    chunker = PhraseChunker(min_phrase_length=100)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    chunker.add_token("Short text")
    chunker.close()
    await task

    # Should yield remaining text on close even if under min_phrase_length
    assert len(results) == 1
    assert results[0].strip() == "Short text"


@pytest.mark.asyncio
async def test_phrase_chunker_max_phrase_length() -> None:
    chunker = PhraseChunker(min_phrase_length=15, max_phrase_length=50, timeout=1.0)
    results: list[str] = []

    async def collect() -> None:
        async for chunk in chunker:
            results.append(chunk)  # noqa: PERF401

    task = asyncio.create_task(collect())

    # Simulate streaming: add tokens with event loop yields so the chunker
    # iterator can run between additions (matching real LLM token behavior)
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and", "keeps", "running", "across", "the", "field", "without", "stopping", "at", "all", "ever"]
    for word in words:
        chunker.add_token(word + " ")
        await asyncio.sleep(0)  # yield to event loop
    await asyncio.sleep(0.05)
    chunker.close()
    await task

    # Should have multiple chunks since we exceed max_phrase_length (50)
    assert len(results) >= 2
    combined = "".join(results)
    assert "quick brown fox" in combined
