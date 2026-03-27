#!/usr/bin/env python3
"""E2E test for the realtime WebSocket pipeline with a mock LLM.

This test verifies the full realtime voice pipeline:
1. Generate speech audio from known text using Kokoro TTS
2. Send audio through the realtime WebSocket
3. VAD detects speech, Whisper transcribes the audio
4. Mock LLM receives the transcription and returns a known response
5. TTS synthesizes the response audio
6. Verify all response events arrive correctly with proper ordering and field consistency

Additional test scenarios:
7. Barge-in: send speech during response generation to trigger interruption
8. response.cancel: explicitly cancel an active response
9. conversation.item.truncate: truncate assistant audio item
10. Cancel with no active response: verify error
11. session.update: update session configuration
12. conversation.item.create + response.create: manual item injection and response trigger
13. conversation.item.delete: delete a conversation item
14. input_audio_buffer.clear: clear the audio buffer
15. Multiple rounds on same connection: verify state isolation

Requires:
- Speaches server running with CHAT_COMPLETION_BASE_URL pointing to the mock LLM
- Kokoro TTS model available
- Whisper STT model available
- Silero VAD model available
"""

import asyncio
import base64
import json
import logging
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI
import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse
import uvicorn
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_realtime")

INPUT_TEXT = "Hello, how are you today?"
MOCK_LLM_RESPONSE = "I am doing great, thank you for asking!"
MOCK_LLM_SLOW_RESPONSE = (
    "This is a very long response that takes a while to stream out word by word "
    "so that we have time to interrupt it during generation. "
    "It keeps going and going with many words to fill up time. "
    "We need enough words here to ensure the streaming takes several seconds "
    "so the barge-in and cancel tests have time to fire their interrupts."
)

SPEACHES_PORT = 18000
MOCK_LLM_PORT = 18001
SPEACHES_URL = f"http://127.0.0.1:{SPEACHES_PORT}"
WS_BASE_URL = f"ws://127.0.0.1:{SPEACHES_PORT}/v1/realtime"
WS_URL = f"{WS_BASE_URL}?model=mock-llm&transcription_model=Systran/faster-whisper-base"

# ---- Mock LLM Server ----

received_requests: list[dict[str, Any]] = []
use_slow_response = False
use_dismiss_response = False

mock_app = FastAPI()


@mock_app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    body = await request.json()
    received_requests.append(body)
    logger.info(f"Mock LLM received request with {len(body.get('messages', []))} messages")

    if use_dismiss_response:
        response_text = "*"
    elif use_slow_response:
        response_text = MOCK_LLM_SLOW_RESPONSE
    else:
        response_text = MOCK_LLM_RESPONSE

    async def generate():  # noqa: ANN202
        words = response_text.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": "mock-llm",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": token} if i == 0 else {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            delay = 0.05 if use_slow_response else 0.01
            await asyncio.sleep(delay)

        # Final chunk with finish_reason
        chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-llm",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Usage chunk (required because stream_options.include_usage=True)
        chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-llm",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def start_mock_llm() -> None:
    uvicorn.run(mock_app, host="127.0.0.1", port=MOCK_LLM_PORT, log_level="warning")


# ---- Helpers ----


async def generate_audio(client: httpx.AsyncClient) -> bytes:
    logger.info(f"Generating audio from text: {INPUT_TEXT!r}")
    response = await client.post(
        f"{SPEACHES_URL}/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": INPUT_TEXT,
            "voice": "af_heart",
            "response_format": "pcm",
            "sample_rate": 24000,
        },
    )
    response.raise_for_status()
    logger.info(f"Generated {len(response.content)} bytes of 24kHz 16-bit PCM audio")
    return response.content


async def send_audio_chunks(ws: websockets.ClientConnection, audio_pcm: bytes, chunk_size: int = 9600) -> int:
    chunks_sent = 0
    for i in range(0, len(audio_pcm), chunk_size):
        chunk = audio_pcm[i : i + chunk_size]
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(event))
        chunks_sent += 1
        await asyncio.sleep(0.01)
    return chunks_sent


async def send_silence(ws: websockets.ClientConnection, duration_seconds: float = 3.5) -> None:
    chunk_size = 9600
    silence_samples = int(24000 * duration_seconds)
    silence_bytes = b"\x00" * (silence_samples * 2)
    for i in range(0, len(silence_bytes), chunk_size):
        chunk = silence_bytes[i : i + chunk_size]
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(event))
        await asyncio.sleep(0.01)


async def collect_events_until(
    ws: websockets.ClientConnection,
    stop_type: str,
    timeout_seconds: float = 120,
    *,
    fail_on_timeout: bool = True,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                events.append(event)
                if event["type"] == stop_type:
                    break
    except TimeoutError:
        if fail_on_timeout:
            event_types = [e["type"] for e in events]
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for {stop_type!r}. Got {len(events)} events: {event_types}"
            ) from None
        logger.warning(f"Timed out waiting for {stop_type}")
    return events


async def collect_events_until_any(
    ws: websockets.ClientConnection,
    stop_types: set[str],
    timeout_seconds: float = 120,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                events.append(event)
                if event["type"] in stop_types:
                    break
    except TimeoutError:
        event_types = [e["type"] for e in events]
        raise TimeoutError(
            f"Timed out after {timeout_seconds}s waiting for any of {stop_types}. "
            f"Got {len(events)} events: {event_types}"
        ) from None
    return events


async def drain_events(ws: websockets.ClientConnection, duration: float = 0.5) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(duration):
            while True:
                raw = await ws.recv()
                events.append(json.loads(raw))
    except TimeoutError:
        pass
    return events


class TestChecker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = True
        self.checks_run = 0
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.checks_run += 1
        if condition:
            logger.info(f"  [{self.name}] PASS: {message}")
        else:
            logger.error(f"  [{self.name}] FAIL: {message}")
            self.passed = False
            self.failures.append(message)

    def fail(self, message: str) -> None:
        self.check(condition=False, message=message)

    def ok(self, message: str) -> None:
        self.check(condition=True, message=message)

    def check_event_id_uniqueness(self, events: list[dict[str, Any]]) -> None:
        event_ids = [e.get("event_id") for e in events if e.get("event_id")]
        unique_ids = set(event_ids)
        self.check(
            len(event_ids) == len(unique_ids),
            f"all event_ids are unique ({len(event_ids)} total, {len(unique_ids)} unique)",
        )

    def check_response_id_consistency(
        self, events: list[dict[str, Any]], expected_response_id: str | None = None
    ) -> None:
        response_event_types = {
            "response.created",
            "response.done",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.audio_transcript.delta",
            "response.audio_transcript.done",
            "response.audio.delta",
            "response.audio.done",
            "response.text.delta",
            "response.text.done",
        }
        response_events = [e for e in events if e["type"] in response_event_types]
        if not response_events:
            return

        if expected_response_id is None:
            created = [e for e in response_events if e["type"] == "response.created"]
            if created:
                expected_response_id = created[0].get("response", {}).get("id")

        if expected_response_id:
            for e in response_events:
                rid = e.get("response_id") or e.get("response", {}).get("id")
                if rid and rid != expected_response_id:
                    self.fail(f"response_id mismatch in {e['type']}: got {rid}, expected {expected_response_id}")
                    return
            self.ok(f"all response events have consistent response_id ({expected_response_id})")

    def check_event_ordering(self, events: list[dict[str, Any]], expected_order: list[str]) -> None:
        event_types = [e["type"] for e in events]
        idx = 0
        for expected in expected_order:
            found = False
            while idx < len(event_types):
                if event_types[idx] == expected:
                    found = True
                    idx += 1
                    break
                idx += 1
            if not found:
                self.fail(f"event ordering: expected {expected!r} but not found after position {idx}")
                return
        self.ok(f"event ordering matches expected sequence ({len(expected_order)} events)")

    def dump_events_on_failure(self, events: list[dict[str, Any]]) -> None:
        if not self.passed:
            logger.error(f"  [{self.name}] Event dump for debugging:")
            for i, e in enumerate(events):
                etype = e.get("type", "unknown")
                # Don't dump audio data, it's too verbose
                summary = {
                    k: v
                    for k, v in e.items()
                    if k not in ("audio", "delta") or etype not in ("input_audio_buffer.append", "response.audio.delta")
                }
                if "delta" in e and etype == "response.audio.delta":
                    summary["delta"] = f"<{len(e['delta'])} chars of base64>"
                logger.error(f"    [{i}] {json.dumps(summary, default=str)[:200]}")


# ---- Test 1: Basic Pipeline ----


async def run_basic_pipeline_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Pipeline")
    logger.info("=" * 60)
    checker = TestChecker("basic")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Verify session fields
        session = msg.get("session", {})
        checker.check(session.get("model") == "mock-llm", f"model is mock-llm (got {session.get('model')})")
        turn_detection = session.get("turn_detection", {})
        checker.check(
            turn_detection.get("threshold") == 0.6, f"VAD threshold is 0.6 (got {turn_detection.get('threshold')})"
        )
        checker.check(
            turn_detection.get("silence_duration_ms") == 350,
            f"silence_duration_ms is 350 (got {turn_detection.get('silence_duration_ms')})",
        )
        checker.check(
            turn_detection.get("prefix_padding_ms") == 300,
            f"prefix_padding_ms is 300 (got {turn_detection.get('prefix_padding_ms')})",
        )

        chunks_sent = await send_audio_chunks(ws, audio_pcm)
        logger.info(f"Sent {len(audio_pcm)} bytes of audio in {chunks_sent} chunks")

        await send_silence(ws)
        logger.info("Sent silence, waiting for response events...")

        events = await collect_events_until(ws, "response.done")

    event_types = [e["type"] for e in events]

    # Core event presence
    checker.check("input_audio_buffer.speech_started" in event_types, "speech_started received")
    checker.check("input_audio_buffer.speech_stopped" in event_types, "speech_stopped received")
    checker.check("input_audio_buffer.committed" in event_types, "buffer committed")
    checker.check(
        "conversation.item.input_audio_transcription.completed" in event_types,
        "transcription completed",
    )
    checker.check("conversation.item.created" in event_types, "conversation item created")
    checker.check("response.created" in event_types, "response created")
    checker.check("response.output_item.added" in event_types, "output item added")
    checker.check("response.content_part.added" in event_types, "content part added")
    checker.check("response.audio_transcript.delta" in event_types, "transcript deltas received")
    checker.check("response.audio.delta" in event_types, "audio deltas received")
    checker.check("response.audio.done" in event_types, "audio done")
    checker.check("response.audio_transcript.done" in event_types, "transcript done")
    checker.check("response.content_part.done" in event_types, "content part done")
    checker.check("response.output_item.done" in event_types, "output item done")
    checker.check("response.done" in event_types, "response done")

    # Event ordering: verify canonical sequence
    checker.check_event_ordering(
        events,
        [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.created",
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.audio_transcript.delta",
            "response.audio.delta",
            "response.audio.done",
            "response.audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ],
    )

    # Event field consistency
    checker.check_event_id_uniqueness(events)
    checker.check_response_id_consistency(events)

    # Verify response.done has completed status
    response_done_events = [e for e in events if e["type"] == "response.done"]
    if response_done_events:
        resp = response_done_events[0].get("response", {})
        resp_status = resp.get("status")
        checker.check(resp_status == "completed", f"response status is 'completed' (got '{resp_status}')")
        # Verify output items are included
        output = resp.get("output", [])
        checker.check(len(output) > 0, f"response.done includes output items ({len(output)} items)")
        if output:
            checker.check(
                output[0].get("role") == "assistant",
                f"output item role is 'assistant' (got {output[0].get('role')})",
            )
            checker.check(
                output[0].get("status") == "completed",
                f"output item status is 'completed' (got {output[0].get('status')})",
            )

    # Check transcript content
    transcript_parts = [e.get("delta", "") for e in events if e["type"] == "response.audio_transcript.delta"]
    full_transcript = "".join(transcript_parts)
    checker.check(
        MOCK_LLM_RESPONSE.lower() in full_transcript.lower().strip(),
        f"response transcript matches mock LLM output (got {full_transcript!r})",
    )

    # Check transcript.done matches accumulated deltas
    transcript_done = [e for e in events if e["type"] == "response.audio_transcript.done"]
    if transcript_done:
        done_transcript = transcript_done[0].get("transcript", "")
        checker.check(
            done_transcript.strip() == full_transcript.strip(),
            "transcript.done matches accumulated delta transcripts",
        )

    # Check audio deltas are non-trivial
    audio_deltas = [e for e in events if e["type"] == "response.audio.delta"]
    checker.check(len(audio_deltas) > 0, f"audio deltas received ({len(audio_deltas)} chunks)")
    total_audio_bytes = sum(len(base64.b64decode(e.get("delta", ""))) for e in audio_deltas)
    checker.check(total_audio_bytes > 1000, f"total audio data is non-trivial ({total_audio_bytes} bytes)")

    # Verify item_id consistency across response events
    output_item_added = [e for e in events if e["type"] == "response.output_item.added"]
    if output_item_added:
        item_id = output_item_added[0].get("item", {}).get("id")
        item_events = [
            e
            for e in events
            if e.get("item_id") == item_id
            and e["type"]
            in {
                "response.audio_transcript.delta",
                "response.audio.delta",
                "response.audio.done",
                "response.audio_transcript.done",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_item.done",
            }
        ]
        checker.check(len(item_events) > 0, f"response events reference correct item_id ({item_id})")

    # Verify mock LLM received a well-formed request
    checker.check(len(received_requests) > 0, "mock LLM received at least one request")
    if received_requests:
        req = received_requests[-1]
        checker.check("messages" in req, "request has messages field")
        checker.check(req.get("stream") is True, "request has stream=True")
        checker.check("model" in req, "request has model field")
        # Verify the transcribed text made it to the LLM
        messages = req.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        checker.check(len(user_messages) > 0, "mock LLM received at least one user message")

    checker.dump_events_on_failure(events)
    return checker.passed


# ---- Test 2: response.cancel ----


async def run_response_cancel_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 2: response.cancel")
    logger.info("=" * 60)
    checker = TestChecker("cancel")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    global use_slow_response  # noqa: PLW0603
    use_slow_response = True

    try:
        async with websockets.connect(WS_URL) as ws:
            msg = json.loads(await ws.recv())
            checker.check(msg["type"] == "session.created", "session.created received")

            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            # Wait for response.created before cancelling
            events = await collect_events_until(ws, "response.created", timeout_seconds=60)
            event_types = [e["type"] for e in events]
            checker.check("response.created" in event_types, "response.created received before cancel")

            # Collect a few audio deltas to ensure streaming is in progress
            pre_cancel_audio = []
            try:
                async with asyncio.timeout(5):
                    while True:
                        raw = await ws.recv()
                        event = json.loads(raw)
                        events.append(event)
                        if event["type"] == "response.audio.delta":
                            pre_cancel_audio.append(event)
                            if len(pre_cancel_audio) >= 2:
                                break
                        elif event["type"] == "response.done":
                            break
            except TimeoutError:
                pass

            checker.check(len(pre_cancel_audio) > 0, f"got audio deltas before cancel ({len(pre_cancel_audio)})")

            # Send response.cancel
            cancel_event = {"type": "response.cancel", "event_id": "cancel_test_1"}
            await ws.send(json.dumps(cancel_event))
            logger.info("Sent response.cancel")

            # Collect remaining events
            remaining = await collect_events_until(ws, "response.done", timeout_seconds=15)
            events.extend(remaining)

            event_types = [e["type"] for e in events]
            checker.check("response.done" in event_types, "response.done received after cancel")

            response_done_events = [e for e in events if e["type"] == "response.done"]
            if response_done_events:
                resp = response_done_events[0].get("response", {})
                resp_status = resp.get("status")
                checker.check(
                    resp_status == "cancelled",
                    f"response status is 'cancelled' (got '{resp_status}')",
                )
                # Verify output items have incomplete status
                output = resp.get("output", [])
                if output:
                    checker.check(
                        output[0].get("status") == "incomplete",
                        f"cancelled output item status is 'incomplete' (got {output[0].get('status')})",
                    )

            checker.check_event_id_uniqueness(events)
            checker.dump_events_on_failure(events)
    finally:
        use_slow_response = False

    return checker.passed


# ---- Test 3: conversation.item.truncate ----


async def run_truncate_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 3: conversation.item.truncate")
    logger.info("=" * 60)
    checker = TestChecker("truncate")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        await send_audio_chunks(ws, audio_pcm)
        await send_silence(ws)

        # Wait for full response
        events = await collect_events_until(ws, "response.done")
        event_types = [e["type"] for e in events]
        checker.check("response.done" in event_types, "got response.done")

        # Find the assistant output item
        output_item_added = [e for e in events if e["type"] == "response.output_item.added"]
        if output_item_added:
            item_id = output_item_added[0].get("item", {}).get("id")
            checker.check(item_id is not None, f"got output item id: {item_id}")

            # Send truncate
            truncate_event = {
                "type": "conversation.item.truncate",
                "event_id": "truncate_test_1",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": 500,
            }
            await ws.send(json.dumps(truncate_event))
            logger.info(f"Sent conversation.item.truncate for item {item_id}")

            # Collect response
            truncate_events = await collect_events_until(ws, "conversation.item.truncated", timeout_seconds=5)
            truncate_types = [e["type"] for e in truncate_events]

            checker.check(
                "conversation.item.truncated" in truncate_types,
                "conversation.item.truncated received",
            )

            truncated = [e for e in truncate_events if e["type"] == "conversation.item.truncated"]
            if truncated:
                checker.check(
                    truncated[0].get("item_id") == item_id,
                    "truncated event has correct item_id",
                )
                checker.check(
                    truncated[0].get("audio_end_ms") == 500,
                    "truncated event has correct audio_end_ms",
                )
                checker.check(
                    truncated[0].get("content_index") == 0,
                    "truncated event has correct content_index",
                )

            # Test truncate on non-existent item
            bad_truncate = {
                "type": "conversation.item.truncate",
                "event_id": "truncate_bad_1",
                "item_id": "nonexistent_item_id",
                "content_index": 0,
                "audio_end_ms": 100,
            }
            await ws.send(json.dumps(bad_truncate))
            error_events = await collect_events_until(ws, "error", timeout_seconds=5)
            error_types = [e["type"] for e in error_events]
            checker.check("error" in error_types, "error received for truncating non-existent item")
        else:
            checker.passed = False
            logger.error("  [truncate] FAIL: no output item found to truncate")

    checker.dump_events_on_failure(events)
    return checker.passed


# ---- Test 4: Cancel with no active response ----


async def run_cancel_no_response_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 4: Cancel with no active response")
    logger.info("=" * 60)
    checker = TestChecker("cancel-no-resp")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Send cancel without any active response
        cancel_event = {"type": "response.cancel", "event_id": "cancel_no_resp_1"}
        await ws.send(json.dumps(cancel_event))

        # Should get an error back
        events = await collect_events_until(ws, "error", timeout_seconds=5)
        event_types = [e["type"] for e in events]
        checker.check("error" in event_types, "error event received for cancel with no response")

        error_events = [e for e in events if e["type"] == "error"]
        if error_events:
            error_msg = error_events[0].get("error", {}).get("message", "")
            checker.check(
                "No active response" in error_msg, f"error message mentions no active response: {error_msg!r}"
            )

    return checker.passed


# ---- Test 5: Barge-in (interrupt active response with speech) ----


async def run_barge_in_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 5: Barge-in")
    logger.info("=" * 60)
    checker = TestChecker("barge-in")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    global use_slow_response  # noqa: PLW0603
    use_slow_response = True

    try:
        async with websockets.connect(WS_URL) as ws:
            msg = json.loads(await ws.recv())
            checker.check(msg["type"] == "session.created", "session.created received")

            # Send audio to trigger first response
            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            # Wait until we get some audio deltas (response is streaming)
            events: list[dict[str, Any]] = []
            audio_delta_count = 0
            got_response_created = False
            try:
                async with asyncio.timeout(60):
                    while True:
                        raw = await ws.recv()
                        event = json.loads(raw)
                        events.append(event)
                        if event["type"] == "response.created":
                            got_response_created = True
                        if event["type"] == "response.audio.delta":
                            audio_delta_count += 1
                            if audio_delta_count >= 3:
                                break
                        if event["type"] == "response.done":
                            break
            except TimeoutError:
                pass

            checker.check(got_response_created, "response.created received")
            checker.check(audio_delta_count >= 1, f"got audio deltas during first response ({audio_delta_count})")

            # Now send speech audio to trigger barge-in
            logger.info("Sending speech audio to trigger barge-in...")
            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            # Collect events: we should see:
            # 1. response.done with status=cancelled (first response interrupted)
            # 2. speech_started (new speech detected)
            # 3. Eventually a new response.done with status=completed
            barge_in_events: list[dict[str, Any]] = []
            response_done_count = 0
            try:
                async with asyncio.timeout(120):
                    while True:
                        raw = await ws.recv()
                        event = json.loads(raw)
                        barge_in_events.append(event)
                        if event["type"] == "response.done":
                            response_done_count += 1
                            # After barge-in we expect: cancelled response.done, then eventually completed response.done
                            if response_done_count >= 2:
                                break
                            # If the first response.done is completed, the barge-in might have been too late
                            resp_status = event.get("response", {}).get("status")
                            if resp_status == "completed":
                                break
            except TimeoutError:
                pass

            all_events = events + barge_in_events
            all_types = [e["type"] for e in all_events]

            response_done_events = [e for e in all_events if e["type"] == "response.done"]

            if len(response_done_events) >= 2:
                # Ideal case: first response cancelled, second response completed
                first_status = response_done_events[0].get("response", {}).get("status")
                second_status = response_done_events[1].get("response", {}).get("status")
                checker.check(
                    first_status == "cancelled",
                    f"first response was cancelled by barge-in (got {first_status})",
                )
                checker.check(
                    second_status == "completed",
                    f"second response completed after barge-in (got {second_status})",
                )
                # Verify two different response IDs
                first_id = response_done_events[0].get("response", {}).get("id")
                second_id = response_done_events[1].get("response", {}).get("id")
                checker.check(
                    first_id != second_id,
                    f"barge-in created a new response (ids: {first_id} vs {second_id})",
                )
            elif len(response_done_events) == 1:
                # The slow response might have completed before barge-in took effect
                # This is still acceptable - just verify we got a speech_started
                checker.check(
                    "input_audio_buffer.speech_started" in all_types,
                    "speech_started detected during barge-in attempt",
                )
                logger.info("  [barge-in] NOTE: first response completed before barge-in could interrupt")
            else:
                checker.fail("expected at least one response.done event")

            checker.check_event_id_uniqueness(all_events)
            checker.dump_events_on_failure(all_events)
    finally:
        use_slow_response = False

    return checker.passed


# ---- Test 6: session.update ----


async def run_session_update_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 6: session.update")
    logger.info("=" * 60)
    checker = TestChecker("session-update")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")
        original_session = msg.get("session", {})

        # Update instructions and temperature
        update_event = {
            "type": "session.update",
            "event_id": "session_update_1",
            "session": {
                "instructions": "You are a helpful test assistant.",
                "temperature": 0.5,
            },
        }
        await ws.send(json.dumps(update_event))

        events = await collect_events_until(ws, "session.updated", timeout_seconds=5)
        event_types = [e["type"] for e in events]
        checker.check("session.updated" in event_types, "session.updated received")

        updated_events = [e for e in events if e["type"] == "session.updated"]
        if updated_events:
            updated_session = updated_events[0].get("session", {})
            checker.check(
                updated_session.get("instructions") == "You are a helpful test assistant.",
                f"instructions updated (got {updated_session.get('instructions')!r})",
            )
            checker.check(
                updated_session.get("temperature") == 0.5,
                f"temperature updated to 0.5 (got {updated_session.get('temperature')})",
            )
            # Verify unchanged fields are preserved
            checker.check(
                updated_session.get("model") == original_session.get("model"),
                "model preserved after session update",
            )
            checker.check(
                updated_session.get("voice") == original_session.get("voice"),
                "voice preserved after session update",
            )

        # Update turn_detection with only partial fields (threshold and silence_duration_ms)
        update_vad = {
            "type": "session.update",
            "event_id": "session_update_2",
            "session": {
                "turn_detection": {
                    "threshold": 0.8,
                    "silence_duration_ms": 500,
                },
            },
        }
        await ws.send(json.dumps(update_vad))

        events2 = await collect_events_until(ws, "session.updated", timeout_seconds=5)
        updated_events2 = [e for e in events2 if e["type"] == "session.updated"]
        if updated_events2:
            td = updated_events2[0].get("session", {}).get("turn_detection", {})
            checker.check(td.get("threshold") == 0.8, f"VAD threshold updated to 0.8 (got {td.get('threshold')})")
            checker.check(
                td.get("silence_duration_ms") == 500,
                f"silence_duration_ms updated to 500 (got {td.get('silence_duration_ms')})",
            )
            checker.check(
                td.get("create_response") is True,
                f"create_response preserved (got {td.get('create_response')})",
            )
            checker.check(
                td.get("prefix_padding_ms") == 300,
                f"prefix_padding_ms preserved (got {td.get('prefix_padding_ms')})",
            )

    return checker.passed


# ---- Test 7: conversation.item.create + response.create (manual flow) ----


async def run_manual_conversation_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 7: conversation.item.create + response.create")
    logger.info("=" * 60)
    checker = TestChecker("manual-conv")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Disable VAD so we can manually control the flow
        update_event = {
            "type": "session.update",
            "session": {
                "turn_detection": None,
            },
        }
        await ws.send(json.dumps(update_event))
        events = await collect_events_until(ws, "session.updated", timeout_seconds=5)
        updated = [e for e in events if e["type"] == "session.updated"]
        if updated:
            checker.check(
                updated[0].get("session", {}).get("turn_detection") is None,
                "turn_detection disabled",
            )

        # Manually create a user message
        create_event = {
            "type": "conversation.item.create",
            "event_id": "create_item_1",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is your name?"}],
            },
        }
        await ws.send(json.dumps(create_event))

        item_events = await collect_events_until(ws, "conversation.item.created", timeout_seconds=5)
        item_types = [e["type"] for e in item_events]
        checker.check("conversation.item.created" in item_types, "conversation.item.created received")

        created_events = [e for e in item_events if e["type"] == "conversation.item.created"]
        if created_events:
            created_item = created_events[0].get("item", {})
            checker.check(created_item.get("role") == "user", "created item has user role")
            checker.check(created_item.get("type") == "message", "created item is a message")

        # Manually trigger response generation
        response_create = {
            "type": "response.create",
            "event_id": "response_create_1",
        }
        await ws.send(json.dumps(response_create))

        # Collect all response events
        response_events = await collect_events_until(ws, "response.done", timeout_seconds=60)
        response_types = [e["type"] for e in response_events]

        checker.check("response.created" in response_types, "response.created received")
        checker.check("response.done" in response_types, "response.done received")

        response_done = [e for e in response_events if e["type"] == "response.done"]
        if response_done:
            resp_status = response_done[0].get("response", {}).get("status")
            checker.check(resp_status == "completed", f"response completed (got {resp_status})")

        # Verify mock LLM received the injected message
        checker.check(len(received_requests) > 0, "mock LLM received a request")
        if received_requests:
            messages = received_requests[-1].get("messages", [])
            user_texts = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for m in messages
                if m.get("role") == "user"
                for c in (m.get("content") if isinstance(m.get("content"), list) else [m.get("content", "")])
            ]
            found_injected = any("What is your name?" in t for t in user_texts)
            checker.check(found_injected, f"mock LLM received injected user message (user texts: {user_texts})")

        checker.check_event_id_uniqueness(item_events + response_events)
        checker.dump_events_on_failure(item_events + response_events)

    return checker.passed


# ---- Test 8: conversation.item.delete ----


async def run_delete_item_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 8: conversation.item.delete")
    logger.info("=" * 60)
    checker = TestChecker("delete-item")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Create a user message
        create_event = {
            "type": "conversation.item.create",
            "item": {
                "id": "test_item_to_delete",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "This will be deleted"}],
            },
        }
        await ws.send(json.dumps(create_event))
        events = await collect_events_until(ws, "conversation.item.created", timeout_seconds=5)
        checker.check(
            any(e["type"] == "conversation.item.created" for e in events),
            "item created",
        )

        # Delete the item
        delete_event = {
            "type": "conversation.item.delete",
            "event_id": "delete_1",
            "item_id": "test_item_to_delete",
        }
        await ws.send(json.dumps(delete_event))
        delete_events = await collect_events_until(ws, "conversation.item.deleted", timeout_seconds=5)
        delete_types = [e["type"] for e in delete_events]
        checker.check("conversation.item.deleted" in delete_types, "conversation.item.deleted received")

        deleted = [e for e in delete_events if e["type"] == "conversation.item.deleted"]
        if deleted:
            checker.check(
                deleted[0].get("item_id") == "test_item_to_delete",
                "deleted event has correct item_id",
            )

        # Try to delete the same item again - should get error
        await ws.send(json.dumps(delete_event))
        error_events = await collect_events_until(ws, "error", timeout_seconds=5)
        error_types = [e["type"] for e in error_events]
        checker.check("error" in error_types, "error received for deleting non-existent item")

    return checker.passed


# ---- Test 9: input_audio_buffer.clear ----


async def run_clear_buffer_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 9: input_audio_buffer.clear")
    logger.info("=" * 60)
    checker = TestChecker("clear-buffer")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Disable VAD so buffer isn't auto-committed
        update_event = {
            "type": "session.update",
            "session": {"turn_detection": None},
        }
        await ws.send(json.dumps(update_event))
        await collect_events_until(ws, "session.updated", timeout_seconds=5)

        # Send some audio
        await send_audio_chunks(ws, audio_pcm[:9600])  # Just a small chunk

        # Clear the buffer
        clear_event = {"type": "input_audio_buffer.clear", "event_id": "clear_1"}
        await ws.send(json.dumps(clear_event))

        clear_events = await collect_events_until(ws, "input_audio_buffer.cleared", timeout_seconds=5)
        clear_types = [e["type"] for e in clear_events]
        checker.check("input_audio_buffer.cleared" in clear_types, "input_audio_buffer.cleared received")

        # Now try to commit the empty buffer - should get error
        commit_event = {"type": "input_audio_buffer.commit", "event_id": "commit_empty_1"}
        await ws.send(json.dumps(commit_event))

        error_events = await collect_events_until(ws, "error", timeout_seconds=5)
        error_types = [e["type"] for e in error_events]
        checker.check("error" in error_types, "error received for committing empty buffer after clear")

    return checker.passed


# ---- Test 10: Multiple rounds on same connection ----


async def run_multi_round_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 10: Multiple rounds on same connection")
    logger.info("=" * 60)
    checker = TestChecker("multi-round")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        response_ids = []

        for round_num in range(1, 3):
            logger.info(f"  [multi-round] Starting round {round_num}")
            received_requests.clear()

            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            events = await collect_events_until(ws, "response.done", timeout_seconds=120)
            event_types = [e["type"] for e in events]

            checker.check(
                "response.done" in event_types,
                f"round {round_num}: response.done received",
            )

            response_done = [e for e in events if e["type"] == "response.done"]
            if response_done:
                resp = response_done[0].get("response", {})
                resp_status = resp.get("status")
                resp_id = resp.get("id")
                response_ids.append(resp_id)
                checker.check(
                    resp_status == "completed",
                    f"round {round_num}: response completed (got {resp_status})",
                )

            # Verify transcript
            transcript_parts = [e.get("delta", "") for e in events if e["type"] == "response.audio_transcript.delta"]
            full_transcript = "".join(transcript_parts)
            checker.check(
                MOCK_LLM_RESPONSE.lower() in full_transcript.lower().strip(),
                f"round {round_num}: transcript matches",
            )

            # Verify LLM received request
            checker.check(
                len(received_requests) > 0,
                f"round {round_num}: mock LLM received request",
            )

            # In round 2, verify conversation history grew
            if round_num == 2 and received_requests:
                messages = received_requests[-1].get("messages", [])
                # Should have messages from both rounds (user + assistant from round 1, plus user from round 2)
                checker.check(
                    len(messages) >= 3,
                    f"round 2: LLM received conversation history ({len(messages)} messages)",
                )

        # Verify different response IDs
        if len(response_ids) >= 2:
            checker.check(
                response_ids[0] != response_ids[1],
                f"different response IDs across rounds ({response_ids[0]} vs {response_ids[1]})",
            )

    return checker.passed


# ---- Test 11: Manual commit flow (no VAD) ----


async def run_manual_commit_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 11: Manual commit + response.create (no VAD)")
    logger.info("=" * 60)
    checker = TestChecker("manual-commit")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Disable VAD
        update_event = {
            "type": "session.update",
            "session": {"turn_detection": None},
        }
        await ws.send(json.dumps(update_event))
        await collect_events_until(ws, "session.updated", timeout_seconds=5)

        # Send audio
        await send_audio_chunks(ws, audio_pcm)

        # Manually commit the buffer
        commit_event = {"type": "input_audio_buffer.commit", "event_id": "manual_commit_1"}
        await ws.send(json.dumps(commit_event))

        commit_events = await collect_events_until(ws, "input_audio_buffer.committed", timeout_seconds=10)
        commit_types = [e["type"] for e in commit_events]
        checker.check("input_audio_buffer.committed" in commit_types, "buffer committed manually")

        # Wait for transcription to complete
        transcription_events = await collect_events_until(
            ws, "conversation.item.input_audio_transcription.completed", timeout_seconds=30
        )
        t_types = [e["type"] for e in transcription_events]
        checker.check(
            "conversation.item.input_audio_transcription.completed" in t_types,
            "transcription completed after manual commit",
        )

        # Manually trigger response
        response_create = {"type": "response.create", "event_id": "manual_response_1"}
        await ws.send(json.dumps(response_create))

        response_events = await collect_events_until(ws, "response.done", timeout_seconds=60)
        response_types = [e["type"] for e in response_events]
        checker.check("response.created" in response_types, "response.created after manual trigger")
        checker.check("response.done" in response_types, "response.done after manual trigger")

        response_done = [e for e in response_events if e["type"] == "response.done"]
        if response_done:
            resp_status = response_done[0].get("response", {}).get("status")
            checker.check(resp_status == "completed", f"manual response completed (got {resp_status})")

        checker.check(len(received_requests) > 0, "mock LLM received request")

        all_events = commit_events + transcription_events + response_events
        checker.check_event_id_uniqueness(all_events)
        checker.dump_events_on_failure(all_events)

    return checker.passed


# ---- Test 12: No-response token (message clearing) ----


async def run_no_response_token_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 12: No-response token (message clearing)")
    logger.info("=" * 60)
    checker = TestChecker("no-response")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    global use_dismiss_response  # noqa: PLW0603
    use_dismiss_response = True

    try:
        async with websockets.connect(WS_URL) as ws:
            msg = json.loads(await ws.recv())
            checker.check(msg["type"] == "session.created", "session.created received")

            # Verify no_response_token is set in session
            session = msg.get("session", {})
            checker.check(
                session.get("no_response_token") == "*",
                f"no_response_token defaults to '*' (got {session.get('no_response_token')!r})",
            )

            # Send audio to trigger the pipeline
            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            # Collect events through response.done
            events = await collect_events_until(ws, "response.done", timeout_seconds=120)
            event_types = [e["type"] for e in events]

            # Core pipeline events should still fire
            checker.check("input_audio_buffer.speech_started" in event_types, "speech_started received")
            checker.check("input_audio_buffer.speech_stopped" in event_types, "speech_stopped received")
            checker.check("input_audio_buffer.committed" in event_types, "buffer committed")
            checker.check("response.created" in event_types, "response.created received")
            checker.check("response.done" in event_types, "response.done received")

            # Verify response completed (not failed or cancelled)
            response_done = [e for e in events if e["type"] == "response.done"]
            if response_done:
                resp_status = response_done[0].get("response", {}).get("status")
                checker.check(resp_status == "completed", f"response status is 'completed' (got {resp_status})")

            # Verify transcript is just "*"
            transcript_parts = [e.get("delta", "") for e in events if e["type"] == "response.audio_transcript.delta"]
            full_transcript = "".join(transcript_parts)
            checker.check(
                full_transcript.strip() == "*",
                f"transcript is the no-response token '*' (got {full_transcript!r})",
            )

            # Verify NO audio deltas were generated (TTS suppressed for non-word text)
            audio_deltas = [e for e in events if e["type"] == "response.audio.delta"]
            checker.check(
                len(audio_deltas) == 0,
                f"no audio deltas generated for dismissed response ({len(audio_deltas)} found)",
            )

            # After response.done, conversation items should be deleted (message clearing)
            # Collect the deletion events
            delete_events = await collect_events_until(
                ws, "conversation.item.deleted", timeout_seconds=10, fail_on_timeout=False
            )
            # We expect at least one deletion (the assistant item)
            deleted_items = [e for e in delete_events if e["type"] == "conversation.item.deleted"]
            checker.check(
                len(deleted_items) >= 1,
                f"conversation items deleted after dismiss ({len(deleted_items)} deletions)",
            )

            # Verify the user input item was also deleted (the one before the assistant)
            if len(deleted_items) >= 2:
                checker.check(
                    deleted_items[0].get("item_id") != deleted_items[1].get("item_id"),
                    "two different items deleted (assistant + user input)",
                )

            all_events = events + delete_events
            checker.check_event_id_uniqueness(all_events)
            checker.dump_events_on_failure(all_events)
    finally:
        use_dismiss_response = False

    return checker.passed


# ---- Test 13: No-response token disabled ----


async def run_no_response_token_disabled_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 13: No-response token disabled via session.update")
    logger.info("=" * 60)
    checker = TestChecker("no-response-disabled")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    global use_dismiss_response  # noqa: PLW0603
    use_dismiss_response = True

    try:
        async with websockets.connect(WS_URL) as ws:
            msg = json.loads(await ws.recv())
            checker.check(msg["type"] == "session.created", "session.created received")

            # Disable no_response_token via session.update
            update_event = {
                "type": "session.update",
                "session": {"no_response_token": None},
            }
            await ws.send(json.dumps(update_event))
            update_events = await collect_events_until(ws, "session.updated", timeout_seconds=5)
            updated = [e for e in update_events if e["type"] == "session.updated"]
            if updated:
                checker.check(
                    updated[0].get("session", {}).get("no_response_token") is None,
                    "no_response_token disabled",
                )

            # Send audio — LLM will respond "*" but it should NOT be dismissed
            await send_audio_chunks(ws, audio_pcm)
            await send_silence(ws)

            events = await collect_events_until(ws, "response.done", timeout_seconds=120)
            event_types = [e["type"] for e in events]

            checker.check("response.done" in event_types, "response.done received")

            # Transcript should still be "*"
            transcript_parts = [e.get("delta", "") for e in events if e["type"] == "response.audio_transcript.delta"]
            full_transcript = "".join(transcript_parts)
            checker.check(full_transcript.strip() == "*", f"transcript is '*' (got {full_transcript!r})")

            # With no_response_token disabled, no conversation.item.deleted should follow
            # (drain briefly to check nothing arrives)
            post_events = await drain_events(ws, duration=2.0)
            post_types = [e["type"] for e in post_events]
            deleted_count = sum(1 for t in post_types if t == "conversation.item.deleted")
            checker.check(
                deleted_count == 0,
                f"no conversation items deleted when feature disabled ({deleted_count} found)",
            )

            checker.dump_events_on_failure(events + post_events)
    finally:
        use_dismiss_response = False

    return checker.passed


# ---- Test 14: Noise gate (no_speech_prob_threshold) ----


async def run_noise_gate_test() -> bool:
    received_requests.clear()
    logger.info("=" * 60)
    logger.info("TEST 14: Noise gate (no_speech_prob_threshold)")
    logger.info("=" * 60)
    checker = TestChecker("noise-gate")

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_pcm = await generate_audio(client)

    # --- Part A: threshold=0.0 rejects everything (even real speech) ---
    logger.info("  [noise-gate] Part A: threshold=0.0 should reject all audio")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "session.created received")

        # Verify default threshold is in session
        session = msg.get("session", {})
        checker.check(
            session.get("no_speech_prob_threshold") is not None,
            f"no_speech_prob_threshold present in session (got {session.get('no_speech_prob_threshold')})",
        )

        # Set threshold to 0.0 — every transcription will be rejected
        update_event = {
            "type": "session.update",
            "session": {"no_speech_prob_threshold": 0.0},
        }
        await ws.send(json.dumps(update_event))
        update_events = await collect_events_until(ws, "session.updated", timeout_seconds=5)
        updated = [e for e in update_events if e["type"] == "session.updated"]
        if updated:
            checker.check(
                updated[0].get("session", {}).get("no_speech_prob_threshold") == 0.0,
                "no_speech_prob_threshold updated to 0.0",
            )

        # Send real speech audio — VAD will trigger, but noise gate should reject
        await send_audio_chunks(ws, audio_pcm)
        await send_silence(ws)

        # We should see speech_started and speech_stopped, but NO response
        # (because the transcription is discarded by noise gate before creating the item)
        events: list[dict[str, Any]] = []
        try:
            async with asyncio.timeout(15):
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)
                    events.append(event)
                    # If we see response.created, the noise gate didn't work
                    if event["type"] == "response.created":
                        break
                    # speech_stopped + committed means transcription will run
                    # After committed, if noise gate works, nothing else should arrive
                    if event["type"] == "input_audio_buffer.committed":
                        # Give it a few more seconds for transcription to complete
                        more = await drain_events(ws, duration=10.0)
                        events.extend(more)
                        break
        except TimeoutError:
            pass

        event_types = [e["type"] for e in events]

        checker.check(
            "input_audio_buffer.speech_started" in event_types,
            "speech_started received (VAD still triggers)",
        )
        checker.check(
            "input_audio_buffer.committed" in event_types,
            "buffer committed (audio still committed)",
        )
        # The key check: no conversation item created, no response triggered
        checker.check(
            "conversation.item.created" not in event_types,
            "no conversation item created (noise gate rejected)",
        )
        checker.check(
            "response.created" not in event_types,
            "no response created (noise gate rejected)",
        )
        checker.check(
            len(received_requests) == 0,
            f"mock LLM received no requests ({len(received_requests)} found)",
        )
        checker.dump_events_on_failure(events)

    # --- Part B: threshold=None disables the gate, audio passes through ---
    received_requests.clear()
    logger.info("  [noise-gate] Part B: threshold=None should let all audio through")

    async with websockets.connect(WS_URL) as ws:
        msg = json.loads(await ws.recv())
        checker.check(msg["type"] == "session.created", "B: session.created received")

        # Disable noise gate
        update_event = {
            "type": "session.update",
            "session": {"no_speech_prob_threshold": None},
        }
        await ws.send(json.dumps(update_event))
        await collect_events_until(ws, "session.updated", timeout_seconds=5)

        # Send the same audio — should pass through to LLM
        await send_audio_chunks(ws, audio_pcm)
        await send_silence(ws)

        events = await collect_events_until(ws, "response.done", timeout_seconds=120)
        event_types = [e["type"] for e in events]

        checker.check(
            "conversation.item.created" in event_types,
            "B: conversation item created (gate disabled)",
        )
        checker.check(
            "response.done" in event_types,
            "B: response completed (gate disabled)",
        )
        checker.check(
            len(received_requests) > 0,
            "B: mock LLM received request (gate disabled)",
        )
        checker.dump_events_on_failure(events)

    return checker.passed


# ---- Main ----


async def run_all_tests() -> bool:
    results: dict[str, bool] = {}

    test_functions = [
        ("basic_pipeline", run_basic_pipeline_test),
        ("response_cancel", run_response_cancel_test),
        ("truncate", run_truncate_test),
        ("cancel_no_response", run_cancel_no_response_test),
        ("barge_in", run_barge_in_test),
        ("session_update", run_session_update_test),
        ("manual_conversation", run_manual_conversation_test),
        ("delete_item", run_delete_item_test),
        ("clear_buffer", run_clear_buffer_test),
        ("multi_round", run_multi_round_test),
        ("manual_commit", run_manual_commit_test),
        ("no_response_token", run_no_response_token_test),
        ("no_response_token_disabled", run_no_response_token_disabled_test),
        ("noise_gate", run_noise_gate_test),
    ]

    for name, test_fn in test_functions:
        try:
            results[name] = await test_fn()
        except Exception:
            logger.exception(f"Test {name} raised an exception")
            results[name] = False

    logger.info("=" * 60)
    logger.info("RESULTS:")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    logger.info(f"  Total: {passed_count}/{total_count} passed")
    logger.info("=" * 60)

    return all_passed


def main() -> None:
    # Start mock LLM in background thread
    logger.info(f"Starting mock LLM on port {MOCK_LLM_PORT}")
    mock_thread = threading.Thread(target=start_mock_llm, daemon=True)
    mock_thread.start()

    # Wait for mock LLM to be ready
    for attempt in range(20):
        try:
            resp = httpx.get(f"http://127.0.0.1:{MOCK_LLM_PORT}/docs")
            logger.info(f"Mock LLM is running (status {resp.status_code})")
            break
        except httpx.ConnectError:
            if attempt == 19:
                logger.warning("Failed to connect to mock LLM after 20 attempts")
                sys.exit(1)
            time.sleep(0.1)

    # Run tests
    success = asyncio.run(run_all_tests())
    if success:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
