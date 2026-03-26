import asyncio
from collections.abc import AsyncGenerator
import re
from typing import Protocol


class TextChunker(Protocol):
    """Protocol defining the interface for text chunkers."""

    def add_token(self, token: str) -> None:
        """Add a token (text chunk) to the chunker."""
        ...

    def close(self) -> None:
        """Close the chunker, preventing further token additions."""
        ...

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        yield ""


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def srt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def vtt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def format_as_vtt(text: str, start: float, end: float, i: int) -> str:
    start = start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(end)}\n{text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def format_as_srt(text: str, start: float, end: float, i: int) -> str:
    return f"{i + 1}\n{srt_format_timestamp(start)} --> {srt_format_timestamp(end)}\n{text}\n\n"


MIN_SENTENCE_LENGTH = 20


# TODO: Add tests
# TODO: take into account various sentence endings like "..."
# TODO: maybe create MultiSentenceChunker to return multiple sentence (when available) at a time
# TODO: consider different handling of small sentences. i.e. if a sentence consist of only couple of words wait until more words are available
class SentenceChunker:
    """A text chunker that yields text in sentence chunks.

    Implements the TextChunker protocol.
    """

    def __init__(self, min_sentence_length: int = MIN_SENTENCE_LENGTH) -> None:
        self._content = ""
        self._is_closed = False
        self._new_token_event = asyncio.Event()
        self._sentence_endings = {".", "!", "?"}
        self._processed_index = 0
        self._min_sentence_length = min_sentence_length
        self._accumulated_text = ""

    def add_token(self, token: str) -> None:
        """Add a token (text chunk) to the chunker."""
        if self._is_closed:
            raise RuntimeError("Cannot add tokens to a closed SentenceChunker")

        self._content += token
        self._new_token_event.set()

    def close(self) -> None:
        """Close the chunker, preventing further token additions."""
        self._is_closed = True
        self._new_token_event.set()

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        while True:
            # Find the next sentence ending after the last processed index
            next_end = -1
            for ending in self._sentence_endings:
                pos = self._content.find(ending, self._processed_index)
                if pos != -1 and (next_end == -1 or pos < next_end):
                    next_end = pos

            if next_end != -1:
                # We found a complete sentence
                sentence_end = next_end + 1
                new_sentence = self._content[self._processed_index : sentence_end]
                self._processed_index = sentence_end

                # Combine with any previously accumulated text
                combined_text = self._accumulated_text + new_sentence

                # Check if the combined text meets the minimum length requirement
                if len(combined_text.strip()) >= self._min_sentence_length:
                    self._accumulated_text = ""  # Reset accumulated text
                    yield combined_text
                else:
                    # If too short, accumulate for next time
                    self._accumulated_text = combined_text
            else:
                # No complete sentence found
                if self._is_closed:
                    # If there's any remaining content, combine with accumulated text and yield
                    if self._processed_index < len(self._content) or self._accumulated_text:
                        remaining = (
                            self._content[self._processed_index :] if self._processed_index < len(self._content) else ""
                        )
                        final_text = self._accumulated_text + remaining
                        if final_text.strip():  # Only yield if there's non-whitespace content
                            yield final_text
                    return

                # Wait for more content
                self._new_token_event.clear()
                await self._new_token_event.wait()


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub(r"", text)


_BOLD_PATTERN = re.compile(r"\*\*(.*?)\*\*")
_ITALIC_STAR_PATTERN = re.compile(r"\*(.*?)\*")
_UNDERLINE_PATTERN = re.compile(r"__(.*?)__")
_ITALIC_UNDERSCORE_PATTERN = re.compile(r"_(.*?)_")


def strip_markdown_emphasis(text: str) -> str:
    # Remove bold (**text**)
    text = _BOLD_PATTERN.sub(r"\1", text)
    # Remove italic (*text*)
    text = _ITALIC_STAR_PATTERN.sub(r"\1", text)
    # Remove underlined (__text__)
    text = _UNDERLINE_PATTERN.sub(r"\1", text)
    # Remove italic with underscore (_text_)
    text = _ITALIC_UNDERSCORE_PATTERN.sub(r"\1", text)

    return text


def clean_for_tts(text: str) -> str:
    text = text.strip()
    text = strip_markdown_emphasis(text)
    text = strip_emojis(text)
    text = text.strip()
    # Skip text with no word characters (e.g. "*", "---", "...")
    if text and not re.search(r"\w", text):
        return ""
    return text


MIN_PHRASE_LENGTH = 15
MAX_PHRASE_LENGTH = 200
PHRASE_TIMEOUT_SECONDS = 0.2


class PhraseChunker:
    """A text chunker that yields text at clause/phrase boundaries for low-latency TTS.

    Yields at commas, semicolons, colons, dashes, sentence endings, or after a
    timeout. The first chunk can be as small as a few words to minimize
    time-to-first-voice.

    Implements the TextChunker protocol.
    """

    def __init__(
        self,
        min_phrase_length: int = MIN_PHRASE_LENGTH,
        max_phrase_length: int = MAX_PHRASE_LENGTH,
        timeout: float = PHRASE_TIMEOUT_SECONDS,
    ) -> None:
        self._content = ""
        self._is_closed = False
        self._new_token_event = asyncio.Event()
        self._boundary_pattern = re.compile(r"[.!?,;:\u2014-]")
        self._processed_index = 0
        self._min_phrase_length = min_phrase_length
        self._max_phrase_length = max_phrase_length
        self._timeout = timeout
        self._is_first_chunk = True

    def add_token(self, token: str) -> None:
        if self._is_closed:
            raise RuntimeError("Cannot add tokens to a closed PhraseChunker")
        self._content += token
        self._new_token_event.set()

    def close(self) -> None:
        self._is_closed = True
        self._new_token_event.set()

    def _find_next_boundary(self) -> int | None:
        m = self._boundary_pattern.search(self._content, self._processed_index)
        if m is None:
            return None
        return m.end()

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        while True:
            boundary = self._find_next_boundary()
            if boundary is not None:
                chunk = self._content[self._processed_index : boundary]
                min_len = 4 if self._is_first_chunk else self._min_phrase_length
                if len(chunk.strip()) >= min_len:
                    self._processed_index = boundary
                    self._is_first_chunk = False
                    yield chunk
                    continue

            unprocessed = self._content[self._processed_index :]

            if len(unprocessed.strip()) >= self._max_phrase_length:
                self._processed_index = len(self._content)
                self._is_first_chunk = False
                yield unprocessed
                continue

            if self._is_closed:
                if unprocessed.strip():
                    yield unprocessed
                return

            self._new_token_event.clear()
            try:
                await asyncio.wait_for(self._new_token_event.wait(), timeout=self._timeout)
            except TimeoutError:
                unprocessed = self._content[self._processed_index :]
                if len(unprocessed.strip()) >= 4:
                    self._processed_index = len(self._content)
                    self._is_first_chunk = False
                    yield unprocessed


class EOFTextChunker:
    """A text chunker that yields all accumulated text only when closed.

    Implements the TextChunker protocol.
    """

    def __init__(self) -> None:
        self._content = ""
        self._is_closed = False
        self._new_token_event = asyncio.Event()

    def add_token(self, token: str) -> None:
        """Add a token (text chunk) to the chunker."""
        if self._is_closed:
            raise RuntimeError("Cannot add tokens to a closed EOFTextChunker")

        self._content += token
        self._new_token_event.set()

    def close(self) -> None:
        """Close the chunker, preventing further token additions."""
        self._is_closed = True
        self._new_token_event.set()

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        while True:
            if self._is_closed:
                # Yield all content once at the end
                if self._content:
                    yield self._content
                return

            # Wait for more content or close signal
            self._new_token_event.clear()
            await self._new_token_event.wait()
