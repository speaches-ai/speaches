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

    async def __aiter__(self) -> AsyncGenerator[str]:
        """Iterate through chunks of text.
        Different implementations may chunk text differently.
        """
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

    async def __aiter__(self) -> AsyncGenerator[str]:
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


def strip_emojis(text: str) -> str:
    # Get all emoji unicode characters
    emoji_pattern = re.compile(
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
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub(r"", text)


def strip_markdown_emphasis(text: str) -> str:
    """Remove markdown emphasis markers from text.

    Examples:
        - "Hello my name is **Jon**" -> "Hello my name is Jon"
        - "I *really* like this" -> "I really like this"
        - "This is __underlined__" -> "This is underlined"
        - "This is _italic_" -> "This is italic"

    """
    # Remove bold (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # Remove italic (*text*)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # Remove underlined (__text__)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # Remove italic with underscore (_text_)
    text = re.sub(r"_(.*?)_", r"\1", text)

    return text


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

    async def __aiter__(self) -> AsyncGenerator[str]:
        while True:
            if self._is_closed:
                # Yield all content once at the end
                if self._content:
                    yield self._content
                return

            # Wait for more content or close signal
            self._new_token_event.clear()
            await self._new_token_event.wait()
