!!! note

    Before proceeding, you should be familiar with the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/realtime-client-events)

!!! warning

    Real-time performance can only be achieved when using CUDA for TTS and STT inference and an LLM provider with a high TPS (tokens per second) rate and low TTFT (time to first token).

## Demo

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/ee9220c4-be6c-4728-a25c-3bddfd66ab34" type="video/webm">
</video>

(Excuse the breathing lol. Didn't have enough time to record a better demo)

## Prerequisites

Follow the prerequisites in the [voice chat](./voice-chat.md.md) guide.

## Architecture

TODO

## Limitations

- You'll want to be using a dedicated microphone to ensure speech produced by the TTS model is not picked up. Otherwise, the VAD and STT model will pick up the TTS audio and transcribe it, resulting in a feedback loop.
- ["response.cancel"](https://platform.openai.com/docs/api-reference/realtime-client-events/response/cancel) and ["conversation.item.truncate"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/truncate) client events are not supported. Interruption handling needs to be flushed out.
- ["conversation.item.create"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create) with `content` field containing `input_audio` message is not supported

## Next Steps

- Address the aforementioned limitations
- Image support
- Speech-to-speech model support
- Performance tuning / optimizations
- [Realtime console](https://github.com/speaches-ai/realtime-console) improvements
