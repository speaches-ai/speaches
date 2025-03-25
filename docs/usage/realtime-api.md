!!! note

    Before proceeding, you should be familiar with the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/realtime-client-events)

## Demo

TODO

!!! warning

    Real-time performance can only be achieved when using CUDA for TTS and STT inference and an LLM model which has a high TPS (tokens per second) rate and low latency.

## Prerequisites

Follow the prerequisites in the [voice chat](./voice-chat.md.md) guide. And set the following environmental variables:

## Architecture

TODO

## Limitations

- ["response.cancel"](https://platform.openai.com/docs/api-reference/realtime-client-events/response/cancel) and ["conversation.item.truncate"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/truncate) client events are not supported
- ["conversation.item.create"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create) with `content` field containing `input_audio` message is not supported
- Interruption handling needs to be flushed out

## Next Steps

- Address the aforementioned limitations
- Image support
- Speech-to-speech model support
- Performance optimizations
