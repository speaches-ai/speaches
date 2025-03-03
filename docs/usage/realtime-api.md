!!! note

    Before proceeding, you should be familiar with the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/realtime-client-events)

## Prerequisites

Follow the prerequisites in the [Text-to-Speech](./text-to-speech.md) guide. And set the following environmental variables:

- `CHAT_COMPLETION_BASE_URL` to the base URL of an OpenAI API compatible endpoint | [Config](../configuration.md#speaches.config.Config.chat_completion_base_url)
- `CHAT_COMPLETION_API_KEY` if the API you are using requires authentication | [Config](../configuration.md#speaches.config.Config.chat_completion_api_key)

## Demo

TODO

## Limitations

- ["response.cancel"](https://platform.openai.com/docs/api-reference/realtime-client-events/response/cancel) and ["conversation.item.truncate"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/truncate) client events are not supported
- ["conversation.item.create"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create) with `content` field containing `input_audio` message is not supported

## Architecture

TODO

## Next Steps

- Image support
- Speech-to-speech model support
- Optimizations and feature
