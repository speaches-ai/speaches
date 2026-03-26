!!! note

    Before proceeding, you should be familiar with the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) and the relevant [OpenAI API reference](https://platform.openai.com/docs/api-reference/realtime-client-events)

!!! warning

    Real-time performance can only be achieved when using CUDA for TTS and STT inference and an LLM provider with a high TPS (tokens per second) rate and low TTFT (time to first token).

## Demo

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc" type="video/webm">
</video>

(Excuse the breathing lol. Didn't have enough time to record a better demo)

## Prerequisites

Follow the prerequisites in the [voice chat](./voice-chat.md) guide.

## Architecture

The Speaches Realtime API provides an OpenAI-compatible WebSocket interface for real-time audio processing, supporting both conversational AI and transcription-only modes.

## OpenAI Compatibility and Extensions

Speaches implements the OpenAI Realtime API specification with some extensions for enhanced usability, particularly for transcription-only scenarios.

### Key Differences from OpenAI

| Feature | OpenAI Realtime API | Speaches Implementation |
|---------|-------------------|------------------------|
<!-- Verified by: tests/realtime_api_test.py::TestRealtimeWebSocketAuthentication::test_websocket_auth_with_bearer_token, test_websocket_auth_with_x_api_key, test_websocket_auth_with_query_param -->
| **Authentication** | ✅ Standard HTTP: `Authorization: Bearer your-key` | ✅ Compatible: `Authorization: Bearer your-key`, `X-API-Key: your-key`, or `api_key` query param |
<!-- Verified by: tests/realtime_api_test.py::TestRealtimeSessionConfiguration::test_transcription_only_mode -->
| **Transcription-only mode** | ✅ Supported: `intent=transcription` parameter | ✅ Fully compatible |
<!-- Verified by: tests/realtime_api_test.py::TestRealtimeAPICompatibility::test_speaches_extension_behavior -->
| **Model parameter behavior** | Always conversation model | **Extension**: In transcription mode, `model` = transcription model |
<!-- Verified by: tests/realtime_api_test.py::TestRealtimeSessionConfiguration::test_transcription_mode_with_language, test_transcription_mode_with_explicit_models -->
| **Additional parameters** | Standard OpenAI params only | **Extension**: `language`, `transcription_model` parameters |
| **Additional headers** | Requires `OpenAI-Beta: realtime=v1` header | No additional headers required |
| **Dynamic transcription model** | Full support via `session.update` | ⚠️ **Limitation**: Changes apply only to new audio buffers |
| **Dynamic speech synthesis model** | Full support via `session.update` | ✅ Supported via `session.update` |

## Operating Modes

Speaches supports two primary operating modes, both compatible with OpenAI Realtime API:

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeSessionConfiguration::test_conversation_mode_default -->
### 1. Conversation Mode (Default)

**Full interactive AI conversation with both speech input and output**

**Event Flow:**
1. `input_audio_buffer.speech_started` → User starts speaking
2. `input_audio_buffer.speech_stopped` → User stops speaking  
3. `input_audio_buffer.committed` → Audio buffer processed
4. `conversation.item.created` → Conversation item created
5. `conversation.item.input_audio_transcription.completed` → Speech transcribed to text
6. `response.created` → **AI response generation begins**
7. Response generation events → AI generates text and audio response

**Use Cases:** Interactive voice assistants, conversational AI, voice chat applications

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeSessionConfiguration::test_transcription_only_mode -->
### 2. Transcription-Only Mode

**Speech-to-text conversion without AI responses**

**Event Flow:**
1. `input_audio_buffer.speech_started` → User starts speaking
2. `input_audio_buffer.speech_stopped` → User stops speaking
3. `input_audio_buffer.committed` → Audio buffer processed  
4. `conversation.item.created` → Conversation item created
5. `conversation.item.input_audio_transcription.completed` → **Stops here - no response generation**

**Use Cases:** Live subtitles, meeting transcription, voice notes, accessibility applications

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeAPICompatibility::test_openai_standard_behavior -->
### Standard OpenAI Behavior

When using `intent=conversation` (default), Speaches follows the OpenAI Realtime API specification exactly:

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeAPICompatibility::test_session_structure_compatibility -->
#### Model Parameters

- **URL `model` parameter**: Specifies the conversation model (e.g., `gpt-4o-realtime-preview`)
- **`input_audio_transcription.model`**: Specifies the transcription model (e.g., `whisper-1`)

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeWebSocketEndpoint::test_websocket_endpoint_exists -->
```javascript
// Standard OpenAI-compatible usage
const ws = new WebSocket("wss://your-speaches-server/v1/realtime?model=gpt-4o-realtime-preview", {
  headers: {
    'Authorization': 'Bearer your-api-key'
    // Note: OpenAI also requires 'OpenAI-Beta': 'realtime=v1' header
  }
});

// Session configuration (OpenAI standard)
// Note: Speaches supports session.update, but transcription model changes
// only apply to new audio buffers, not currently processing ones
ws.send(JSON.stringify({
  type: "session.update",
  session: {
    input_audio_transcription: {
      model: "whisper-1"
    }
  }
}));
```

#### Event Flow

1. Audio input triggers transcription via `input_audio_transcription.model`
2. Transcription completion automatically triggers response generation via conversation `model`
3. Response includes both text and audio output

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeAPICompatibility::test_default_models_configuration -->
### Default Models

When models are not explicitly specified, Speaches uses these defaults:

- **Transcription model**: `Systran/faster-distil-whisper-small.en`
- **Speech synthesis model**: `speaches-ai/Kokoro-82M-v1.0-ONNX` 
- **Voice**: `af_heart`

**Note**: Default models must be available/downloaded, or session will fail. For transcription-only mode without specifying a model, ensure the default transcription model is installed.

### Speaches Extensions

<!-- Verified by: tests/realtime_api_test.py::TestRealtimeAPICompatibility::test_speaches_extension_behavior -->
#### Transcription-Only Mode

For transcription-only scenarios (common with .NET OpenAI SDK and simple clients), Speaches provides an extension:

```javascript
// Transcription-only mode (Speaches extension)
const ws = new WebSocket(
  "wss://your-speaches-server/v1/realtime?model=deepdml/faster-whisper-large-v3-turbo-ct2&intent=transcription&api_key=your-api-key"
);

// Or with headers
const ws = new WebSocket("wss://your-speaches-server/v1/realtime?model=deepdml/faster-whisper-large-v3-turbo-ct2&intent=transcription", {
  headers: {
    'Authorization': 'Bearer your-api-key'
  }
});
```

**Key differences in transcription mode:**
- **URL `model` parameter**: Specifies the transcription model (not conversation model)
- **Response generation**: Disabled (`create_response=false`)
- **Conversation model**: Uses default `gpt-4o-realtime-preview` (unused)

#### Additional Parameters

Speaches supports additional URL parameters for enhanced flexibility:

```
/v1/realtime?model=your-model&intent=transcription&language=en&transcription_model=whisper-1
```

- **`intent`**: `"conversation"` (default) or `"transcription"`
- **`language`**: ISO-639-1 language code for transcription (e.g., `"en"`, `"ru"`)
- **`transcription_model`**: Explicit transcription model (overrides model parameter logic)

### Usage Examples

#### .NET OpenAI SDK - Transcription Only

```csharp
using OpenAI.Realtime;
using OpenAI;

var options = new OpenAIClientOptions();
options.Endpoint = new Uri("http://speaches:8000/v1");
var apiKey = "sk-233dadawd"; // your optional speaches API key
var openAiClient = new OpenAIClient(new System.ClientModel.ApiKeyCredential(apiKey), options);
var realtimeClient = openAiClient.GetRealtimeClient();

var cancellationTokenSource = new CancellationTokenSource();
using var session = await realtimeClient.StartSessionAsync("your-transcription-model", "transcription", new RequestOptions()
{
    CancellationToken = cancellationTokenSource.Token,
});

var transcriptionText = new StringBuilder();
await foreach (RealtimeUpdate update in _session.ReceiveUpdatesAsync(cancellationToken))
{
    switch (update)
    {
        case InputAudioTranscriptionDeltaUpdate transcriptionDelta:
            transcriptionText.Append(transcriptionDelta.Delta);
            Console.Write(transcriptionDelta.Delta);
            break;
        // Handle other events as needed...
    }
}
```

#### JavaScript - Simple Transcription

```javascript
const ws = new WebSocket("wss://speaches-server/v1/realtime?model=your-transcription-model&intent=transcription&api_key=your-api-key");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'conversation.item.input_audio_transcription.completed') {
        console.log('Transcription:', data.transcript);
    }
};
```

## Limitations

- You'll want to be using a dedicated microphone to ensure speech produced by the TTS model is not picked up. Otherwise, the VAD and STT model will pick up the TTS audio and transcribe it, resulting in a feedback loop.
<!-- Verified by: tests/test_interruption.py::test_speech_started_generating_barge_in_immediate, test_response_cancel_with_active_response -->
<!-- Verified by: tests/test_interruption.py::test_truncate_nonexistent_item, test_truncate_assistant_audio_message -->
- Interruption handling is basic: when the user starts speaking while the assistant is generating a response, the response is cancelled via ["response.cancel"](https://platform.openai.com/docs/api-reference/realtime-client-events/response/cancel). However, truncation of audio that has already been played back is not yet supported — ["conversation.item.truncate"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/truncate) is not implemented.
<!-- Verified by: tests/test_doc_claims.py::test_conversation_item_input_audio_has_no_audio_data_field -->
- ["conversation.item.create"](https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create) with `content` field containing `input_audio` message is not supported

## Next Steps

- Image support
- Speech-to-speech model support
- Performance tuning / optimizations
- [Realtime console](https://github.com/speaches-ai/realtime-console) improvements
