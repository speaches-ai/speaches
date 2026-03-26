## Intro

Unlike other API features. The VAD API isn't OpenAI compatible, as OpenAI doesn't provide a VAD API. Therefore, you cannot use OpenAI SDKs to access this API; You'll need to use an HTTP client like `httpx`(Python), `requests`(Python), `reqwest`(Rust), etc.

<!-- Verified by: tests/vad_test.py::test_speech_timestamps_basic (v5), tests/test_vad_v6.py::test_vad_v6_basic (v6) -->
There are 2 supported VAD models: `silero_vad_v5` and `silero_vad_v6`. The default is `silero_vad_v6`. These models are packaged in one of the dependencies, so you don't need to download them separately. Because of this, you won't see them when querying local models or listing models from model registry.

<!-- Verified by: src/speaches/config.py::Config.vad_model (config field, default "silero_vad_v6") -->
The active VAD model can be configured via the `VAD_MODEL` environment variable. Accepted values are `silero_vad_v5` and `silero_vad_v6`.

The VAD model TTL (time-to-live) can be configured via the `VAD_MODEL_TTL` environment variable. The behavior matches the STT and TTS TTL settings: `-1` means never unload the model, `0` means unload it immediately after each use.

The VAD is also used internally by the Realtime API for voice activity detection during conversation and transcription sessions.

<!-- Verified by: tests/test_vad_v6.py::test_vad_v6_response_schema -->
Refer to the [../api.md] for additional details such as supported request parameters and response format.

## Usage

<!-- Verified by: tests/test_vad_v6.py::test_vad_v6_silence_duration -->
<!-- Verified by: tests/test_vad_v6.py::test_vad_v6_threshold -->
```sh
export SPEACHES_BASE_URL="http://localhost:8000"


curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"
# [{"start":64,"end":1323}]


curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2"
# [{"start":64,"end":256},{"start":288,"end":480},{"start":512,"end":704},{"start":800,"end":992},{"start":1024,"end":1216}]

curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2" -F "threshold=0.99"
# [{"start":96,"end":288},{"start":320,"end":512},{"start":544,"end":736},{"start":832,"end":1024},{"start":1056,"end":1248}]
```
