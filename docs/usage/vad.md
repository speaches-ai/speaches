!!! note

## Intro

Unlike other API features. The VAD API isn't OpenAI compatible, as OpenAI doesn't provide a VAD API. Therefore, you cannot use OpenAI SDKs to access this API; You'll need to use an HTTP client like `httpx`(Python), `requests`(Python), `reqwest`(Rust), etc.

There's only 1 supported model for VAD, which is `silero_vad_v5`. This model is packaged in one of the dependencies, so you don't need to download it separately. Because of this, you won't see it when querying local models or listing models from model registry.

Refer to the [../api.md] for additional details such as supported request parameters and response format.

## Usage

```sh
export SPEACHES_BASE_URL="http://localhost:8000"


curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"
# [{"start":64,"end":1323}]


curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2"
# [{"start":64,"end":256},{"start":288,"end":480},{"start":512,"end":704},{"start":800,"end":992},{"start":1024,"end":1216}]

curl "$SPEACHES_BASE_URL/v1/audio/speech/timestamps" -F "file=@audio.wav"  -F "max_speech_duration_s=0.2" -F "threshold=0.99"
# [{"start":96,"end":288},{"start":320,"end":512},{"start":544,"end":736},{"start":832,"end":1024},{"start":1056,"end":1248}]
```
