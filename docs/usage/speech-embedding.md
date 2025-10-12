# Speech Embedding Extraction

Speech embedding extraction allows you to extract high-dimensional vector representations of speech audio. These embeddings can be used for speaker verification, speaker identification, and voice similarity comparison tasks.

!!! note

    This feature uses ONNX models from the PyAnnote ecosystem for speaker embedding extraction. The embeddings are 512-dimensional vectors that capture speaker-specific characteristics.

## Download a Speech Embedding Model

```bash
export SPEACHES_BASE_URL="http://localhost:8000"

# Listing all available speech embedding models
uvx speaches-cli registry ls --task speaker-embedding | jq '.data | [].id'

# Downloading a model
uvx speaches-cli model download deepghs/pyannote-embedding-onnx

# Check that the model has been installed
uvx speaches-cli model ls --task speaker-embedding | jq '.data | map(select(.id == "deepghs/pyannote-embedding-onnx"))'
```

## Usage

### Curl

```bash
export SPEACHES_BASE_URL="http://localhost:8000"
export EMBEDDING_MODEL_ID="deepghs/pyannote-embedding-onnx"

curl -s "$SPEACHES_BASE_URL/v1/audio/speech/embedding" \
  -F "file=@audio.wav" \
  -F "model=$EMBEDDING_MODEL_ID"
```

### Python

=== "httpx"

    ```python
    import httpx

    with open('audio.wav', 'rb') as f:
        files = {'file': ('audio.wav', f)}
        data = {'model': 'deepghs/pyannote-embedding-onnx'}
        response = httpx.post(
            'http://localhost:8000/v1/audio/speech/embedding',
            files=files,
            data=data
        )

    result = response.json()
    embedding = result['data'][0]['embedding']
    print(f"Embedding dimension: {len(embedding)}")
    ```

=== "requests"

    ```python
    import requests

    with open('audio.wav', 'rb') as f:
        files = {'file': ('audio.wav', f)}
        data = {'model': 'deepghs/pyannote-embedding-onnx'}
        response = requests.post(
            'http://localhost:8000/v1/audio/speech/embedding',
            files=files,
            data=data
        )

    result = response.json()
    embedding = result['data'][0]['embedding']
    print(f"Embedding dimension: {len(embedding)}")
    ```

## Comparing Speech Embeddings

Speech embeddings can be compared using cosine similarity to measure how similar two voice samples are. This is useful for speaker verification tasks.

### Example: Voice Similarity Comparison

```python
import httpx
import numpy as np

def get_embedding(audio_path: str, model_id: str) -> list[float]:
    with open(audio_path, 'rb') as f:
        files = {'file': (audio_path, f)}
        data = {'model': model_id}
        response = httpx.post(
            'http://localhost:8000/v1/audio/speech/embedding',
            files=files,
            data=data
        )
    result = response.json()
    return result['data'][0]['embedding']

def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model_id = 'deepghs/pyannote-embedding-onnx'

embedding1 = get_embedding('speaker1.wav', model_id)
embedding2 = get_embedding('speaker2.wav', model_id)

similarity = cosine_similarity(embedding1, embedding2)
print(f"Cosine similarity: {similarity:.4f}")

if similarity > 0.8:
    print("High similarity - likely the same speaker")
elif similarity > 0.5:
    print("Moderate similarity - possibly the same speaker")
else:
    print("Low similarity - likely different speakers")
```

### Example: Speaker Verification

```python
import httpx
import numpy as np
from pathlib import Path

class SpeakerVerifier:
    def __init__(self, base_url: str, model_id: str, threshold: float = 0.7):
        self.base_url = base_url
        self.model_id = model_id
        self.threshold = threshold

    def get_embedding(self, audio_path: str | Path) -> np.ndarray:
        with open(audio_path, 'rb') as f:
            files = {'file': (str(audio_path), f)}
            data = {'model': self.model_id}
            response = httpx.post(
                f'{self.base_url}/v1/audio/speech/embedding',
                files=files,
                data=data
            )
        result = response.json()
        return np.array(result['data'][0]['embedding'])

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def verify(self, enrollment_audio: str | Path, test_audio: str | Path) -> tuple[bool, float]:
        enrollment_embedding = self.get_embedding(enrollment_audio)
        test_embedding = self.get_embedding(test_audio)
        similarity = self.cosine_similarity(enrollment_embedding, test_embedding)
        is_same_speaker = similarity > self.threshold
        return is_same_speaker, similarity

verifier = SpeakerVerifier(
    base_url='http://localhost:8000',
    model_id='deepghs/pyannote-embedding-onnx',
    threshold=0.7
)

is_same, score = verifier.verify('enrollment.wav', 'test.wav')
print(f"Same speaker: {is_same}, Similarity score: {score:.4f}")
```

## Response Format

The response follows a structure similar to OpenAI's text embedding endpoint:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        -4.547132266452536e-5,
        -0.024047505110502243,
        ...
      ]
    }
  ],
  "model": "deepghs/pyannote-embedding-onnx",
  "usage": {
    "prompt_tokens": 48000,
    "total_tokens": 48000
  }
}
```

## Use Cases

1. **Speaker Verification**: Verify if a voice sample matches an enrolled speaker
2. **Speaker Identification**: Identify which speaker from a set of known speakers is speaking
3. **Voice Clustering**: Group audio samples by speaker
4. **Voice Search**: Find audio samples containing a specific speaker's voice
5. **Voice Biometrics**: Use voice as a biometric authentication factor

## Tips

- For best results, use audio samples that are at least 1-2 seconds long
- The model expects 16kHz audio, but the API handles resampling automatically
- Remove background noise when possible for more accurate embeddings
- Use multiple enrollment samples per speaker for more robust verification
