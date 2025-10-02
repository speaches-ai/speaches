Before you can do anything useful with `speaches`, you'll need to want to download a machine learning model for your specific task. You can see all the supported models by querying the `/v1/registry` endpoint. This will return a list of all available models, including their IDs, types, and other metadata.

!!! note

    You'll want to make sure you have the `SPEACHES_BASE_URL` environment variable set to the URL of your `speaches` server. If you're running it locally, without modifying the default port, this will be `http://localhost:8000`.

=== "Speaches CLI"

    ```bash
    uvx speaches-cli registry ls
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/registry"
    ```

The above command will display a list of all available models. You can filter the models by task type, such as `automatic-speech-recognition`, `text-to-speech`, etc. To do this, you'll need to specify the `task` query parameter.

=== "Speaches CLI"

    ```bash
    uvx speaches-cli registry ls --task automatic-speech-recognition
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/registry?task=automatic-speech-recognition"
    ```

TIP: You may further filter the response by piping the output to `jq` and using the `select` function.

You'll then want to download the model you want to use. You can do this by making a `POST /v1/models` request.

=== "Speaches CLI"

    ```bash
    uvx speaches-cli model download Systran/faster-distil-whisper-small.en
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/models/Systran/faster-distil-whisper-small.en" -X POST
    ```

The downloaded model will now be included in the list of available models when you query the `/v1/models` endpoint (OpenAI compatible).

=== "Speaches CLI"

    ```bash
    uvx speaches-cli model ls
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/models"
    ```

## Model Aliasing

`speaches` supports model aliasing, which allows you to use friendly names for models instead of their full repository paths. This is particularly useful for maintaining compatibility with OpenAI's API naming conventions.

Model aliases are defined in the `model_aliases.json` file in the root directory of your `speaches` installation. The file contains a JSON object mapping alias names to actual model IDs:

```json
{
  "tts-1": "speaches-ai/Kokoro-82M-v1.0-ONNX",
  "tts-1-hd": "speaches-ai/Kokoro-82M-v1.0-ONNX",
  "whisper-1": "Systran/faster-whisper-large-v3"
}
```

### Using Model Aliases

Once configured, you can use the alias name anywhere you would normally use a full model ID:

=== "Speaches CLI"

    ```bash
    # Use alias instead of full model path
    uvx speaches-cli model download whisper-1

    # This is equivalent to:
    uvx speaches-cli model download Systran/faster-whisper-large-v3
    ```

=== "cURL"

    ```bash
    # Use alias in API requests
    curl "$SPEACHES_BASE_URL/v1/models/whisper-1" -X POST

    # This is equivalent to:
    curl "$SPEACHES_BASE_URL/v1/models/Systran/faster-whisper-large-v3" -X POST
    ```

### Configuring Model Aliases

To add or modify model aliases:

1. **Edit the `model_aliases.json` file** in your `speaches` root directory
2. **Add or modify entries** using the format `"alias_name": "actual_model_id"`
3. **Restart the server** for changes to take effect

**Example:**

```json
{
  "my-whisper": "openai/whisper-large-v3",
  "fast-tts": "speaches-ai/Kokoro-82M-v1.0-ONNX"
}
```

!!! note

    Model aliases are loaded only once when the server starts. You must restart the server after modifying the `model_aliases.json` file.

#### Docker Deployment

If you're using a Docker deployment, you'll need to bind mount your local `model_aliases.json` file to the container:

```bash
# Mount your local model_aliases.json file
docker run -v /path/to/your/model_aliases.json:/home/ubuntu/speaches/model_aliases.json speaches
```

Or in your `compose.yaml`:

```yaml
services:
  speaches:
    volumes:
      - ./model_aliases.json:/home/ubuntu/speaches/model_aliases.json
```

## FAQ

???+ question "Where do the models go?"

    The models are downloaded into `~/.cache/huggingface/hub` by default. This is the default cache directory for Hugging Face models. You can change this by setting the `HF_HUB_CACHE` environment variable to a different directory. If you are using Docker, you can mount a volume to this directory to persist the models across container restarts (the provided Docker Compose files do this).

???+ question "Can I download the models directly using the `huggingface-cli` command?"

    Theoretically, you could in some cases, but it's not recommended. The `speaches` server manages the models and their versions. If you download the models directly, you may run into issues with model compatibility or versioning. It's best to use the `speaches` CLI or API to download and manage the models.

???+ question "How can I add a new model to the registry?"

    Model discovery works by querying the Hugging Face Hub with specific tags. If you want to add a new model to the registry, you can do so by adding the appropriate tags to your model on Hugging Face and ensuring your Hugging Face repository follows the expected structrure (you'll need to take a look at `src/speaches/executors` to find out what the expected tags and repository structrure is). The server will automatically pick up the new model when it queries the registry.

???+ question "Can I use private models from Hugging Face?"

    Yes. You'll want to first set the `HF_TOKEN`(you may need to restart the server after setting this variable) environment variable to your Hugging Face token. This will allow the server to access private models.

???+ question "How do I use Pyannote speaker diarization models?"

    Pyannote models require additional setup because they are gated models on Hugging Face:

    1. **Set your Hugging Face token**: You must set the `HF_TOKEN` environment variable with a valid Hugging Face access token:
       ```bash
       export HF_TOKEN="your_huggingface_token_here"
       ```

    2. **Accept the model EULAs**: Visit the following model pages on Hugging Face and accept the terms of use for each gated model:
       - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
       - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

    3. **Restart the server**: After setting the token and accepting the terms, restart the `speaches` server for the changes to take effect.

    Without these steps, you'll receive authentication errors when attempting to download or use Pyannote models.
