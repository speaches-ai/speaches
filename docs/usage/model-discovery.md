Before you can do anything useful with `speaches`, you'll need to want to download a machine learning model for your specific task. You can see all the supported models by querying the `/v1/registry` endpoint. This will return a list of all available models, including their IDs, types, and other metadata.

!!! note

    You'll want to make sure you have the `SPEACHES_BASE_URL` environment variable set to the URL of your `speaches` server. If you're running it locally, without modifying the default port, this will be `http://localhost:8000`.

=== "Speaches CLI"

    ```bash
    uvx speaches registry ls
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/registry"
    ```

The above command will display a list of all available models. You can filter the models by task type, such as `automatic-speech-recognition`, `text-to-speech`, etc. To do this, you'll need to specify the `task` query parameter.

=== "Speaches CLI"

    ```bash
    uvx speaches registry ls --task automatic-speech-recognition
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/registry?task=automatic-speech-recognition"
    ```

TIP: You may further filter the response by piping the output to `jq` and using the `select` function.

You'll then want to download the model you want to use. You can do this by making a `POST /v1/models` request.

=== "Speaches CLI"

    ```bash
    uvx speaches model download Systran/faster-distil-whisper-small.en
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/models/Systran/faster-distil-whisper-small.en" -X POST
    ```

The downloaded model will now be included in the list of available models when you query the `/v1/models` endpoint (OpenAI compatible).

=== "Speaches CLI"

    ```bash
    uvx speaches model ls
    ```

=== "cURL"

    ```bash
    curl "$SPEACHES_BASE_URL/v1/models"
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
