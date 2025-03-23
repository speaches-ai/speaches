import asyncio
from pathlib import Path

from httpx import AsyncClient
import huggingface_hub

# each model should have: model.onnx, config.json, a README.md

# https://huggingface.co/docs/datasets/en/dataset_card

# TODO: add details to to the README.md: dataset links, author attributions, etc.
README_TEMPLATE = """
---
{model_card_data_formated}
---

Run this model using [speaches](https://github.com/speaches-ai/speaches)

WARN: the structure of this model repository is experimental and **will** change in the near future.
""".strip()


MODEL_NAME_ONNX_FILE_MAP = {
    "Kokoro-82M-v1.0-ONNX": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "Kokoro-82M-v1.0-ONNX-fp16": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx",
    "Kokoro-82M-v1.0-ONNX-int8": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
}
VOICES_FILE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


client = AsyncClient()


async def download_from_github_to_file(url: str, file_path: Path) -> None:
    res = await client.get(url, follow_redirects=True)
    res = res.raise_for_status()  # HACK
    file_path.touch(exist_ok=True)
    file_path.write_bytes(res.content)


async def create_local_model_repo(repo_id: str, model_url: str) -> None:
    repo_path = Path(repo_id)
    repo_path.mkdir(parents=True, exist_ok=True)

    await download_from_github_to_file(
        model_url,
        repo_path / "model.onnx",
    )

    await download_from_github_to_file(
        VOICES_FILE,
        repo_path / "voices.bin",
    )

    model_card_data = huggingface_hub.ModelCardData(
        # license="MIT",
        library="onnx",
        pipeline_tag="text-to-speech",
        tags=["speaches", "kokoro"],  # TODO
        language="multilingual",  # TODO
    )

    content = README_TEMPLATE.format(model_card_data_formated=model_card_data.to_yaml())

    model_card = huggingface_hub.ModelCard(content, ignore_metadata_errors=False)

    repo_path.mkdir(parents=True, exist_ok=True)
    model_card.save(repo_path / "README.md")


async def main() -> None:
    for model_name, model_url in MODEL_NAME_ONNX_FILE_MAP.items():
        repo_id = f"speaches-ai/{model_name}"
        print(f"Creating repo {repo_id}...")
        # await create_local_model_repo(repo_id, model_url)

        huggingface_hub.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        huggingface_hub.upload_folder(
            repo_id=repo_id,
            folder_path=repo_id,
            commit_message="init",
            repo_type="model",
            create_pr=False,
        )
        print(f"Repo {repo_id} created.")
        break


if __name__ == "__main__":
    asyncio.run(main())
