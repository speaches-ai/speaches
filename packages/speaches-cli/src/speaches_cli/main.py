import json
import os
import re
from typing import override

import click
import httpx
import typer
from typer.core import TyperGroup


# Taken from: https://github.com/fastapi/typer/issues/132#issuecomment-2417492805
class AliasGroup(TyperGroup):
    _CMD_SPLIT_P = re.compile(r" ?[,|] ?")  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd_name = self._group_cmd_name(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _group_cmd_name(self, default_name: str) -> str:
        for cmd in self.commands.values():
            name = cmd.name
            if name and default_name in self._CMD_SPLIT_P.split(name):
                return name
        return default_name


app = typer.Typer(cls=AliasGroup)
registry_app = typer.Typer(cls=AliasGroup)
model_app = typer.Typer(cls=AliasGroup)
audio_app = typer.Typer(cls=AliasGroup)
audio_speech_app = typer.Typer(cls=AliasGroup)

SPEACHES_BASE_URL = os.getenv("SPEACHES_BASE_URL", "http://localhost:8000")
SPEACHES_OPENAI_BASE_URL = SPEACHES_BASE_URL + "/v1"
client = httpx.Client(base_url=SPEACHES_BASE_URL, timeout=httpx.Timeout(None))

MODELS_URL = f"{SPEACHES_OPENAI_BASE_URL}/models"
REGISTRY_URL = f"{SPEACHES_OPENAI_BASE_URL}/registry"


def dump_response(response: httpx.Response) -> None:
    if response.headers.get("Content-Type") == "application/json":
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(response.text)


@registry_app.command("list | ls")
def registry_ls(task: str | None = None) -> None:
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    response = client.get(REGISTRY_URL, params=params)
    dump_response(response)


@model_app.command("list | ls")
def models_ls(task: str | None = None) -> None:
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    response = client.get(MODELS_URL, params=params)
    dump_response(response)


@model_app.command("remove | rm")
def model_rm(model_id: str) -> None:
    response = client.delete(f"{MODELS_URL}/{model_id}")
    dump_response(response)


@model_app.command("download")
def model_download(model_id: str) -> None:
    response = client.post(f"{MODELS_URL}/{model_id}")
    dump_response(response)


app.add_typer(registry_app, name="registry")
app.add_typer(model_app, name="model")

if __name__ == "__main__":
    app()
