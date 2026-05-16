"""FastAPI server exposing NeMo Guardrails to the chat frontend.

Endpoints
---------
GET  /api/configs           list available config folders
GET  /api/health            sanity check + which provider is active
POST /api/chat              { config, messages } -> { content }

Provider switching
------------------
The LLM provider is chosen by environment variables, so students can flip
between OpenAI and a self-hosted vLLM instance just by editing .env:

    LLM_PROVIDER=openai           # or "vllm"
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini

    VLLM_BASE_URL=http://vllm:8000/v1
    VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
    VLLM_API_KEY=EMPTY

When LLM_PROVIDER=vllm we rewrite the loaded RailsConfig in memory to point
at the vLLM OpenAI-compatible endpoint. The .co files and config.yml on
disk stay untouched, so the same lab works against either backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nemoguardrails import LLMRails, RailsConfig


CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", "/app/configs"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


# --------------------------------------------------------------------------- #
# Provider wiring
# --------------------------------------------------------------------------- #
def _apply_provider(config: RailsConfig) -> RailsConfig:
    """Mutate the loaded config so it uses the provider chosen in .env."""
    if not config.models:
        return config

    main_model = next((m for m in config.models if m.type == "main"), config.models[0])

    if LLM_PROVIDER == "vllm":
        # vLLM ships an OpenAI-compatible /v1 endpoint, so we keep engine="openai"
        # and override the base URL + model id via parameters.
        base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
        model_id = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        main_model.engine = "openai"
        main_model.model = model_id
        main_model.parameters = {
            **(main_model.parameters or {}),
            "openai_api_base": base_url,
            "openai_api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
        }
    else:
        main_model.engine = "openai"
        main_model.model = os.getenv("OPENAI_MODEL", main_model.model or "gpt-4o-mini")
        # OPENAI_API_KEY is read from the env by the OpenAI SDK directly.

    return config


# --------------------------------------------------------------------------- #
# Rails cache (configs are expensive to build, ~seconds per load)
#
# We invalidate the cache when any file under the config dir has a newer
# mtime than the time we cached it. That way students editing .co/.yml files
# can just re-select the config in the sidebar — no container restart needed.
# --------------------------------------------------------------------------- #
_rails_cache: dict[str, tuple[float, LLMRails]] = {}


def _latest_mtime(path: Path) -> float:
    latest = 0.0
    for p in path.rglob("*"):
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    return latest


def get_rails(config_name: str) -> LLMRails:
    config_path = CONFIGS_DIR / config_name
    if not config_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown config: {config_name}")

    mtime = _latest_mtime(config_path)
    cached = _rails_cache.get(config_name)
    if cached and cached[0] >= mtime:
        return cached[1]

    config = RailsConfig.from_path(str(config_path))
    config = _apply_provider(config)
    rails = LLMRails(config)
    _rails_cache[config_name] = (mtime, rails)
    return rails


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    config: str
    messages: list[Message]


class ChatResponse(BaseModel):
    content: str


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="NeMo Guardrails Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": (
            os.getenv("VLLM_MODEL") if LLM_PROVIDER == "vllm"
            else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ),
    }


@app.get("/api/configs")
def list_configs() -> list[dict[str, str]]:
    if not CONFIGS_DIR.is_dir():
        return []
    items = []
    for path in sorted(CONFIGS_DIR.iterdir()):
        if path.is_dir() and (path / "config.yml").exists():
            items.append({"name": path.name, "title": path.name.replace("_", " ")})
    return items


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    rails = get_rails(req.config)
    messages = [m.model_dump() for m in req.messages]

    # All lab configs use Colang 2.0, whose runtime is stateful and rejects
    # `assistant` messages in the input — it expects only the latest user
    # turn and tracks dialog state internally. Multi-turn memory would
    # require passing the runtime `state` object back and forth, which is
    # out of scope for this teaching lab.
    messages = [m for m in messages if m.get("role") == "user"][-1:]

    result = await rails.generate_async(messages=messages)

    # generate_async returns either a dict {"role","content"} or a string
    if isinstance(result, dict):
        content = result.get("content", "")
    else:
        content = str(result)

    return ChatResponse(content=content)
