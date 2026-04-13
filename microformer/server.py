"""FastAPI inference server for MicroFormer.

Run with::

    MICROFORMER_CHECKPOINT=checkpoints/latest.pt \\
    MICROFORMER_TOKENIZER=tokenizer.json \\
    uvicorn microformer.server:app
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from microformer.config import ModelConfig
from microformer.kv_cache import generate_cached
from microformer.model import MicroFormer
from microformer.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    prompt: str | list[str]
    max_tokens: int = Field(default=50, gt=0, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Prompt must not be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Prompt list must not be empty")
            for p in v:
                if not p.strip():
                    raise ValueError("Each prompt must not be empty")
        return v


class Completion(BaseModel):
    """Single generated completion."""

    text: str
    tokens_generated: int


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate``."""

    completions: list[Completion]


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    device: str
    uptime_seconds: float
    vocab_size: int
    n_parameters: int
    max_seq_len: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    model: MicroFormer | None = None,
    tokenizer: BPETokenizer | None = None,
    device: str = "cpu",
) -> FastAPI:
    """Create the FastAPI application.

    For production, leave *model* and *tokenizer* as ``None`` — they will
    be loaded from ``MICROFORMER_CHECKPOINT`` / ``MICROFORMER_TOKENIZER``
    environment variables at startup.

    For testing, pass pre-built instances directly.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load model from env vars if not pre-configured (e.g. in tests).
        if app.state.model is None:
            ckpt_path = os.environ.get("MICROFORMER_CHECKPOINT")
            if not ckpt_path:
                raise RuntimeError("Set MICROFORMER_CHECKPOINT to a checkpoint path")

            logger.info("Loading checkpoint from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ModelConfig(**ckpt["model_config"])
            loaded_model = MicroFormer(cfg)
            loaded_model.load_state_dict(ckpt["model_state_dict"])
            loaded_model.to(device)
            loaded_model.eval()
            app.state.model = loaded_model

            tok_path = os.environ.get("MICROFORMER_TOKENIZER")
            if tok_path:
                app.state.tokenizer = BPETokenizer.load(tok_path)
                logger.info("Loaded tokenizer from %s", tok_path)

        yield  # Application runs here.

    application = FastAPI(title="MicroFormer Inference Server", lifespan=_lifespan)
    application.state.model = model
    application.state.tokenizer = tokenizer
    application.state.device = device
    application.state.start_time = time.time()

    # ---- Endpoints --------------------------------------------------------

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        m: MicroFormer | None = application.state.model
        if m is None:
            raise HTTPException(503, detail="Model not loaded")
        return HealthResponse(
            status="ok",
            device=application.state.device,
            uptime_seconds=time.time() - application.state.start_time,
            vocab_size=m.config.vocab_size,
            n_parameters=m.count_parameters(),
            max_seq_len=m.config.max_seq_len,
        )

    @application.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        m: MicroFormer | None = application.state.model
        tok: BPETokenizer | None = application.state.tokenizer
        if m is None:
            raise HTTPException(503, detail="Model not loaded")
        if tok is None:
            raise HTTPException(503, detail="Tokenizer not loaded")

        prompts = (
            request.prompt if isinstance(request.prompt, list) else [request.prompt]
        )

        completions = await asyncio.gather(
            *[
                _generate_one(m, tok, p, request, application.state.device)
                for p in prompts
            ]
        )
        return GenerateResponse(completions=list(completions))

    return application


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _generate_one(
    model: MicroFormer,
    tokenizer: BPETokenizer,
    prompt: str,
    request: GenerateRequest,
    device: str,
) -> Completion:
    """Generate for a single prompt, off-loading to a worker thread."""

    def _run() -> Completion:
        token_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        output = generate_cached(
            model,
            prompt_tensor,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        generated_ids = output[0, len(token_ids) :].tolist()
        text = tokenizer.decode(generated_ids)
        return Completion(text=text, tokens_generated=len(generated_ids))

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Default app instance for ``uvicorn microformer.server:app``
# ---------------------------------------------------------------------------

app = create_app()
