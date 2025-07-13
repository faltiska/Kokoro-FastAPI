"""
FastAPI OpenAI Compatible API
"""

import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers.debug import router as debug_router
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .routers.web_player import router as web_router


def setup_logger():
    """Configure loguru logger with custom formatting"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "<fg #4169E1>{module}:{line}</fg #4169E1> | "
                "{message}",
                "colorize": True,
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager
    from .services.temp_manager import cleanup_temp_files

    # Clean old temp files on startup
    await cleanup_temp_files()

    logger.info("Loading TTS model and voice packs...")

    try:
        # Initialize managers
        model_manager = await get_manager()

        # Initialize model with warmup and get status
        device, model, load_time = await model_manager.initialize_and_load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    boundary = "░" * 2 * 12
    logger.info(f"""

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │ 
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                \n""")

    startup_msg = f"Model {model} loaded on {device} in {load_time}ms."
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"{startup_msg} CUDA available.\n")
    else:
        logger.error(f"{startup_msg} CUDA unavailable.\n")

    if settings.enable_web_player:
        logger.info(f"Web Player: http://{settings.host}:{settings.port}/web/ or http://localhost:{settings.port}/web/\n")
    else:
        logger.info("Web Player: disabled\n")

    yield


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)

# Add CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # Development endpoints
app.include_router(debug_router)  # Debug endpoints
if settings.enable_web_player:
    app.include_router(web_router, prefix="/web")  # Web player static files


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
