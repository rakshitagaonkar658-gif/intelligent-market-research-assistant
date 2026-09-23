"""
utils/config.py
───────────────
Central configuration loader for the Intelligent Market Research Assistant.

All settings are read from environment variables (loaded from .env via python-dotenv).
No credentials are ever hard-coded here.

Usage
-----
    from utils.config import config

    if config.is_demo():
        ...  # use mock / sample data
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (the directory that contains this utils/ package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass
class _Config:
    """Immutable application configuration loaded from environment variables."""

    # ── watsonx.ai ────────────────────────────────────────────────────────────
    watsonx_api_key: str = field(default_factory=lambda: os.getenv("WATSONX_API_KEY", ""))
    watsonx_project_id: str = field(default_factory=lambda: os.getenv("WATSONX_PROJECT_ID", ""))
    watsonx_url: str = field(
        default_factory=lambda: os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    )

    # ── IBM Granite model ─────────────────────────────────────────────────────
    granite_model_id: str = field(
        default_factory=lambda: os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct")
    )

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))

    # ── Demo mode flag (explicit override) ───────────────────────────────────
    _demo_mode_env: str = field(
        default_factory=lambda: os.getenv("DEMO_MODE", "false"), repr=False
    )

    # ── Derived paths ─────────────────────────────────────────────────────────
    project_root: Path = field(default=_PROJECT_ROOT, repr=False)
    data_dir: Path = field(default=_PROJECT_ROOT / "data", repr=False)
    chroma_dir: Path = field(default=_PROJECT_ROOT / "data" / "chroma_db", repr=False)
    exports_dir: Path = field(default=_PROJECT_ROOT / "data" / "exports", repr=False)
    sample_reports_dir: Path = field(
        default=_PROJECT_ROOT / "data" / "sample_reports", repr=False
    )

    def __post_init__(self) -> None:
        # Ensure output directories exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # Emit a clear warning when running without credentials
        if self.is_demo():
            logger.warning(
                "⚠️  Demo mode active — IBM Granite credentials not detected. "
                "The application will use MockLLM and bundled sample data. "
                "To enable live mode, set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file."
            )

    def is_demo(self) -> bool:
        """Return True when no live watsonx credentials are available or DEMO_MODE=true."""
        explicit_demo = self._demo_mode_env.strip().lower() in ("true", "1", "yes")
        missing_creds = not self.watsonx_api_key.strip() or not self.watsonx_project_id.strip()
        return explicit_demo or missing_creds

    def has_newsapi(self) -> bool:
        """Return True when a NewsAPI key is configured."""
        return bool(self.newsapi_key.strip())

    def granite_params(self) -> dict:
        """Return the LLM parameters dict for the Granite model."""
        return {
            "decoding_method": "greedy",
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

    def __repr__(self) -> str:  # never expose the API key
        return (
            f"Config(model={self.granite_model_id!r}, "
            f"demo={self.is_demo()}, "
            f"newsapi={'set' if self.has_newsapi() else 'unset'}, "
            f"url={self.watsonx_url!r})"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
config = _Config()
