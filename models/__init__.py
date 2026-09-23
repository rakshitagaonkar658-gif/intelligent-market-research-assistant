"""
models/__init__.py
──────────────────
LLM factory — returns the correct LLM based on the active configuration.

Usage
-----
    from models import get_llm
    llm = get_llm()       # WatsonxLLM (Granite) in live mode, MockLLM in demo mode

The returned object is a LangChain-compatible BaseLLM instance and can be
used as a drop-in anywhere LangChain expects an LLM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache so the LLM is only instantiated once per process
_llm_instance: Any = None


def get_llm(force_refresh: bool = False) -> Any:
    """
    Return the application LLM (cached after first call).

    In live mode  → WatsonxLLM backed by IBM Granite via watsonx.ai.
    In demo mode  → MockLLM using bundled sample data (no external calls).

    Parameters
    ----------
    force_refresh:
        If True, discard the cached instance and create a new one.
        Useful in tests or after config changes.
    """
    global _llm_instance

    if _llm_instance is not None and not force_refresh:
        return _llm_instance

    from utils.config import config  # deferred to avoid circular imports

    if config.is_demo():
        logger.info("LLM factory: demo mode active — returning MockLLM")
        from models.mock_llm import MockLLM
        _llm_instance = MockLLM()
    else:
        logger.info(
            "LLM factory: live mode — initialising Granite (%s)", config.granite_model_id
        )
        try:
            from models.granite_llm import get_granite_llm
            _llm_instance = get_granite_llm()
        except RuntimeError as exc:
            logger.warning(
                "Granite initialisation failed (%s) — falling back to MockLLM.\n"
                "Set DEMO_MODE=true to suppress this warning.",
                exc,
            )
            from models.mock_llm import MockLLM
            _llm_instance = MockLLM()

    return _llm_instance


def reset_llm_cache() -> None:
    """Clear the cached LLM instance (used in tests)."""
    global _llm_instance
    _llm_instance = None
