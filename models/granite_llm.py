"""
models/granite_llm.py
─────────────────────
IBM Granite LLM wrapper via langchain-ibm WatsonxLLM.

All credentials are read exclusively from utils/config.py (which in turn
reads from environment variables / .env).  Nothing is hard-coded here.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_granite_llm() -> Any:
    """
    Instantiate and return a LangChain-compatible WatsonxLLM backed by the
    configured IBM Granite model.

    Raises
    ------
    RuntimeError
        If the langchain-ibm or ibm-watsonx-ai packages are not installed,
        or if the credentials in Config are rejected by the watsonx API.
        The error message always includes a hint to switch to DEMO_MODE.
    """
    try:
        from langchain_ibm import WatsonxLLM  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "langchain-ibm is not installed.  "
            "Run: pip install langchain-ibm ibm-watsonx-ai\n"
            "Or set DEMO_MODE=true in your .env to use the offline MockLLM."
        ) from exc

    from utils.config import config  # local import avoids circular dep at module load

    if not config.watsonx_api_key or not config.watsonx_project_id:
        raise RuntimeError(
            "WATSONX_API_KEY and WATSONX_PROJECT_ID are required for live mode.\n"
            "Set DEMO_MODE=true (or leave WATSONX_API_KEY blank) to use the offline MockLLM."
        )

    params = config.granite_params()

    try:
        llm = WatsonxLLM(
            model_id=config.granite_model_id,
            url=config.watsonx_url,
            apikey=config.watsonx_api_key,
            project_id=config.watsonx_project_id,
            params=params,
        )
        logger.info(
            "Granite LLM initialised — model: %s, url: %s",
            config.granite_model_id,
            config.watsonx_url,
        )
        return llm
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise WatsonxLLM ({config.granite_model_id}): {exc}\n"
            "Check your WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL.\n"
            "Set DEMO_MODE=true to fall back to the offline MockLLM."
        ) from exc
