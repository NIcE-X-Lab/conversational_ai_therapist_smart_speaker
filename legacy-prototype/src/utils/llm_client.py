import os
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
from src.utils.log_util import get_logger

logger = get_logger("LLMClient")

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
client = OpenAI(api_key=_api_key, base_url=OPENAI_BASE_URL)


def llm_complete(system_content: str, user_content: str) -> str:
    """
    Unified LLM caller used across the app.
    Inputs:
      - system_content: system prompt/instructions
      - user_content: user input/payload
    Output:
      - plain text content returned by the model
    """
    logger.info("Sending request to LLM")
    logger.debug({"model": OPENAI_MODEL, "user": user_content})
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            instructions=system_content,
            input=user_content,
        )
        logger.info("Received response from LLM (client.responses)")
        return resp.output_text
    except AttributeError:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE,
        )
        logger.info("Received response from LLM (client.chat.completions)")
        return resp.choices[0].message.content


__all__ = ["llm_complete"]


