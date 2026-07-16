"""Shared Anthropic SDK wrapper with retry and token tracking."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


def _find_text_block(message: "anthropic.types.Message") -> "anthropic.types.TextBlock":
    """Return the first text block in a message, or raise ValueError.

    Using a for-loop instead of next(generator) avoids PEP 479: StopIteration
    escaping a generator inside a coroutine is re-raised as RuntimeError.
    """
    for block in message.content:
        if block.type == "text":
            return block
    types = [b.type for b in message.content]
    raise ValueError(f"No text block in API response. Content block types: {types}")


class BaseAgent:
    """Async Anthropic agent with retry, token tracking, and streaming."""

    MODEL = "claude-haiku-4-5"
    MAX_TOKENS = 8192
    MAX_RETRIES = 3
    BASE_DELAY = 1.0

    def __init__(self, system_prompt: str, effort: str = "high") -> None:
        self._client = anthropic.AsyncAnthropic()
        self._system_prompt = system_prompt
        self._effort = effort
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # Models that support extended thinking and output_config/effort
    _THINKING_MODELS = ("claude-opus", "claude-sonnet")

    def _supports_thinking(self) -> bool:
        return any(t in self.MODEL for t in self._THINKING_MODELS)

    async def _call(
        self,
        messages: list[dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
    ) -> anthropic.types.Message:
        """Call the API with retry logic and optional structured output."""
        create_kwargs: dict[str, Any] = dict(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=self._system_prompt,
            messages=messages,
        )

        if self._supports_thinking():
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": self._effort}

        if output_schema:
            if self._supports_thinking():
                create_kwargs["output_config"] = {
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": output_schema},
                }
            else:
                # Haiku: use tool_use for structured output.
                # tool input_schema must be type:object, so wrap arrays in an envelope.
                if output_schema.get("type") == "array":
                    tool_schema = {
                        "type": "object",
                        "properties": {"results": output_schema},
                        "required": ["results"],
                        "additionalProperties": False,
                    }
                else:
                    tool_schema = output_schema
                create_kwargs["tools"] = [{
                    "name": "structured_output",
                    "description": "Return the structured result.",
                    "input_schema": tool_schema,
                }]
                create_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._client.messages.stream(**create_kwargs) as stream:
                    message = await stream.get_final_message()
                self.total_input_tokens += message.usage.input_tokens
                self.total_output_tokens += message.usage.output_tokens
                logger.debug(
                    "Agent call OK | in=%d out=%d stop=%s",
                    message.usage.input_tokens,
                    message.usage.output_tokens,
                    message.stop_reason,
                )
                if message.stop_reason == "max_tokens":
                    last_exc = RuntimeError(
                        f"Response truncated (max_tokens hit). "
                        f"Model={self.MODEL} MAX_TOKENS={self.MAX_TOKENS}. "
                        f"Retrying (attempt {attempt + 1}/{self.MAX_RETRIES})."
                    )
                    logger.warning(str(last_exc))
                    await asyncio.sleep(self.BASE_DELAY * (2 ** attempt))
                    continue
                # For tool-use responses (Haiku structured output), inject a text
                # block so callers can always use _find_text_block(message).
                if output_schema and not self._supports_thinking():
                    import json
                    tool_block = next(
                        (b for b in message.content if b.type == "tool_use"), None
                    )
                    if tool_block is not None:
                        import anthropic as _ant
                        # Unwrap the envelope if we wrapped an array schema.
                        # Haiku occasionally serialises the inner array as a JSON string
                        # instead of a real list — parse it when that happens.
                        payload = tool_block.input
                        logger.debug("Tool input type: %s", type(payload).__name__)
                        if not isinstance(payload, dict):
                            raise ValueError(
                                f"Expected tool_block.input to be dict, got {type(payload).__name__}"
                            )
                        if output_schema.get("type") == "array" and "results" in payload:
                            payload = payload["results"]
                            if isinstance(payload, str):
                                # The "results" value may itself be a stringified JSON array
                                # (Haiku quirk). Parse it to get the actual list.
                                logger.debug("Parsing stringified results array (len=%d)", len(payload))
                                try:
                                    payload = json.loads(payload)
                                except (json.JSONDecodeError, TypeError) as e:
                                    # If parsing fails, log and re-raise
                                    logger.error("Failed to parse stringified results: %s", e)
                                    raise
                        # Convert the parsed tool input back to JSON text. Use default
                        # separators and ensure_ascii=True for proper escaping of all
                        # special characters (including quotes, newlines, etc.).
                        json_text = json.dumps(payload, ensure_ascii=True)
                        logger.debug(
                            "Converted tool_use structured output to JSON (len=%d, first 200 chars): %s",
                            len(json_text),
                            json_text[:200],
                        )
                        text_content = _ant.types.TextBlock(
                            type="text",
                            text=json_text,
                        )
                        message.content = [text_content]
                return message
            except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
                last_exc = exc
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning("Retrying in %.1fs after %s", delay, type(exc).__name__)
                await asyncio.sleep(delay)
            except anthropic.APIError:
                raise

        raise last_exc  # type: ignore[misc]

    def log_usage(self) -> None:
        logger.info(
            "%s total tokens | in=%d out=%d",
            type(self).__name__,
            self.total_input_tokens,
            self.total_output_tokens,
        )
