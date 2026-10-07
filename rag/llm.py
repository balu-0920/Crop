"""LLM client (Anthropic). The key is read from the environment / .env only - never hard-coded."""
from . import config


class LLMError(Exception):
    pass


class AnthropicLLM:
    def available(self):
        return bool(config.anthropic_api_key())

    def generate(self, system, user):
        key = config.anthropic_api_key()
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key, timeout=60.0)
            resp = client.messages.create(
                model=config.llm_model(), max_tokens=900, temperature=0,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
