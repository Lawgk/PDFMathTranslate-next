import logging

import httpx
import openai
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import wait_exponential_jitter

logger = logging.getLogger(__name__)

# Failures where the request never reached a verdict, so another attempt can
# still produce one: quota pressure, a connection that dropped or timed out
# (APITimeoutError subclasses APIConnectionError) and any 5xx from the upstream
# (every status >= 500 is mapped to InternalServerError by the OpenAI SDK).
#
# Everything else is deterministic — a malformed request or a content filter
# returns the same answer however often it is asked.
#
# Retrying only RateLimitError is what let the silent-untranslated failures
# through: an upstream timeout propagated out of here, il_translator swallowed
# it, and the source text stayed in the page while the task reported success.
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# A rate limit clears on its own once the window rolls over, so waiting it out
# costs nothing but time. A degraded upstream is the opposite: retries run below
# the QPS limiter, so every worker in the pool retries unthrottled and a blip
# turns into a stampede. Give up early there.
_RATE_LIMIT_MAX_ATTEMPTS = 100
_TRANSIENT_MAX_ATTEMPTS = 6


def _stop_by_error_kind(retry_state) -> bool:
    """Cap attempts by failure kind rather than applying one budget to all."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    limit = (
        _RATE_LIMIT_MAX_ATTEMPTS
        if isinstance(exception, openai.RateLimitError)
        else _TRANSIENT_MAX_ATTEMPTS
    )
    return retry_state.attempt_number >= limit


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.timeout = settings.translate_engine_settings.openai_timeout
        self.client = openai.OpenAI(
            base_url=settings.translate_engine_settings.openai_base_url,
            api_key=settings.translate_engine_settings.openai_api_key,
            timeout=float(self.timeout) if self.timeout else openai.NOT_GIVEN,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                )
            ),
        )
        self.options = {}
        self.temperature = settings.translate_engine_settings.openai_temperature
        self.reasoning_effort = (
            settings.translate_engine_settings.openai_reasoning_effort
        )
        self.send_temperature = (
            settings.translate_engine_settings.openai_send_temprature
        )
        self.send_reasoning_effort = (
            settings.translate_engine_settings.openai_send_reasoning_effort
        )
        self.extra_body = settings.translate_engine_settings._openai_extra_body

        if self.send_temperature and self.temperature:
            self.add_cache_impact_parameters("temperature", self.temperature)
            self.options["temperature"] = float(self.temperature)
        if self.send_reasoning_effort and self.reasoning_effort:
            self.add_cache_impact_parameters("reasoning_effort", self.reasoning_effort)
            self.options["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            self.add_cache_impact_parameters("extra_body", self.extra_body)
            self.options["extra_body"] = self.extra_body

        self.model = settings.translate_engine_settings.openai_model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

        self.enable_json_mode = (
            settings.translate_engine_settings.openai_enable_json_mode
        )
        if self.enable_json_mode:
            self.add_cache_impact_parameters("enable_json_mode", self.enable_json_mode)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=_stop_by_error_kind,
        wait=wait_exponential_jitter(initial=1, max=15, jitter=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        options = self.options.copy()
        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            options["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=self.prompt(text),
        )
        try:
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    self.token_count.inc(response.usage.total_tokens)
                if hasattr(response.usage, "prompt_tokens"):
                    self.prompt_token_count.inc(response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    self.completion_token_count.inc(response.usage.completion_tokens)
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_cache_hit_tokens
                    )
                elif hasattr(response.usage, "prompt_tokens_details") and hasattr(
                    response.usage.prompt_tokens_details, "cached_tokens"
                ):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_tokens_details.cached_tokens
                    )
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            pass
        message = response.choices[0].message.content.strip()
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=_stop_by_error_kind,
        wait=wait_exponential_jitter(initial=1, max=15, jitter=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None
        options = self.options.copy()
        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            options["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        try:
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    self.token_count.inc(response.usage.total_tokens)
                if hasattr(response.usage, "prompt_tokens"):
                    self.prompt_token_count.inc(response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    self.completion_token_count.inc(response.usage.completion_tokens)
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_cache_hit_tokens
                    )
                elif hasattr(response.usage, "prompt_tokens_details") and hasattr(
                    response.usage.prompt_tokens_details, "cached_tokens"
                ):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_tokens_details.cached_tokens
                    )
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            pass
        message = response.choices[0].message.content.strip()
        message = self._remove_cot_content(message)
        return message
