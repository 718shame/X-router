import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.clients.API_BASE import API_BASEClient

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"answer\s*:\s*(.*)", flags=re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class GenerationTemplate:
    default: str
    cot: str


class LLMGenerator:
    def __init__(
        self,
        client: API_BASEClient,
        template: GenerationTemplate,
        max_new_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._template = template
        self._max_new_tokens = max_new_tokens

    def _build_prompt(self, context: str, query: str, use_cot: bool) -> str:
        tpl = self._template.cot if use_cot else self._template.default
        return tpl.format(context_str=context, query_str=query)

    async def _call_model(
        self,
        prompt: str,
        temperature: float,
    ) -> Dict[str, Any]:
        return await self._client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=self._max_new_tokens,
        )

    async def generate(
        self,
        context: str,
        query: str,
        use_cot: bool,
        temperature: float = 0.3,
    ) -> Tuple[str, int, int]:
        prompt = self._build_prompt(context, query, use_cot)
        response = await self._call_model(prompt, temperature)
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", usage.get("generated_tokens", 0))
        answer = extract_answer(text)
        total_tokens = prompt_tokens + completion_tokens
        return answer, total_tokens, completion_tokens

    async def sample(
        self,
        context: str,
        query: str,
        use_cot: bool,
        n_samples: int,
        temperature: float,
    ) -> List[str]:
        if n_samples <= 0:
            return []

        prompt = self._build_prompt(context, query, use_cot)

        async def _sample_once() -> str:
            response = await self._call_model(prompt, temperature)
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return extract_answer(text)

        tasks = [_sample_once() for _ in range(n_samples)]
        return await asyncio.gather(*tasks)
