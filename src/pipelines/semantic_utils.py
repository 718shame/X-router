from __future__ import annotations

from typing import Optional, Any

from src.clients.API_BASE import API_BASEClient
from Router.cognitive_LLM.src.metrics.nll import (
    # NLIPredictor  # ✅ NLI model no longer needed, keeping class definition in ccp.py only for backward compatibility
    # NLIConfig,
    compute_ccp_from_logprobs,
)


async def compute_ccp_metric(
    client: API_BASEClient,
    generator: Any,
    *,
    context: str,
    query: str,
    use_cot: bool,
    temperature: float,
    top_logprobs: int = 10,
    max_tokens: int = 256,
    nli: Optional[Any] = None,
) -> float:
    """
    Interface for computing "CCP metric" for a single (context, query) pair, but internally changed to **NLL-based metric**.

    ⚠️ Important note (behavior change):
      - Before: This function would call the CCP module, rely on NLI model, and return claim-level CCP (1 - Π_j CCP_word).
      - Now: To reduce overhead and adapt to short answer tasks, this function now calculates **average NLL (negative log-likelihood)**:
            nll_avg = - (1 / T) * sum_t log p(y_t | y_<t, x)
        Where y_t is the current generated token, and log p is given by the LLM's logprobs field.
      - Therefore:
          • Larger return value → model is more uncertain / "harder"; suitable as difficulty signal for CoT-router.
          • Smaller return value → model is more confident; may not need expensive CoT or RAG.

    Interface remains unchanged:
      - Function name still compute_ccp_metric, to facilitate reuse of all upstream code.
      - Parameter list unchanged, including nli, but nli is no longer used now, kept only for compatibility.

    Calculation process:
      1. Use generator to construct prompt / messages.
      2. Call client.generate_with_logprobs to get one response + logprobs (single decoding, no multi-sampling).
      3. Call src.metrics.ccp.compute_ccp_from_logprobs:
         - In new implementation, this function calculates "average NLL" based on logprobs and returns it.
    """
    # ✅ No longer using NLI, so not instantiating NLIPredictor to avoid extra model loading.
    # if nli is None:
    #     nli = NLIPredictor(NLIConfig())

    # 1) Construct prompt (adjust appropriately according to your project's LLMGenerator definition)
    prompt = generator._build_prompt(context=context, query=query, use_cot=use_cot)

    # 2) Call vLLM's /chat/completions to get logprobs
    resp = await client.generate_with_logprobs(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
    )

    choices = resp.get("choices") or []
    if not choices:
        # No response returned, handle as 0.0 (can be considered as "uninformative" uncertainty)
        return 0.0

    choice0 = choices[0]
    answer_text = choice0.get("message", {}).get("content", "") or ""

    logprobs = choice0.get("logprobs") or {}
    content = logprobs.get("content") or []

    if not answer_text or not content:
        # No text or no logprobs, handle as 0.0 as well
        return 0.0

    # 3) Call "CCP" calculation function (actually now NLL-based metric)
    #    Note: Parameter name still called nli for compatibility with previous signature, but will be ignored in new implementation.
    ccp_like_score = compute_ccp_from_logprobs(
        answer_text=answer_text,
        logprobs_content=content,
        nli=nli,  # Keep the call signature unchanged; the function will ignore this parameter internally
        max_tokens=max_tokens,
    )
    return ccp_like_score