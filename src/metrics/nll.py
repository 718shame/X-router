from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


NLI_LABEL = Literal["entail", "neutral", "contradict"]


@dataclass
class NLIConfig:
    model_name: str = "nli-deberta-v3-xsmall"  # Local model name or path
    device: str = "cpu"  # CCP computation is not intensive, CPU is sufficient
    # Label mapping for the corresponding model: index -> label name
    id2label: Dict[int, NLI_LABEL] = None

    def __post_init__(self) -> None:
        if self.id2label is None:
            # Most NLI models follow this order: 0: entailment, 1: neutral, 2: contradiction
            self.id2label = {0: "entail", 1: "neutral", 2: "contradict"}


class NLIPredictor:


    def __init__(self, config: NLIConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            local_files_only=True,
        ).to(config.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, premise: str, hypothesis: str) -> NLI_LABEL:
        """Return 'entail' / 'neutral' / 'contradict'."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.config.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        label_id = int(torch.argmax(logits).item())
        return self.config.id2label[label_id]


def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Legacy function (currently only used for CCP legacy implementation, can be deleted if no longer needed).

    No longer called in the new NLL-based implementation.
    """
    x = x - x.max()
    exps = np.exp(x)
    return exps / (exps.sum() + 1e-9)


def compute_word_ccp(
    *,
    greedy_sentence: str,
    prefix_sentence: str,
    candidate_tokens: Sequence[str],
    candidate_logprobs: Sequence[float],
    nli: NLIPredictor,
) -> float:
    """
    Legacy implementation (old word-level CCP):

    - Current X-router "CCP metric" no longer uses this function.
    - Keep this function only for completeness and potential reuse; if CCP is no longer used in the future, consider deleting.

    Old definition:
      CCP_word = sum_e P(e) / sum_{e or c} P(e/c)
    """
    if not candidate_tokens:
        return 1.0

    probs = _softmax(np.array(candidate_logprobs, dtype=np.float32))

    entail_prob = 0.0
    e_or_c_prob = 0.0

    for token, p in zip(candidate_tokens, probs):
        hypothesis_sentence = prefix_sentence + token
        relation = nli.predict(greedy_sentence, hypothesis_sentence)

        if relation == "entail":
            entail_prob += float(p)
            e_or_c_prob += float(p)
        elif relation == "contradict":
            e_or_c_prob += float(p)

    if e_or_c_prob <= 0.0:
        return 1.0

    return float(entail_prob / e_or_c_prob)


def compute_claim_ccp(
    *,
    full_sentence: str,
    token_infos: Sequence[Dict],
    nli: NLIPredictor,
    max_tokens: int | None = None,
) -> float:
    """
    Legacy implementation (old claim-level CCP):

    - Current main metric has changed to NLL-based, no longer calls this function.
    - Keep only for possible research code or subsequent ablation experiments.

    Old definition:
      CCP_claim = 1 - Π_j CCP_word(x_j)
    """
    if not token_infos:
        return 0.0

    if max_tokens is not None:
        token_infos = token_infos[:max_tokens]

    word_ccps: List[float] = []
    prefix = ""

    for tok_info in token_infos:
        greedy_token = tok_info.get("token", "")
        top = tok_info.get("top_logprobs") or []
        if not greedy_token or not top:
            continue

        candidate_tokens = [t["token"] for t in top]
        candidate_logprobs = [float(t["logprob"]) for t in top]

        word_ccp = compute_word_ccp(
            greedy_sentence=full_sentence,
            prefix_sentence=prefix,
            candidate_tokens=candidate_tokens,
            candidate_logprobs=candidate_logprobs,
            nli=nli,
        )
        word_ccps.append(word_ccp)

        prefix += greedy_token

    if not word_ccps:
        return 0.0

    word_ccps_arr = np.clip(np.array(word_ccps, dtype=np.float32), 1e-6, 1.0)
    claim_ccp = 1.0 - float(np.prod(word_ccps_arr))
    return claim_ccp


def compute_ccp_from_logprobs(
    *,
    answer_text: str,
    logprobs_content: Sequence[Dict],
    nli: Optional[NLIPredictor] = None,
    max_tokens: int | None = None,
    reduction: str = "mean",
) -> float:
    """
    New main implementation: **NLL-based uncertainty metric** based on logprobs.

    Important behavioral note:
      - Function name still called `compute_ccp_from_logprobs` for compatibility with all historical calls.
      - But implementation has switched from "CCP (Claim-Conditioned Probability)" to:
          → Computing **NLL or average NLL** for this generation response.
      - That is, the "ccp value" returned by the current version has semantically changed to:
          • reduction="mean" (default): nll_avg = - (1/T) * sum_t log p(y_t)
          • reduction="sum": nll_sum = - sum_t log p(y_t)
        where log p(y_t) comes directly from LLM returned logprobs["content"][t]["logprob"].

    Design rationale:
      - CCP often lacks discrimination on short-answer tasks and requires additional NLI models, with high overhead.
      - To obtain a **CoT-router signal that can be obtained with one decoding and is stable for both short/long tasks**,
        we adopt NLL as the main difficulty/uncertainty metric (referencing multiple QA/Code calibration works).

    Parameters:
      answer_text: Complete text of the answer (not involved in NLL calculation, but retains interface for future expansion).
      logprobs_content: logprobs["content"] returned by vLLM /chat.completions, in the form:
        [
          {
            "token": "Tokyo",
            "logprob": -0.1,
            "top_logprobs": [
              {"token": "Tokyo", "logprob": -0.1},
              {"token": "Osaka", "logprob": -1.3},
              ...
            ],
          },
          ...
        ]
      nli: Parameter for backward compatibility, ignored in current implementation.
      max_tokens: Optional, takes at most first N tokens to participate in NLL calculation to avoid overly long responses.
      reduction:
        - "mean": Returns average NLL, suitable for cross-length comparison (recommended as router feature).
        - "sum": Returns total NLL.

    Returns:
      float:
        - When reduction="mean", the larger the value, the less confident the model is in the answer (more difficult).
        - When reduction="sum", the larger the value, the higher the total "surprise" of the entire answer.
    """
    if not logprobs_content:
        return 0.0

    if max_tokens is not None:
        logprobs_content = logprobs_content[:max_tokens]

    logps: List[float] = []
    for tok_info in logprobs_content:
        lp = tok_info.get("logprob", None)
        if lp is None:
            continue
        logps.append(float(lp))

    if not logps:
        return 0.0

    logps_arr = np.array(logps, dtype=np.float32)
    nll_sum = -float(logps_arr.sum())  # Negative log-likelihood

    if reduction == "sum":
        return nll_sum
    else:
        # Default: Average NLL, better suited as "uncertainty / difficulty" metric
        return nll_sum / float(len(logps_arr))