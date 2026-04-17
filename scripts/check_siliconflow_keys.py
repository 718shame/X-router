#!/usr/bin/env python3
"""Quickly probe API_BASE API keys and surface basic rate-limit (TPM) hints."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

BASE_URL = "https://api.API_BASE.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPT = "Briefly introduce yourself in one sentence."


def _load_keys(raw_keys: Iterable[str], keys_file: Optional[str]) -> List[str]:
    """Collect keys from CLI flags, optional file, and API_BASE_KEYS env var."""
    keys: List[str] = [key.strip() for key in raw_keys if key.strip()]
    if keys_file:
        with open(keys_file, "r", encoding="utf-8") as handle:
            keys.extend(line.strip() for line in handle if line.strip())
    env_keys = os.getenv("API_BASE_KEYS")
    if env_keys:
        keys.extend(k.strip() for k in env_keys.split(",") if k.strip())
    if not keys:
        raise ValueError("No API keys provided. Use --keys, --keys-file, or API_BASE_KEYS.")
    return keys


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key
    return f"{key[:8]}...{key[-4:]}"


def _extract_tpm(headers: httpx.Headers) -> Tuple[Optional[int], Dict[str, str]]:
    """Return TPM hint (when headers expose it) and the raw ratelimit headers."""
    rate_headers = {k: v for k, v in headers.items() if "ratelimit" in k.lower()}
    tpm_hint: Optional[int] = None
    limit_tokens = rate_headers.get("x-ratelimit-limit-tokens") or rate_headers.get(
        "X-RateLimit-Limit-Tokens"
    )
    if limit_tokens:
        try:
            tpm_hint = int(float(limit_tokens))
        except ValueError:
            tpm_hint = None
    return tpm_hint, rate_headers


async def _check_key(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    key: str,
    prompt: str,
) -> Dict[str, object]:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 32,
    }
    async with semaphore:
        try:
            response = await client.post("/chat/completions", json=payload, headers=headers)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "key": key,
                "ok": False,
                "status": None,
                "error": str(exc),
            }

    if response.status_code == 200:
        data = response.json()
        tpm_hint, rl_headers = _extract_tpm(response.headers)
        usage = data.get("usage", {})
        return {
            "key": key,
            "ok": True,
            "status": response.status_code,
            "tpm_hint": tpm_hint,
            "ratelimit_headers": rl_headers,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    try:
        error_body = response.json()
    except Exception:  # pylint: disable=broad-except
        error_body = response.text

    return {
        "key": key,
        "ok": False,
        "status": response.status_code,
        "error": error_body,
    }


async def run(keys: List[str], prompt: str, timeout: float, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        tasks = [_check_key(client, semaphore, key, prompt) for key in keys]
        results = await asyncio.gather(*tasks)

    for res in results:
        key_repr = _mask_key(str(res["key"]))
        if res.get("ok"):
            print(
                f"[OK] {key_repr} status={res['status']} "
                f"tpm_hint={res.get('tpm_hint')} "
                f"ratelimit={res.get('ratelimit_headers')} "
                f"usage={{prompt:{res.get('prompt_tokens')}, completion:{res.get('completion_tokens')}, total:{res.get('total_tokens')}}}"
            )
        else:
            print(f"[BAD] {key_repr} status={res.get('status')} error={res.get('error')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe API_BASE keys against Qwen/Qwen2.5-7B-Instruct and report TPM hints."
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=[],
        help="Keys to test (space-separated).",
    )
    parser.add_argument(
        "--keys-file",
        default="scripts\keys_to_test.txt",
        help="Path to a file containing one key per line (optional).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the smoke request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum in-flight requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = _load_keys(args.keys, args.keys_file)
    asyncio.run(run(keys, args.prompt, args.timeout, args.concurrency))


if __name__ == "__main__":
    main()
