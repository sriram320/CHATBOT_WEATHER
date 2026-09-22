"""LLM client (NVIDIA NIM, OpenAI-compatible endpoint).

The model is used at exactly three points in the graph: intent extraction,
the two fuzzy SOPs, and final wording. It never decides which policy applies.

Note on the model: nemotron-3-super is a reasoning model, so its chain of
thought arrives in a separate `reasoning_content` field. We disable thinking
explicitly for speed and keep `.content` as the only thing we ever read, so a
stray reasoning trace can never leak into a user-facing answer.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env)")
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def call_llm(prompt: str, system: str, max_tokens: int = 600, temperature: float = 0.2,
             attempts: int = 3) -> str:
    """Call the model, retrying briefly on transient failures.

    Without this, a rate-limit blip during a burst of calls (an eval run, or a
    reviewer clicking quickly) surfaces as a silently worse answer: the caller's
    except-branch falls back to 'unknown' slots and the user sees a vaguer
    reply with no indication anything went wrong. Retrying makes transient
    failures transient rather than invisible.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

    raise last_error
