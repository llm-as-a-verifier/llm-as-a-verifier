"""
Fine-grained reward model.

Instead of collapsing the verifier's judgement into one discrete label (as in
LLM-as-a-Judge), read its probability distribution over an ordered set of
score tokens and take the expectation:

    R(t, tau) = (1 / C K) * sum_c sum_k sum_g  p_theta(v_g | t, c, tau) * phi(v_g)

with C criteria, K repeated verifications, and G score tokens (granularity);
phi maps each score token to its scalar value.

Provides the granularity-20 scale, the verifier clients (DeepSeek / Gemini /
any OpenAI-compatible server with logprobs), the score expectation
`extract_score`, the pairwise prompt, and a cached batch scorer.
"""

import base64
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-v4-flash"

DEFAULT_MAX_WORKERS = 50
DEEPSEEK_MAX_WORKERS = 500
CACHE_INPUT_VERSION = 1

# DeepSeek reasoning settings. The reasoning trace shares the output budget
# with the answer, so the budget must cover both — too small and the call is
# cut off before the score tags. "high" effort with a 32k budget matched
# "max" accuracy on Terminal-Bench 2.1 at ~17% fewer output tokens.
DEEPSEEK_THINKING = {"type": "enabled"}
DEEPSEEK_REASONING_EFFORT = "high"
DEEPSEEK_MAX_TOKENS = 32768


def deepseek_reasoning_params():
    """(extra_body, max_tokens) for a DeepSeek call. ``DEEPSEEK_EFFORT``
    (``off`` / ``low`` / ``high`` / ``max``) and ``DEEPSEEK_MAX_TOKENS`` in
    the environment override the module defaults."""
    load_dotenv()
    effort = os.environ.get("DEEPSEEK_EFFORT", DEEPSEEK_REASONING_EFFORT)
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS",
                                    DEEPSEEK_MAX_TOKENS))
    if effort in ("off", "disabled", "none"):
        return {"thinking": {"type": "disabled"}}, max_tokens
    return ({"thinking": DEEPSEEK_THINKING, "reasoning_effort": effort},
            max_tokens)


class MissingAPIKeyError(RuntimeError):
    """No verifier backend found in the environment.

    Set ``OPENAI_BASE_URL`` (OpenAI-compatible server), ``DEEPSEEK_API_KEY``
    (DeepSeek), or ``VERTEX_API_KEY`` (Gemini via Vertex AI) — in the
    environment or a ``.env`` file — or pass a pre-built ``client=``. The
    backend must expose token-level logprobs.
    """


# ---------------------------------------------------------------------------
# g = 20 score-token scale
# ---------------------------------------------------------------------------

GRANULARITY = 20

SCALE = {
    "scale_description": (
        "Rate how likely the agent correctly solved the task on a "
        "20-point scale using letters A through T:\n"
        "  A = clearly and completely succeeded with verified output (best)\n"
        "  B-D = succeeded with only minor issues\n"
        "  E-G = above average, mostly correct with some issues\n"
        "  H-J = uncertain, leans toward success\n"
        "  K-M = uncertain, leans toward failure\n"
        "  N-P = below average, significant issues remain\n"
        "  Q-S = failed with some partial progress\n"
        "  T = clearly and completely failed (worst)"
    ),
    "score_format": "LETTER_A_TO_T",
    "valid_tokens": {
        **{chr(65 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
        **{chr(97 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
    },
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def load_dotenv(root_dir=None):
    """Load KEY=value pairs from `<root_dir>/.env` (default: the working
    directory) into the environment, without overriding existing values."""
    env_path = os.path.join(root_dir or os.getcwd(), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def create_gemini_client():
    """Build a ``google-genai`` client from ``VERTEX_API_KEY``. Vertex AI
    only — the plain Gemini API does not expose token-level logprobs."""
    from google import genai
    load_dotenv()
    vertex_key = os.environ.get("VERTEX_API_KEY")
    if vertex_key:
        return genai.Client(vertexai=True, api_key=vertex_key)
    raise MissingAPIKeyError(
        "set VERTEX_API_KEY in .env or environment (Vertex AI only — "
        "logprob extraction needs the Vertex API)")


# ---------------------------------------------------------------------------
# OpenAI-compatible client (vLLM, SGLang, OpenAI, ...)
# ---------------------------------------------------------------------------

def create_openai_client(base_url=None, api_key=None):
    """Build an ``openai`` client for any OpenAI-compatible server that
    returns token-level logprobs (vLLM, SGLang, OpenAI, DeepSeek).
    ``base_url`` defaults to ``OPENAI_BASE_URL``, ``api_key`` to
    ``OPENAI_API_KEY`` then ``DEEPSEEK_API_KEY``. A DeepSeek ``base_url``
    gets the DeepSeek call path."""
    from openai import OpenAI
    load_dotenv()
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise MissingAPIKeyError(
            "set OPENAI_BASE_URL to an OpenAI-compatible endpoint "
            "(e.g. http://localhost:8000/v1 for vLLM, or "
            "https://api.deepseek.com for DeepSeek)")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY", "EMPTY"))
    if "api.deepseek.com" in base_url:
        client._llm_verifier_model = DEEPSEEK_MODEL
        client._llm_verifier_deepseek = True
    return client


def create_deepseek_client(api_key=None, model=DEEPSEEK_MODEL):
    """Build an ``openai`` client for DeepSeek's hosted API from
    ``DEEPSEEK_API_KEY``. The client is tagged so scoring reads the
    distribution from DeepSeek's own sampled score tags instead of the
    vLLM-only prefill trick."""
    from openai import OpenAI
    load_dotenv()
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "set DEEPSEEK_API_KEY in .env or environment to use DeepSeek "
            "as the verifier")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    client._llm_verifier_model = model
    client._llm_verifier_deepseek = True
    return client


def create_client():
    """Build a verifier client from the environment: ``OPENAI_BASE_URL``,
    then ``DEEPSEEK_API_KEY``, then ``VERTEX_API_KEY``. Raises
    `MissingAPIKeyError` when none is configured."""
    load_dotenv()
    if os.environ.get("OPENAI_BASE_URL"):
        return create_openai_client()
    if os.environ.get("DEEPSEEK_API_KEY"):
        return create_deepseek_client()
    return create_gemini_client()


def default_max_workers():
    """Default verifier-call concurrency: 500 on DeepSeek (no concurrency
    rate limit), 50 otherwise."""
    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if "api.deepseek.com" in base_url or (
            not base_url and os.environ.get("DEEPSEEK_API_KEY")):
        return DEEPSEEK_MAX_WORKERS
    return DEFAULT_MAX_WORKERS


def _is_openai_client(client):
    """OpenAI clients expose `.chat.completions`; google-genai clients
    don't have `.chat`."""
    return hasattr(client, "chat")


# ---------------------------------------------------------------------------
# Token accounting — every verifier call records what it was billed for in
# the process-wide `USAGE` counter (input, cached-input share, output).
# ---------------------------------------------------------------------------

def _int(value):
    return int(value) if isinstance(value, (int, float)) else 0


def _usage_from_response(response):
    """(input, cached_input, output, reasoning) tokens for one verifier
    response. Handles the OpenAI-compatible shape (``usage``) and the
    google-genai shape (``usage_metadata``); unreported fields count as 0."""
    usage = getattr(response, "usage", None)
    if usage is not None:  # OpenAI-compatible (vLLM, SGLang, OpenAI, DeepSeek)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached = _int(getattr(prompt_details, "cached_tokens", 0))
        if not cached:
            cached = _int(getattr(usage, "prompt_cache_hit_tokens", 0))
        completion_details = getattr(usage, "completion_tokens_details", None)
        return (_int(getattr(usage, "prompt_tokens", 0)),
                cached,
                _int(getattr(usage, "completion_tokens", 0)),
                _int(getattr(completion_details, "reasoning_tokens", 0)))

    meta = getattr(response, "usage_metadata", None)
    if meta is not None:  # google-genai
        # Gemini reports the reasoning trace separately from the answer; both
        # are billed as output.
        thoughts = _int(getattr(meta, "thoughts_token_count", 0))
        return (_int(getattr(meta, "prompt_token_count", 0)),
                _int(getattr(meta, "cached_content_token_count", 0)),
                _int(getattr(meta, "candidates_token_count", 0)) + thoughts,
                thoughts)

    return 0, 0, 0, 0


class TokenUsage:
    """Thread-safe running total of verifier token usage.

    Attributes (all cumulative since the last `reset`):
        calls:          verifier requests that reported usage.
        input_tokens:   prompt tokens billed, cached and uncached together.
        cached_input_tokens: the prefix-cache-hit share of `input_tokens`.
        output_tokens:  completion tokens, reasoning trace included.
        reasoning_tokens: the reasoning-trace share of `output_tokens`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Zero every counter (e.g. between phases of a run)."""
        with self._lock:
            self.calls = 0
            self.input_tokens = 0
            self.cached_input_tokens = 0
            self.output_tokens = 0
            self.reasoning_tokens = 0

    def add(self, input_tokens=0, cached_input_tokens=0, output_tokens=0,
            reasoning_tokens=0, calls=1):
        with self._lock:
            self.calls += calls
            self.input_tokens += input_tokens
            self.cached_input_tokens += cached_input_tokens
            self.output_tokens += output_tokens
            self.reasoning_tokens += reasoning_tokens

    def record(self, response):
        """Add one backend response's usage. Safe to call from worker
        threads, and a no-op for a response that carries no usage block."""
        inp, cached, out, reasoning = _usage_from_response(response)
        if inp or cached or out:
            self.add(inp, cached, out, reasoning)
        return inp, cached, out, reasoning

    def snapshot(self):
        """The current counters as a plain dict, plus the derived
        `cache_hit_rate` (cached input / input) and `uncached_input_tokens`."""
        with self._lock:
            inp, cached = self.input_tokens, self.cached_input_tokens
            return {
                "calls": self.calls,
                "input_tokens": inp,
                "cached_input_tokens": cached,
                "uncached_input_tokens": inp - cached,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cache_hit_rate": (cached / inp) if inp else 0.0,
            }

    def __sub__(self, other):
        """`after - before` — the usage accumulated between two snapshots,
        as a `TokenUsage`, so a phase can be reported without resetting."""
        delta = TokenUsage()
        delta.add(self.input_tokens - other.input_tokens,
                  self.cached_input_tokens - other.cached_input_tokens,
                  self.output_tokens - other.output_tokens,
                  self.reasoning_tokens - other.reasoning_tokens,
                  calls=self.calls - other.calls)
        return delta

    def copy(self):
        c = TokenUsage()
        c.add(self.input_tokens, self.cached_input_tokens, self.output_tokens,
              self.reasoning_tokens, calls=self.calls)
        return c

    def __str__(self):
        s = self.snapshot()
        return (f"{s['calls']:,} calls  "
                f"input {s['input_tokens']:,} "
                f"(cached {s['cached_input_tokens']:,}, "
                f"{100 * s['cache_hit_rate']:.1f}%)  "
                f"output {s['output_tokens']:,}")


#: Process-wide verifier token counter, updated by every `call_*` in this
#: module. Read it with ``USAGE.snapshot()``; zero it with ``USAGE.reset()``.
USAGE = TokenUsage()


def token_usage():
    """Snapshot of the verifier token usage accumulated so far."""
    return USAGE.snapshot()


def format_usage(usage, title="Verifier tokens"):
    """Render a `TokenUsage` (or its snapshot dict) as report lines."""
    s = usage.snapshot() if isinstance(usage, TokenUsage) else usage
    return [
        f"{title} ({s['calls']:,} verifier calls)",
        f"  {'input':<24s}  {s['input_tokens']:>16,d}",
        f"  {'  cached input':<24s}  {s['cached_input_tokens']:>16,d}  "
        f"({100 * s['cache_hit_rate']:.1f}% hit rate)",
        f"  {'  uncached input':<24s}  {s['uncached_input_tokens']:>16,d}",
        f"  {'output':<24s}  {s['output_tokens']:>16,d}",
        f"  {'  reasoning':<24s}  {s['reasoning_tokens']:>16,d}",
    ]


# ---------------------------------------------------------------------------
# Image inputs — every public API accepts `images` as one image or a list;
# each image is a local file path, an http(s) URL, or raw bytes. Images are
# attached to the verifier message after the text prompt, in order.
# ---------------------------------------------------------------------------

def as_image_list(images):
    """Normalize the `images` argument: None -> [], a single image (path /
    URL / bytes) -> [image], a sequence -> list."""
    if images is None:
        return []
    if isinstance(images, (str, bytes, os.PathLike)):
        return [images]
    return list(images)


def _sniff_mime(data):
    """Image MIME type from magic bytes (PNG/JPEG/GIF/WebP; PNG default)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def load_image(image):
    """Load one image into (bytes, mime_type). Accepts a local file path,
    an http(s) URL, or raw image bytes."""
    if isinstance(image, bytes):
        data = image
    elif isinstance(image, (str, os.PathLike)):
        s = os.fspath(image)
        if s.startswith(("http://", "https://")):
            from urllib.request import urlopen
            with urlopen(s) as r:
                data = r.read()
        else:
            with open(s, "rb") as f:
                data = f.read()
    else:
        raise TypeError(
            f"unsupported image type {type(image).__name__}: expected a "
            "file path, an http(s) URL, or raw bytes")
    return data, _sniff_mime(data)


def resolve_model(client, model=DEFAULT_MODEL):
    """The model name to send to `client`. For an OpenAI-compatible client
    left on the Gemini default, ask the server what it serves and cache the
    answer, so a local server works without a ``model=`` argument."""
    if not _is_openai_client(client) or model != DEFAULT_MODEL:
        return model
    served = getattr(client, "_llm_verifier_model", None)
    if served is None:
        served = client.models.list().data[0].id
        client._llm_verifier_model = served
    return served


def call_openai(client, prompt, model=DEFAULT_MODEL, top_logprobs=20,
                images=None):
    """Call an OpenAI-compatible verifier with logprobs.
    Returns (text, tokens, position_logprobs) — the same shape as
    `call_gemini`, so `extract_score` works unchanged."""
    content = prompt
    imgs = as_image_list(images)
    if imgs:
        content = [{"type": "text", "text": prompt}]
        for img in imgs:
            data, mime = load_image(img)
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
    params = dict(
        model=resolve_model(client, model),
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=1.0,
        logprobs=True,
        top_logprobs=top_logprobs,  # the OpenAI API caps this at 20
    )
    try:
        # vLLM/SGLang only: skip hybrid-thinking so score tags come fast.
        response = client.chat.completions.create(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **params)
    except Exception:
        response = client.chat.completions.create(**params)
    USAGE.record(response)

    choice = response.choices[0]
    text = choice.message.content or ""
    tokens = None
    position_logprobs = None

    if choice.logprobs and choice.logprobs.content:
        tokens, position_logprobs = [], []
        for pos in choice.logprobs.content:
            tokens.append(pos.token)
            alts = [(alt.token, alt.logprob)
                    for alt in (pos.top_logprobs or [])]
            if not alts:
                alts = [(pos.token, pos.logprob)]
            position_logprobs.append(alts)

    # Open models don't reliably emit the score tags, so keep only the
    # analysis, prefill each tag, and read the letter distribution at that
    # exact position. DeepSeek emits the tags itself (and lacks the prefill
    # params), so it skips this.
    tags = [t for t in ("<score_A>", "<score_B>") if t in prompt]
    if tags and not getattr(client, "_llm_verifier_deepseek", False):
        idx = min([text.find(t) for t in tags if t in (text or "")]
                  or [len(text or "")])
        analysis = (text or "")[:idx].rstrip()
        text, tokens, position_logprobs = _score_tags_by_prefill(
            client, params["model"], params["messages"], analysis, tags,
            top_logprobs)

    return text, tokens, position_logprobs


def _score_tags_by_prefill(client, model, messages, text, tags,
                           top_logprobs=20):
    """Read each `<score_X>` letter distribution by prefilling the tag
    (vLLM/SGLang ``continue_final_message``), constrained to the scale
    letters via structured outputs. Returns (text, tokens,
    position_logprobs); a server without prefill support returns them
    tag-less (scores fall back to 0.5)."""
    tokens = []
    position_logprobs = []
    for tag in tags:
        prefix = (text or "") + f"\n{tag}"
        # Some models (e.g. Qwen VL) put nearly all their mass on the
        # letter WITH a leading space after the prefilled ">"; allow both
        # spellings so the grammar mask keeps the real distribution.
        letters = [chr(65 + i) for i in range(GRANULARITY)]
        letters += [" " + c for c in letters]
        try:
            # Constrain the prefilled position to the 20 score letters, so
            # the returned top-logprobs are the renormalized distribution
            # over the scale itself.
            response = client.chat.completions.create(
                model=model,
                messages=messages + [{"role": "assistant",
                                      "content": prefix}],
                max_tokens=1,
                temperature=1.0,
                logprobs=True,
                top_logprobs=top_logprobs,
                extra_body={"add_generation_prompt": False,
                            "continue_final_message": True,
                            "structured_outputs": {"choice": letters}},
            )
        except Exception:
            return text, tokens or None, position_logprobs or None
        USAGE.record(response)
        choice = response.choices[0]
        letter = (choice.message.content or "").strip()
        alts = []
        if choice.logprobs and choice.logprobs.content:
            pos = choice.logprobs.content[0]
            alts = [(alt.token, alt.logprob)
                    for alt in (pos.top_logprobs or [])]
        closing = "</" + tag[1:]
        text = prefix + letter + closing
        tokens += [f"\n{tag}", letter, closing]
        position_logprobs += [[(f"\n{tag}", 0.0)], alts, [(closing, 0.0)]]
    return text, tokens or None, position_logprobs or None


def call_deepseek(client, prompt, model=DEFAULT_MODEL, top_logprobs=20,
                  images=None):
    """Call the DeepSeek API with token-level logprobs and thinking enabled.

    A call whose reasoning consumed the whole output budget (no answer, no
    logprobs) raises rather than silently scoring a 0.5/0.5 tie. Returns
    (text, tokens, position_logprobs) in the `call_gemini` shape.
    """
    content = prompt
    imgs = as_image_list(images)
    if imgs:
        content = [{"type": "text", "text": prompt}]
        for img in imgs:
            data, mime = load_image(img)
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
    extra_body, max_tokens = deepseek_reasoning_params()
    response = client.chat.completions.create(
        model=resolve_model(client, model),
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=1.0,
        logprobs=True,
        top_logprobs=top_logprobs,
        extra_body=extra_body,
    )
    USAGE.record(response)
    choice = response.choices[0]
    text = choice.message.content or ""
    tokens = None
    position_logprobs = None
    if choice.logprobs and choice.logprobs.content:
        tokens, position_logprobs = [], []
        for pos in choice.logprobs.content:
            tokens.append(pos.token)
            alts = [(alt.token, alt.logprob)
                    for alt in (pos.top_logprobs or [])]
            if not alts:
                alts = [(pos.token, pos.logprob)]
            position_logprobs.append(alts)
    if not position_logprobs:
        # Reasoning consumed the whole budget: no score distribution to read.
        raise RuntimeError(
            "DeepSeek returned no answer logprobs "
            f"(finish_reason={choice.finish_reason!r}, "
            f"reasoning consumed the {max_tokens}-token budget); "
            "raise DEEPSEEK_MAX_TOKENS or lower DEEPSEEK_EFFORT")
    return text, tokens, position_logprobs


def call_verifier(client, prompt, model=DEFAULT_MODEL, top_logprobs=20,
                  images=None):
    """Route to the DeepSeek, OpenAI-compatible, or Gemini call based on the
    client type. Returns (text, tokens, position_logprobs)."""
    if getattr(client, "_llm_verifier_deepseek", False):
        return call_deepseek(client, prompt, model, top_logprobs,
                             images=images)
    if _is_openai_client(client):
        return call_openai(client, prompt, model, top_logprobs, images=images)
    return call_gemini(client, prompt, model, top_logprobs, images=images)


def call_gemini(client, prompt, model=DEFAULT_MODEL, top_logprobs=20,
                images=None):
    """Call the verifier model with logprobs.
    Returns (text, tokens, position_logprobs)."""
    from google.genai.types import (
        Content, GenerateContentConfig, Part, ThinkingConfig)

    config = GenerateContentConfig(
        max_output_tokens=4096,
        temperature=1.0,
        response_logprobs=True,
        logprobs=top_logprobs,
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

    parts = [Part(text=prompt)]
    for img in as_image_list(images):
        data, mime = load_image(img)
        parts.append(Part.from_bytes(data=data, mime_type=mime))

    response = client.models.generate_content(
        model=model,
        contents=[Content(role="user", parts=parts)],
        config=config,
    )
    USAGE.record(response)

    text = response.text or ""
    tokens = None
    position_logprobs = None

    candidate = response.candidates[0]
    if candidate.logprobs_result and candidate.logprobs_result.top_candidates:
        position_logprobs = []
        for pos in candidate.logprobs_result.top_candidates:
            alts = [(lp.token, lp.log_probability) for lp in pos.candidates]
            position_logprobs.append(alts)
        if candidate.logprobs_result.chosen_candidates:
            tokens = [c.token
                      for c in candidate.logprobs_result.chosen_candidates]

    return text, tokens, position_logprobs


# ---------------------------------------------------------------------------
# Score-token expectation:  sum_g p(v_g) * phi(v_g), normalized to [0, 1]
# ---------------------------------------------------------------------------

def _find_tag_logprobs(tokens, position_logprobs, tag):
    if not tokens or not position_logprobs:
        return None
    # Some tokenizers fuse the closing '>' with the score letter ('>A'), so
    # try the exact tag first, then the tag without its trailing '>'. Take
    # the LAST match: the verdict is the score block at the end of the reply,
    # not the model quoting the format mid-analysis.
    for suffix in (tag, tag[:-1]):
        found = None
        text_so_far = ""
        for i, tok in enumerate(tokens):
            text_so_far += tok
            if text_so_far.rstrip().endswith(suffix):
                if i + 1 < len(position_logprobs):
                    found = position_logprobs[i + 1]
        if found is not None:
            return found
    return None


def extract_score(text, tokens, position_logprobs, tag):
    """Expected score over the verifier's token distribution at `tag`,
    normalized to [0, 1]. Falls back to parsing the literal text token."""
    valid_tokens = SCALE["valid_tokens"]

    tag_lp = _find_tag_logprobs(tokens, position_logprobs, tag)
    probs = {}
    if tag_lp:
        for tok_str, logprob in tag_lp:
            tok = tok_str.strip()
            if tok.startswith(">"):  # DeepSeek fuses '>' with the letter
                tok = tok[1:].strip()
            if tok in valid_tokens:
                val = valid_tokens[tok]
                p = math.exp(logprob)
                probs[val] = max(probs.get(val, 0.0), p)

    if probs:
        unique_vals = sorted(set(valid_tokens.values()))
        min_val, max_val = min(unique_vals), max(unique_vals)
        total_p = sum(probs.values())
        expected = sum(v * p for v, p in probs.items()) / total_p
        return (expected - min_val) / (max_val - min_val) \
            if max_val > min_val else 0.5

    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    # Last match again: the verdict is the score block at the end.
    matches = list(re.finditer(pattern, text or "", re.IGNORECASE))
    match = matches[-1] if matches else None
    if match:
        tok = match.group(1).strip()
        raw_val = valid_tokens.get(tok)
        if raw_val is None:
            for vt, val in valid_tokens.items():
                if tok.lower() == vt.lower():
                    raw_val = val
                    break
        if raw_val is not None:
            unique_vals = sorted(set(valid_tokens.values()))
            min_val, max_val = min(unique_vals), max(unique_vals)
            return (raw_val - min_val) / (max_val - min_val) \
                if max_val > min_val else 0.5

    return 0.5


# ---------------------------------------------------------------------------
# Pairwise prompt + single-criterion scoring
# ---------------------------------------------------------------------------

def build_prompt(problem, trace_a, trace_b, criterion, ground_truth_note,
                 n_images=0):
    """One pairwise prompt focused on a single evaluation criterion.

    Everything not specific to the criterion (task, both trajectories, rating
    scale) comes first; only the criterion varies at the tail. This maximizes
    the shared prompt prefix across criteria, so a prefix-caching backend
    serves the trace-heavy body from cache. Keep criterion-specific text
    strictly at the end when editing."""
    images_note = (
        f"**Attached images:** {n_images} image(s) are attached to this "
        "message, in order; they are part of the task context.\n\n"
        if n_images else "")
    return (
        "You are an expert evaluator of AI coding agents. "
        "You will see a task description and two agent trajectories, then "
        "evaluate them on ONE specific criterion, stated at the end.\n\n"
        f"{ground_truth_note}\n\n"
        f"**Task:**\n{problem}\n\n"
        f"{images_note}"
        f"**Trajectory A:**\n{trace_a}\n\n"
        f"**Trajectory B:**\n{trace_b}\n\n"
        f"**Rating Scale:**\n{SCALE['scale_description']}\n\n"
        f"**Evaluation Guideline — {criterion['name']}:**\n"
        f"{criterion['description']}\n\n"
        f"Score each trajectory ONLY on this specific criterion "
        f"(\"{criterion['name']}\"). Ignore other aspects of the trajectory "
        f"that are not relevant to it.\n\n"
        "Reason it through first, then END your reply with exactly these two "
        "lines and nothing after them. Replace each placeholder with a single "
        "letter A-T, keeping the spaces around the letter exactly as shown:\n"
        f"<score_A> {SCALE['score_format']} </score_A>\n"
        f"<score_B> {SCALE['score_format']} </score_B>\n\n"
        "Begin your analysis now."
    )


def score_pair_criterion(client, problem, trace_a, trace_b, criterion,
                         ground_truth_note, model=DEFAULT_MODEL, images=None):
    """Score (A, B) for a single criterion, returning fine-grained rewards
    (R_A, R_B) in [0, 1]. `images` is attached as task context."""
    imgs = as_image_list(images)
    prompt = build_prompt(
        problem, trace_a, trace_b, criterion, ground_truth_note,
        n_images=len(imgs))
    text, tokens, position_logprobs = call_verifier(client, prompt, model,
                                                    images=imgs)
    ra = extract_score(text, tokens, position_logprobs, "<score_A>")
    rb = extract_score(text, tokens, position_logprobs, "<score_B>")
    return ra, rb


# ---------------------------------------------------------------------------
# Directed cache key + the per-comparison reward used by the tournament.
# Comparisons are *directed*: (a, b) and (b, a) are distinct cache entries,
# which the ring pass relies on to cancel the verifier's slot bias. Odd reps
# swap the prompt slots, so with K >= 2 the bias also cancels within one
# directed comparison. "score_A"/"score_B" always mean candidate a's / b's
# reward, whichever slot they occupied.
# ---------------------------------------------------------------------------

def cache_key(crit_id, task_name, a, b, rep):
    return f"{crit_id}|{task_name}|{a},{b}|{rep}"


def _image_cache_identity(image):
    """Stable identity for an image supplied to the verifier.

    Local files and raw bytes are content-addressed. URLs are identified by
    their declared value; callers using mutable URLs should use a fresh cache
    path when the remote content changes.
    """
    if isinstance(image, bytes):
        return {"kind": "bytes", "sha256": hashlib.sha256(image).hexdigest()}
    if isinstance(image, (str, os.PathLike)):
        value = os.fspath(image)
        if value.startswith(("http://", "https://")):
            return {"kind": "url", "value": value}
        try:
            with open(value, "rb") as image_file:
                digest = hashlib.sha256(image_file.read()).hexdigest()
            return {"kind": "file", "sha256": digest}
        except OSError:
            # Preserve the scorer's existing error handling for unreadable
            # paths instead of failing while the job list is assembled.
            return {"kind": "file", "value": os.path.abspath(value)}
    return {"kind": type(image).__name__, "value": repr(image)}


def _input_fingerprint(problem, trace_a, trace_b, criterion,
                       ground_truth_note, model, images):
    """SHA-256 of everything that determines a cached verifier request."""
    imgs = as_image_list(images)
    payload = {
        "version": CACHE_INPUT_VERSION,
        "model": model,
        "prompt": build_prompt(
            problem, trace_a, trace_b, criterion, ground_truth_note,
            n_images=len(imgs)),
        "images": [_image_cache_identity(image) for image in imgs],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def directed_reward(scores, task_name, a, b, criteria_ids, n_reps):
    """Fine-grained rewards (R_a, R_b) for the directed comparison (a, b),
    averaged over criteria and repeats. Missing entries default to 0.5."""
    if a == b:
        return 0.5, 0.5
    sa = sb = 0.0
    cnt = 0
    for cid in criteria_ids:
        for rep in range(n_reps):
            entry = scores.get(cache_key(cid, task_name, a, b, rep), {})
            sa += entry.get("score_A", 0.5)
            sb += entry.get("score_B", 0.5)
            cnt += 1
    return (sa / cnt, sb / cnt) if cnt else (0.5, 0.5)


class LazyClient:
    """Create the verifier client on first use, so reproducing a fully
    cached run never needs an API key."""

    def __init__(self):
        self._client = None

    def get(self):
        if self._client is None:
            self._client = create_client()
        return self._client


# ---------------------------------------------------------------------------
# Cached batch scoring — only the directed pairs PPT actually needs
# ---------------------------------------------------------------------------

def score_directed_pairs(lazy_client, tasks, needed_pairs, criteria,
                         ground_truth_note, n_reps, max_workers, cache_file,
                         model=DEFAULT_MODEL, progress=True, on_error="tie"):
    """Score every (criterion, rep) for the requested directed (task, a, b)
    pairs and merge into the cache on disk.

    `needed_pairs` maps task_name -> iterable of directed (a, b) comparisons;
    only comparisons missing from the cache trigger API calls. Odd reps swap
    the prompt slots (scores are recorded back in candidate order), so with
    `n_reps` >= 2 slot bias cancels within each comparison.

    Prefix-cache warming: the backend's prefix cache is only populated once a
    request returns, so one job per distinct prompt prefix runs to completion
    first; the rest then fan out against the warm cache.

    `on_error`: ``"tie"`` scores a failed call 0.5/0.5 for this run only
    (failures are never written to the cache), ``"raise"`` re-raises the
    first failure. Returns the merged scores dict."""
    if on_error not in ("tie", "raise"):
        raise ValueError(f"on_error must be 'tie' or 'raise', got {on_error!r}")

    cached = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)

    jobs = []
    for task_name, pairs in needed_pairs.items():
        trials = tasks[task_name]
        for a, b in pairs:
            for crit in criteria:
                for rep in range(n_reps):
                    key = cache_key(crit["id"], task_name, a, b, rep)
                    swap = rep % 2 == 1
                    ta, tb = trials[a]["trace"], trials[b]["trace"]
                    if swap:
                        ta, tb = tb, ta
                    fingerprint = _input_fingerprint(
                        trials[a]["problem"], ta, tb, crit,
                        ground_truth_note, model,
                        trials[a].get("images"))
                    entry = cached.get(key)
                    if (not isinstance(entry, dict)
                            or entry.get("input_sha256") != fingerprint):
                        # Prompts with the same (task, slot-A, slot-B) share
                        # a prefix.
                        sa, sb = (b, a) if swap else (a, b)
                        prefix = (task_name, sa, sb)
                        jobs.append((key, trials[a]["problem"], ta, tb,
                                     crit, trials[a].get("images"), swap,
                                     prefix, fingerprint))

    log = print if progress else (lambda *a, **kw: None)

    if not jobs:
        log(f"  All scores cached ({len(cached)} entries)")
        return cached

    # Warm-up wave: one job per distinct prompt prefix, then the rest.
    seen = set()
    warm, rest = [], []
    for job in jobs:
        prefix = job[7]
        if prefix in seen:
            rest.append(job)
        else:
            seen.add(prefix)
            warm.append(job)

    log(f"  {len(jobs)} scoring jobs ({len(cached)} cached); "
        f"warming {len(warm)} prefixes")

    client = lazy_client.get()
    usage_before = USAGE.copy()
    # `results` is what this run sees; `cached` is what gets persisted
    # (error ties go into `results` only).
    results = dict(cached)
    errors = 0
    done = 0
    save_every = max(1, len(jobs) // 20)

    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(jobs), desc="Scoring")
        except ImportError:
            pbar = None

    def run_phase(phase_jobs):
        nonlocal errors, done
        if not phase_jobs:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(score_pair_criterion, client, prob, ta, tb,
                                crit, ground_truth_note, model, images):
                    (key, swap, fingerprint)
                for key, prob, ta, tb, crit, images, swap, _, fingerprint
                in phase_jobs
            }
            for future in as_completed(futures):
                key, swap, fingerprint = futures[future]
                try:
                    ra, rb = future.result()
                    if swap:  # scores back in candidate order
                        ra, rb = rb, ra
                    entry = {
                        "score_A": ra,
                        "score_B": rb,
                        "input_sha256": fingerprint,
                    }
                    cached[key] = entry
                    results[key] = entry
                except Exception as e:
                    if on_error == "raise":
                        for f in futures:
                            f.cancel()
                        raise
                    errors += 1
                    results[key] = {"score_A": 0.5, "score_B": 0.5}
                    if errors <= 3:
                        log(f"\n  Error: {e}")
                done += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(errors=errors)
                if cache_file and done % save_every == 0:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cached, f)

    run_phase(warm)
    run_phase(rest)
    if pbar is not None:
        pbar.close()

    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached, f)

    log(f"  Done ({errors} errors)")
    log(f"  Tokens: {USAGE - usage_before}")
    return results
