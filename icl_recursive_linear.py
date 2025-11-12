import os, math, statistics, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict, Any
from together import Together

# ---------- Config ----------
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"  # free reasoning model on Together
TEMPERATURE = 0.2
MAX_TOKENS = 200

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

# ---------- Data structures ----------
@dataclass
class Pair:
    x: float
    y: float

# ---------- Prompt helpers ----------
SYSTEM = (
    "You are an In-Context Learning (ICL) assistant. "
    "You are given a set of 'context' examples of the form "
    "'x_i corresponds to y_i'. "
    "When asked about a new 'x_test', infer the numeric y value that fits the pattern. "
    "You MUST answer in strict JSON with the schema provided (no extra keys, no text outside JSON). "
    "If you need to round, use standard rounding to 6 decimals."
)

def format_pair(p: Pair) -> str:
    # A minimal, deterministic phrasing to help the model lock onto the mapping
    return f"{p.x} corresponds to {p.y}"

def build_messages(context_pairs: List[Pair], query_x: float) -> List[Dict[str, Any]]:
    # Context as a small, clear set of lines + the query
    ctx_lines = "\n".join(format_pair(p) for p in context_pairs)
    user = (
        "Training context:\n"
        f"{ctx_lines}\n\n"
        f"Question: {query_x} corresponds to ?"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

# ---------- Structured output (JSON schema) ----------
# Together’s chat.completions supports response_format with json_schema for compliant JSON.
# We keep it minimal: a single number y_hat.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "icl_numeric_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "y_hat": {"type": "number"}
            },
            "required": ["y_hat"],
            "additionalProperties": False
        }
    }
}

def ask_model_numeric(context_pairs: List[Pair], x_query: float) -> float:
    messages = build_messages(context_pairs, x_query)
    # print("Asking model with context:\n", messages)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format=RESPONSE_FORMAT,
    )
    # Together returns content as JSON when using json_schema. Parse safely:
    content = resp.choices[0].message.content
    # content is guaranteed to be JSON per schema; eval via json library:
    import json
    obj = json.loads(content)
    return float(obj["y_hat"])

# ---------- ICL flow ----------
def recursive_rollout_and_validate(
    full_context: List[Pair],          # seed labeled context (take first M)
    partial_context_x: List[float],    # N unlabeled x's to get from the model and append
    validation: List[Pair]             # ground truth for final evaluation
) -> Dict[str, Any]:
    # 1) seed context
    context = list(full_context)

    # 2) recursively answer each partial x and append (x, y_hat) to context
    for x in partial_context_x:
        y_hat = ask_model_numeric(context, x)
        context.append(Pair(x, y_hat))

    # 3) answer validation with final context
    y_true, y_pred = [], []
    for pair in validation:
        pred = ask_model_numeric(context, pair.x)
        y_true.append(pair.y)
        y_pred.append(pred)

    # 4) compute MSE
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "mse": mse,
        "y_true": y_true,
        "y_pred": y_pred,
        "final_context": context,
        "final_context_size": len(context)
    }

# ---------- Toy linear tasks ----------
def gen_linear_pairs(a: float, b: float, xs: List[float], noise_std: float = 0.0) -> List[Pair]:
    out = []
    for x in xs:
        y = a + b * x
        if noise_std > 0:
            y += random.gauss(0.0, noise_std)
        out.append(Pair(x, y))
    return out

# Additional utilities for weighted sampling without replacement and replicates

def weighted_sample_without_replacement(
    population: Sequence[Any],
    probs: Sequence[float],
    k: int,
    rng: np.random.Generator,
) -> List[Any]:
    """
    Sample k items from `population` without replacement using weights `probs`.
    Uses numpy's choice with replace=False.
    """
    probs = np.array(probs, dtype=float)
    if probs.shape[0] != len(population):
        raise ValueError("probs and population must have the same length")
    if np.any(probs < 0):
        raise ValueError("probs must be nonnegative")
    s = probs.sum()
    if s <= 0:
        raise ValueError("sum(probs) must be > 0")
    p = probs / s
    idx = rng.choice(len(population), size=k, replace=False, p=p)
    return [population[i] for i in idx]

def run_single_replicate(
    context: List[Pair],          # full candidate labeled pool to sample from
    partial_x: List[float],              # N unlabeled x's to roll out sequentially
    validation: List[Pair],              # validation labeled pairs
) -> Dict[str, Any]:
    """
    One full replicate:
      - sample M labeled seeds (without replacement, weighted by probs)
      - N sequential rollouts on partial_x, appending each (x, y_hat) to the context
      - predict on validation
      - return per-point squared errors and overall MSE
    """
    # rng = np.random.default_rng(rng_seed)
    # full_context_seed = weighted_sample_without_replacement(
    #     data_all_pairs, probs, M, rng
    # )

    res = recursive_rollout_and_validate(
        full_context=context,
        partial_context_x=partial_x,
        validation=validation,
    )

    # Per-point squared errors for this replicate
    se = [(yt - yp) ** 2 for yt, yp in zip(res["y_true"], res["y_pred"])]
    return {
        "mse": float(res["mse"]),
        "se_per_val": se,          # list aligned with validation order
        "y_pred": res["y_pred"],   # optional: keep predictions if you want later analyses
        "context": res["final_context"],
        "context_size": res["final_context_size"],
    }

def aggregate_replicates(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate across B replicates:
      - overall MSE mean & variance
      - per-validation-point SE mean & variance
    """
    B = len(results)
    mses = np.array([r["mse"] for r in results], dtype=float)

    # Stack per-point SEs to shape (B, V) where V = len(validation)
    se_context = [[s.y for s in r["context"]] for r in results]
    se_context.insert(0, [p.x for p in results[0]["context"]])  # header for easier reading
    se_matrix = np.stack([np.array(r["se_per_val"], dtype=float) for r in results], axis=0)
    se_mean = se_matrix.mean(axis=0)
    se_var  = se_matrix.var(axis=0, ddof=1) if B > 1 else np.zeros_like(se_mean)

    summary = {
        "B": B,
        "contexts": se_context,
        "mse_mean": float(mses.mean()),
        "mse_var": float(mses.var(ddof=1) if B > 1 else 0.0),
        "per_val_se_mean": se_mean.tolist(),
        "per_val_se_var": se_var.tolist(),
    }
    return summary

def run_b_replicates_parallel(
    B: int,
    data_all_pairs: List[Pair],
    M: int,
    probs: Sequence[float],
    partial_x: List[float],
    validation: List[Pair],
    rng_seed: int = 123,             # replicate-specific seed
    max_workers: int = 4     # tune based on your rate limit & machine
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Launch B independent replicates in parallel threads (I/O-bound).
    Each replicate still performs N rollouts sequentially (context grows),
    but replicates run concurrently.
    """
    jobs = []
    results: List[Dict[str, Any]] = []
    
    rng = np.random.default_rng(rng_seed)
    context = weighted_sample_without_replacement(
        data_all_pairs, probs, M, rng
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for b in range(B):
            fut = ex.submit(
                run_single_replicate,
                context, partial_x, validation
            )
            jobs.append(fut)
        for fut in as_completed(jobs):
            results.append(fut.result())

    summary = aggregate_replicates(results)
    return results, summary

if __name__ == "__main__":
    random.seed(7)
    np.random.seed(7)

    # ----- Linear function: y = 2 + 3x -----
    a, b = 2.0, 3.0

    xs_all    = [0, 0.25, 0.75, 1.25, 2, 2.5, 3, 3.5, 4.25, 4.5, 5, 5.5, 6.75]
    probs_all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # xs_all    = [0, 0.20, 0.5, 1, 1.3, 1.5, 2, 2.5, 3, 3.2, 3.5, 4, 4.5, 5, 5.5]
    # probs_all = [1, 2, 1, 1, 1, 1, 1.5, 1, 1, 1, 1, 2, 2, 1.5]  

    # Build labeled pool (noiseless)
    data_all = gen_linear_pairs(a, b, xs_all, noise_std=0.0)

    # N rollouts and validation set
    M = 12               # choose M <= len(xs_all)
    N = 12
    partial_x = [0.5, 1.5, 1.75, 2.25, 2.75, 3.25, 3.75, 4, 4.75, 5.25, 6, 6.25][:N]
    validation = gen_linear_pairs(a, b, [1, 5.75, 6.5], noise_std=0.0)

    # Replicates
    B = 20
    max_workers = 10   # be mindful of API rate limits

    print("=== NOISELESS: B replicates of sequential rollouts ===")
    res_list_clean, summary_clean = run_b_replicates_parallel(
        B=B,
        data_all_pairs=data_all,
        M=M,
        probs=probs_all,
        partial_x=partial_x,
        validation=validation,
        rng_seed=123,
        max_workers=max_workers,
    )
    print("Overall MSE mean:", summary_clean["mse_mean"])
    print("Overall MSE var :", summary_clean["mse_var"])
    print("Per-val SE mean :", summary_clean["per_val_se_mean"])
    print("Per-val SE var  :", summary_clean["per_val_se_var"])
    print("Context examples:", summary_clean["contexts"])  # print replicate's final context

    # ----- Noisy linear function: y = 2 + 3x + ε, ε ~ N(0, 0.2^2) -----
    noise_std = 0.2
    data_all_noisy = gen_linear_pairs(a, b, xs_all, noise_std=noise_std)
    validation_noisy = gen_linear_pairs(a, b, [-0.25, 1, 5.75, 6.5], noise_std=noise_std)

    print("\n=== NOISY: B replicates of sequential rollouts ===")
    res_list_noisy, summary_noisy = run_b_replicates_parallel(
        B=B,
        data_all_pairs=data_all_noisy,
        M=M,
        probs=probs_all,
        partial_x=partial_x,
        validation=validation_noisy,
        rng_seed=123,
        max_workers=max_workers,
    )
    print("Overall MSE mean:", summary_noisy["mse_mean"])
    print("Overall MSE var :", summary_noisy["mse_var"])
    print("Per-val SE mean :", summary_noisy["per_val_se_mean"])
    print("Per-val SE var  :", summary_noisy["per_val_se_var"])