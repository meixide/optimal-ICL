# icl_recursive_linear.py
import os, math, statistics, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
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

## --- formatting helpers (drop-in) ---

# def format_pair(p: Pair) -> str:
#     # Stable, one-liner representation; no spaces around separator to reduce drift.
#     # Use compact float formatting so numbers don't grow unnecessarily.
#     return f"{p.x:.12g}->{p.y:.12g}"

# def build_messages(context_pairs: list[Pair], query_x: float) -> list[dict]:
#     # Single-line context using a pipe as a hard separator.
#     ctx_line = " | ".join(format_pair(p) for p in context_pairs)

#     user = (
#         "CONTEXT: " + ctx_line + "\n"
#         f"QUERY: {query_x:.12g}\n"
#         # "TASK: Infer y so that 'x->y' matches the pattern implied by CONTEXT.\n"
#         # "ANSWER: Return STRICT JSON per schema (no extra text)."
#     )

#     return [
#         {"role": "system", "content":
#             "You are an ICL assistant. You will see two lines:\n"
#             " - 'CONTEXT:' with a single line of items like 'x->y' separated by ' | '\n"
#             " - 'QUERY:' with a single numeric x.\n"
#             "Use ONLY the 'QUERY:' line as the question. "
#             "Ignore any other numbers not in 'QUERY:'. "
#             "Return STRICT JSON per the provided schema (no extra text). "
#             "If rounding is needed, round to 6 decimals."
#         },
#         {"role": "user", "content": user},
#     ]

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
    print("Asking model with context:\n", messages)
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

if __name__ == "__main__":
    random.seed(7)

    # ----- Noiseless linear function: y = 2 + 3x -----
    a, b = 2.0, 3.0

    # Full dataset to sample from (you can replace by your own)
    xs_all = [0, 0.20, 0.5, 1, 1.3, 1.5, 2, 2.5, 3, 3.2, 3.5, 4, 4.5, 5.5]
    data_all = gen_linear_pairs(a, b, xs_all, noise_std=0.0)

    # Choose M seed context, N partials, and a validation set
    M = 14
    N = 6
    full_context_seed = data_all[:M]              # [(0,2), (0.5,3.5), (1,5)]
    partial_x = [0.25, 1.75, 3.75, 2.25, 4.75, 5.25][:N]                    # unlabeled at first
    validation = gen_linear_pairs(a, b, [3.1, 1.2, 5.75], noise_std=0.0)

    print("=== NOISELESS ===")
    res_clean = recursive_rollout_and_validate(full_context_seed, partial_x, validation)
    print("MSE:", res_clean["mse"])
    print("y_true:", res_clean["y_true"])
    print("y_pred:", res_clean["y_pred"])
    print("final_context_size:", res_clean["final_context_size"])

    # ----- Noisy linear function: y = 2 + 3x + ε, ε ~ N(0, 0.2^2) -----
    noise_std = 0.2
    data_all_noisy = gen_linear_pairs(a, b, xs_all, noise_std=noise_std)
    full_context_seed_noisy = data_all_noisy[:M]
    # Use different partials if you wish; reuse here
    validation_noisy = gen_linear_pairs(a, b, [3.1, 1.2, 5.75], noise_std=noise_std)

    print("\n=== NOISY ===")
    res_noisy = recursive_rollout_and_validate(full_context_seed_noisy, partial_x, validation_noisy)
    print("MSE:", res_noisy["mse"])
    print("y_true:", res_noisy["y_true"])
    print("y_pred:", res_noisy["y_pred"])
    print("final_context_size:", res_noisy["final_context_size"])
