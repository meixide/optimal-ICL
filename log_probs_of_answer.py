from together import Together
import os, math, statistics, random

# ---------- Config ----------
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # free reasoning model on Together

# MODEL =  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

client = Together(api_key=os.environ["TOGETHER_API_KEY"])
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

SYSTEM = (
    "You are an ICL assistant.\n"
    "You'll receive two lines:\n"
    "  CONTEXT: a single line with items 'x_i corresponds to y_i' separated by ' ; '\n"
    "  QUERY: What does 'x_test' correspond to?\n"
    "Infer y_test for QUERY to match the pattern from CONTEXT.\n"
    "You may do any reasoning internally, but DO NOT show it.\n"
    "Return ONLY one line in the exact format: ANS: <number>\n"
    "No extra words, no JSON, no units."
)

# SYSTEM = (
#     "You are an In-Context Learning (ICL) assistant. "
#     "You are given a set of 'context' examples of the form "
#     "'x_i corresponds to y_i'. "
#     "When asked about a new 'x_test', infer the numeric y value that fits the pattern. "
#     "You MUST answer in strict JSON with the schema provided (no extra keys, no text outside JSON). "
#     "If you need to round, use standard rounding to 6 decimals."
# )


resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": SYSTEM}, {"role":"user","content":"CONTEXT: 0.1 corresponds to 2.01 ; 2.4 corresponds to 7.76 ; 3 corresponds to 11 ; \n QUERY: What does 5 correspond to?"}],
    logprobs=5,
    # response_format=RESPONSE_FORMAT,
    echo=False        # set True if you also want prompt tokens/logprobs
)

lp = resp.choices[0].logprobs
lp.tokens
# for token, logprob, top_logprobs in zip(lp.tokens, lp.token_logprobs, lp.top_logprobs):
#     print(f"Token: {token}\n  Logprob: {logprob}\n  Top logprobs: {top_logprobs}\n")