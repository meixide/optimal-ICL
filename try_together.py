from together import Together

TOGETHER_API_KEY = "409df7818af6854dd6aedb5764395a8c2eca0f137251e434425e2606ef3f6063"
client = Together(api_key=TOGETHER_API_KEY)

prompt = """
You are given pairs of numbers. Infer the pattern and answer with only the resulting number.

1 corresponds to 2.8
2 corresponds to 5.2
3 corresponds to 7.5
4 corresponds to 8.8


What does 3.3 correspond to?

Return ONLY the number with no explanation, no text, no punctuation.
"""

response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
)
print(response.choices[0].message.content)