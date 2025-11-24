from together import Together
import os
import numpy as np

# Make sure this env var is set: export TOGETHER_API_KEY="..."
client = Together(api_key=os.environ["TOGETHER_API_KEY"])

EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # good general sentence-embedding model

prompt = "Explain in a sentence what is the sport of basketball."
answer = "Basketball is a team sport where players score by shooting a ball through a hoop."

joint_text = prompt + "\n\n[ANSWER]\n" + answer

resp = client.embeddings.create(
    model=EMBED_MODEL,
    input=[prompt, answer, joint_text],
)

e_prompt = np.array(resp.data[0].embedding)
e_answer = np.array(resp.data[1].embedding)
e_joint  = np.array(resp.data[2].embedding)

# Cosine similarity kernel on the joint embedding:
def K_joint(e_joint_i, e_joint_j):
    num = np.dot(e_joint_i, e_joint_j)
    den = np.linalg.norm(e_joint_i) * np.linalg.norm(e_joint_j)
    return num / den

# Or combine separate embeddings in a structured way:
def K_sep(e_p_i, e_a_i, e_p_j, e_a_j):
    # e.g. average the two cosine similarities
    def cos(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    return 0.5 * (cos(e_p_i, e_p_j) + cos(e_a_i, e_a_j))


print("Cosine similarity between prompt and answer embeddings:")
sim_prompt_answer = K_joint(e_prompt, e_answer)  # using joint embedding
print(sim_prompt_answer)
print("Cosine similarity between prompt and prompt + answer embeddings:")
sim_prompt_joint = K_joint(e_prompt, e_joint)  # using joint embedding
print(sim_prompt_joint)
print("Cosine similarity between answer and prompt + answer embeddings:")
sim_answer_joint = K_joint(e_answer, e_joint)  # using joint embedding
print(sim_answer_joint)