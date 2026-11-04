import time
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")

def measure(prompt):
    start = time.time(); first = None; n = 0
    stream = client.chat.completions.create(
        model="mlx-community/Qwen3-0.6B-4bit", stream=True, max_tokens=100, temperature=0,
        messages=[{"role": "user", "content": prompt + " /no_think"}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            n += 1
            if first is None: first = time.time()
    end = time.time()
    print(f"prompt ~{len(prompt.split())} words | TTFT {first-start:.3f}s | decode {n/(end-first):.1f} tok/s")

measure("Write a haiku about the sea.")
measure("Summarize this text in one sentence: " + "The ocean is vast and deep. " * 300)

