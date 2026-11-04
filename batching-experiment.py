"""Goal: show that batching raises total throughput while per-request speed holds.

Sends 1, 2, 4, 8 and 16 concurrent requests to a local SGLang server on port 30000.
Prediction for an M2 with Qwen3-0.6B: total tok/s grows with batch size up to
about 10 requests, then flattens. Watch `#running-req` in the server log meanwhile.
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI

MODEL = "mlx-community/Qwen3-0.6B-4bit"
TOPICS = ["the sea", "a city", "a forest", "the moon", "a robot", "a river", "a train", "winter"]


async def one_request(client: AsyncOpenAI, *, index: int, max_tokens: int) -> dict:
    prompt = f"Write a long story about {TOPICS[index % len(TOPICS)]}. /no_think"
    start = time.perf_counter()
    first = None
    tokens = 0
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=max_tokens,
        temperature=0.7,
        # Keep generating to max_tokens so every request has the same length.
        extra_body={"ignore_eos": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            tokens += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    decode_rate = (tokens - 1) / (end - first) if tokens > 1 else 0.0
    return {"ttft": first - start, "tokens": tokens, "decode_rate": decode_rate}


async def run_level(client: AsyncOpenAI, *, batch: int, max_tokens: int) -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        *(one_request(client, index=i, max_tokens=max_tokens) for i in range(batch))
    )
    wall = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)
    avg_ttft = sum(r["ttft"] for r in results) / batch
    avg_decode = sum(r["decode_rate"] for r in results) / batch
    print(
        f"{batch:>5} | {total_tokens / wall:>13.1f} | {avg_decode:>15.1f} | "
        f"{avg_ttft:>8.3f} | {wall:>6.1f}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
    # Warm-up so the first measured level does not pay one-time startup costs.
    await one_request(client, index=0, max_tokens=10)

    print("batch | total tok/s   | per-req tok/s   | TTFT (s) | wall (s)")
    for batch in args.batches:
        await run_level(client, batch=batch, max_tokens=args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
