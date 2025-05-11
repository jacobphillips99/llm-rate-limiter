import asyncio
import os
import time
import aiohttp
import litellm

from llm_rate_limiter.rate_limit import rate_limiter


def estimate_tokens(payload: dict) -> int:
    return len(payload["messages"][0]["content"]) // 4 + payload['max_tokens']


async def make_api_call(prompt: str, model_name: str) -> tuple[str, dict]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1000,
    }
    est_token_consumption = estimate_tokens(payload)
    time_dict = dict(instantiation_time=time.time())
    # wait for rate limit to become available
    provider = model_name.split("/")[0]
    await rate_limiter.wait_and_acquire(provider=provider, model=model_name, tokens=est_token_consumption)
    time_dict["rate_limit_acquired_time"] = time.time()
    # once we've acquired the rate limit, make the API call
    print(f"Running call {prompt}")
    response = await litellm.acompletion(**payload)
    time_dict["response_time"] = time.time() - time_dict["rate_limit_acquired_time"]
    return response.choices[0].message.content, time_dict


async def main(model_name: str) -> None:
    prompts = [str(i) for i in range(6)]
    tasks = [make_api_call(prompt, model_name) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    breakpoint()


if __name__ == "__main__":
    model_name = "gemini/gemini-2.5-flash-preview-04-17"
    provider = model_name.split("/")[0]
    # override the rate limit config for this example script
    rate_limiter._provider_model_configs[provider][model_name].requests_per_minute = 2
    asyncio.run(main(model_name=model_name))
