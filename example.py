"""
Example script demonstrating the usage of the rate limiter with concurrent API calls.
We generate N requests with a rate limit of N-1 requests per minute and show that the
last request must wait for the rate limit to clear.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import draccus
import litellm
import pandas as pd

from llm_rate_limiter.rate_limit import rate_limiter

litellm._logging._disable_debugging()


async def make_api_call(prompt: str, provider: str, model_name: str) -> tuple[str, dict]:
    # setup payload
    max_tokens = 1000
    payload = {
        "model": f"{provider}/{model_name}" if "gemini" not in provider else model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
    }
    est_token_consumption = len(prompt) // 4 + max_tokens
    time_dict = dict(instantiation_time=time.time())

    # announce that the function is waiting for the rate limit
    print(f"Call {prompt} waiting for rate limit!")

    # wait for rate limit to become available
    await rate_limiter.wait_and_acquire(
        provider=provider, model=model_name, tokens=est_token_consumption
    )
    time_dict["rate_limit_acquired_time"] = time.time()

    # once we've acquired the rate limit, make the API call
    print(f"Running call {prompt}")
    response = await litellm.acompletion(**payload)
    content = response.choices[0].message.content
    time_dict["response_time"] = time.time()

    # record usage of tokens and request to the rate limiter
    tokens_used = (
        response.usage.model_dump()["total_tokens"]
        if hasattr(response.usage, "model_dump")
        else est_token_consumption
    )
    rate_limiter.record_usage(provider=provider, model=model_name, tokens_used=tokens_used)

    return content, time_dict


async def run(endpoints: list[tuple[str, str]], prompts: list[str]) -> pd.DataFrame:
    tasks = [
        make_api_call(prompt, provider, model_name)
        for prompt in prompts
        for provider, model_name in endpoints
    ]
    responses = await asyncio.gather(*tasks)
    timings = [res[1] for res in responses]
    return pd.DataFrame(timings)


@dataclass
class ExampleConfig:
    run_all: bool = False  # select to run ALL models or just one
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    n_reqs: int = 3
    rpm_limit: Optional[int] = None


@draccus.wrap()
def main(cfg: ExampleConfig) -> None:
    if cfg.run_all and rate_limiter._config is not None:
        provider_model_names = [
            (provider, name)
            for provider, names in rate_limiter._config.get_provider_to_model_name().items()
            for name in names
        ]
    else:
        provider_model_names = [(cfg.provider, cfg.model_name)]

    prompts = [str(i) for i in range(cfg.n_reqs)]

    # override the rate limit config for this example script --> make one request lag the others
    rpm_limit = cfg.rpm_limit if cfg.rpm_limit else cfg.n_reqs - 1
    for provider, model in provider_model_names:
        rate_limiter._provider_model_configs[provider][model].requests_per_minute = rpm_limit
        rate_limiter._provider_model_configs[provider][model].tokens_per_minute = cfg.n_reqs * 1500

    print("-" * 100)
    print(
        f"Instantiating {cfg.n_reqs} requests over {len(provider_model_names)} model{'s' if len(provider_model_names) > 1 else ''} each with a rate limit of {rpm_limit} requests per minute"
    )
    print("-" * 100)
    # actually instantiate the requests and run the main function
    time_df = asyncio.run(run(endpoints=provider_model_names, prompts=prompts))
    wait_times = time_df.rate_limit_acquired_time - time_df.instantiation_time

    print("-" * 100)
    print(
        f"Wait times should be close to 0 for the first {cfg.n_reqs - 1} requests and slightly over a minute for the last request"
    )
    print("-" * 100)
    print(f"Wait times: {wait_times}")


if __name__ == "__main__":
    main()
