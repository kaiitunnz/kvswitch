"""
Simple vLLM API server over UDP.

Similar to vllm/entrypoints/api_server.py but accepts UDP requests
instead of HTTP on TCP. Not intended for production use.
"""

import asyncio
import logging
import signal
from argparse import Namespace
from dataclasses import asdict
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

from kvswitch.utils.udp import UDPRequest, UDPResponse, UDPServer

logger = logging.getLogger(__name__)

engine: AsyncLLMEngine | None = None


async def handle_request(request: UDPRequest) -> UDPResponse:
    """Route incoming UDP requests based on the 'endpoint' field."""
    endpoint = request.data.get("endpoint", "generate")

    match endpoint:
        case "health":
            return UDPResponse(data={"status": "ok"})
        case "generate":
            return await handle_generate(request)
        case "reset":
            return await handle_reset(request)
        case _:
            return UDPResponse(data={"error": f"unknown endpoint: {endpoint}"})


async def handle_generate(request: UDPRequest) -> UDPResponse:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the text prompt to use for the generation, OR
    - prompt_token_ids: a list of token IDs to use as the prompt.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert engine is not None

    request_dict = dict(request.data)
    request_dict.pop("endpoint", None)

    prompt = request_dict.pop("prompt", None)
    prompt_token_ids = request_dict.pop("prompt_token_ids", None)

    if prompt is None and prompt_token_ids is None:
        return UDPResponse(data={"error": "missing 'prompt' or 'prompt_token_ids'"})

    # Build engine-compatible prompt input.
    engine_prompt: Any
    if prompt_token_ids is not None:
        engine_prompt = {"prompt_token_ids": prompt_token_ids}
    else:
        engine_prompt = prompt

    try:
        sampling_params = SamplingParams(**request_dict, skip_clone=True)
    except Exception as e:
        return UDPResponse(data={"error": f"invalid sampling params: {e}"})

    request_id = random_uuid()

    final_output = None
    async for request_output in engine.generate(
        engine_prompt, sampling_params, request_id
    ):
        final_output = request_output

    assert final_output is not None

    text_outputs: list[str] = []
    for output in final_output.outputs:
        prefix = final_output.prompt or ""
        text_outputs.append(prefix + output.text)

    data: dict[str, Any] = {
        "text": text_outputs,
        "num_cached_tokens": final_output.num_cached_tokens,
    }
    if final_output.metrics is not None:
        data["metrics"] = asdict(final_output.metrics)

    return UDPResponse(data=data)


async def handle_reset(request: UDPRequest) -> UDPResponse:
    """Reset the engine state."""
    assert engine is not None
    await engine.reset_prefix_cache()
    return UDPResponse(data={"status": "ok"})


async def run_server(
    args: Namespace,
    llm_engine: AsyncLLMEngine | None = None,
) -> None:
    logger.info("vLLM UDP API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    global engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (
        llm_engine
        if llm_engine is not None
        else AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )
    )

    server = UDPServer(
        host=args.host or "0.0.0.0", port=args.port, handler=handle_request
    )
    await server.start()

    # Register signal handlers so SIGTERM/SIGINT trigger graceful shutdown
    # instead of an immediate process kill (which would leak GPU memory).
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down…")
        server.close()
        if engine is not None:
            engine.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    assert args is not None
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    asyncio.run(run_server(args))
