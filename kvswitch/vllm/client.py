"""Simple UDP client for the vLLM UDP API server."""

import argparse
import asyncio
import json
import logging
from typing import Any

from kvswitch.utils.logger import setup_logging
from kvswitch.utils.udp import UDPClient

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for the vLLM UDP API server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: float = 30.0,
    ) -> None:
        self.udp = UDPClient(host=host, port=port, timeout=timeout)

    async def health(self) -> dict[str, Any]:
        """Check if the server is healthy."""
        return await self.udp.send({"endpoint": "health"})

    async def generate(self, prompt: str, **sampling_kwargs: Any) -> dict[str, Any]:
        """Send a generate request to the server with a text prompt."""
        request = {"endpoint": "generate", "prompt": prompt, **sampling_kwargs}
        return await self.udp.send(request)

    async def generate_tokens(
        self, prompt_token_ids: list[int], **sampling_kwargs: Any
    ) -> dict[str, Any]:
        """Send a generate request to the server with token IDs."""
        request: dict[str, Any] = {
            "endpoint": "generate",
            "prompt_token_ids": prompt_token_ids,
            **sampling_kwargs,
        }
        return await self.udp.send(request)

    async def reset(self) -> dict[str, Any]:
        """Send a reset request to the server."""
        return await self.udp.send({"endpoint": "reset"})


async def main(args: argparse.Namespace) -> None:
    client = VLLMClient(host=args.host, port=args.port, timeout=args.timeout)

    if args.endpoint == "health":
        response = await client.health()
    elif args.endpoint == "generate":
        kwargs: dict[str, Any] = {}
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        response = await client.generate(prompt=args.prompt, **kwargs)
    else:
        response = {"error": f"unknown endpoint: {args.endpoint}"}

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM UDP API client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="generate",
        choices=["health", "generate"],
    )
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    asyncio.run(main(args))
