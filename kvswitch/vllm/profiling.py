"""TTFT profiling for vLLM with prefix caching on/off.

Measures time-to-first-token (TTFT) across different prompt lengths and
prefix sharing ratios at batch size 1.  The vLLM UDP server is
automatically spawned in a separate process so that measurements
include the full UDP network-stack overhead.

Results are saved as CSV for downstream analysis and plotting.
"""

import argparse
import asyncio
import csv
import logging
import multiprocessing as mp
import random
import time
from dataclasses import asdict, dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path

from kvswitch.utils.logger import setup_logging
from kvswitch.vllm.client import VLLMClient

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "prompt_tokens",
    "prefix_ratio",
    "trial",
    "ttft",
    "engine_ttft",
    "num_cached_tokens",
]


@dataclass
class ProfilingConfig:
    """Configuration for a TTFT profiling run."""

    prompt_lengths: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024, 2048, 4096]
    )
    prefix_ratios: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    num_trials: int = 5
    max_output_tokens: int = 1
    output_path: Path = Path("results/profiling/ttft_traces.csv")
    seed: int = 42
    token_id_min: int = 100
    token_id_max: int = 10000


@dataclass
class ProfilingResult:
    """Single TTFT measurement."""

    prompt_tokens: int
    prefix_ratio: float
    trial: int
    ttft: float
    engine_ttft: float | None
    num_cached_tokens: int


# ---------------------------------------------------------------------------
# Prompt generation helpers
# ---------------------------------------------------------------------------


def generate_token_ids(
    length: int,
    seed: int,
    token_id_min: int = 100,
    token_id_max: int = 10000,
) -> list[int]:
    """Generate deterministic random token IDs of the given length."""
    rng = random.Random(seed)
    return [rng.randint(token_id_min, token_id_max) for _ in range(length)]


def build_prompt_pair(
    prompt_length: int,
    prefix_ratio: float,
    config: ProfilingConfig,
    trial: int,
) -> tuple[list[int], list[int]]:
    """Build a `(prime_tokens, measure_tokens)` pair.

    Both prompts are always `prompt_length` tokens long. The two prompts
    share `prefix_ratio` of their tokens. The priming prompt is sent first
    to populate the cache; the measurement prompt is the one whose TTFT we
    record.

    When `prefix_ratio == 0`, the prompts simply share no prefix.
    `run_profiling` skips the priming request in that case because there is
    no cacheable overlap to establish.
    """
    prefix_len = int(prompt_length * prefix_ratio)
    suffix_len = prompt_length - prefix_len

    shared_prefix = generate_token_ids(
        prefix_len,
        seed=config.seed,
        token_id_min=config.token_id_min,
        token_id_max=config.token_id_max,
    )

    # Two distinct suffixes so the requests are different.
    prime_suffix = generate_token_ids(
        suffix_len,
        seed=config.seed + trial * 2 + 1,
        token_id_min=config.token_id_min,
        token_id_max=config.token_id_max,
    )
    measure_suffix = generate_token_ids(
        suffix_len,
        seed=config.seed + trial * 2 + 2,
        token_id_min=config.token_id_min,
        token_id_max=config.token_id_max,
    )

    prime_tokens = shared_prefix + prime_suffix
    measure_tokens = shared_prefix + measure_suffix
    return prime_tokens, measure_tokens


# ---------------------------------------------------------------------------
# Server process management
# ---------------------------------------------------------------------------


def _run_server(host: str, port: int, engine_args: list[str]) -> None:
    """Multiprocessing target: parse engine CLI args and run the UDP server.

    All heavy imports (vLLM, torch) happen inside this function so that the
    parent process never touches CUDA.
    """
    import asyncio
    import logging

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from kvswitch.vllm.server import run_server

    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(["--host", host, "--port", str(port)] + engine_args)
    assert args is not None
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    asyncio.run(run_server(args))


def start_server_process(
    host: str,
    port: int,
    engine_args: list[str],
) -> BaseProcess:
    """Spawn the vLLM UDP server in a child process."""
    # Use "spawn" to get a clean process without inherited CUDA state.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_server, args=(host, port, engine_args))
    proc.start()
    logger.info("Started server process (pid=%s)", proc.pid)
    return proc


def stop_server_process(proc: BaseProcess) -> None:
    """Gracefully stop the server process."""
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=10)
    if proc.is_alive():
        logger.warning("Server did not exit after terminate, killing")
        proc.kill()
        proc.join()
    logger.info("Server process stopped (exitcode=%s)", proc.exitcode)


async def wait_for_server(
    client: VLLMClient,
    proc: BaseProcess,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> None:
    """Poll the health endpoint until the server is ready."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not proc.is_alive():
            raise RuntimeError(
                f"Server process exited early (exitcode={proc.exitcode})"
            )
        try:
            resp = await client.health()
            if resp.get("status") == "ok":
                logger.info("Server is ready")
                return
        except Exception:
            pass
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Server not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Measurement helpers (via UDP client)
# ---------------------------------------------------------------------------


async def measure_ttft(
    client: VLLMClient,
    prompt_token_ids: list[int],
    max_tokens: int = 1,
    temperature: float = 0.0,
) -> tuple[float, float | None, int]:
    """Send one request via UDP and return (wall_ttft, engine_ttft, num_cached)."""
    t_start = time.perf_counter()
    resp = await client.generate_tokens(
        prompt_token_ids,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    wall_ttft = time.perf_counter() - t_start

    engine_ttft: float | None = None
    metrics = resp.get("metrics")
    if isinstance(metrics, dict):
        raw = metrics.get("first_token_latency")
        if raw is not None:
            engine_ttft = float(raw)

    num_cached = resp.get("num_cached_tokens") or 0

    return wall_ttft, engine_ttft, int(num_cached)


async def run_profiling(
    client: VLLMClient,
    config: ProfilingConfig,
) -> list[ProfilingResult]:
    """Run the full profiling sweep and return results."""
    results: list[ProfilingResult] = []

    total = len(config.prompt_lengths) * len(config.prefix_ratios) * config.num_trials
    done = 0

    for prompt_length in config.prompt_lengths:
        for prefix_ratio in config.prefix_ratios:
            for trial in range(config.num_trials):
                # Reset prefix cache before each measurement.
                await client.reset()

                prime_tokens, measure_tokens = build_prompt_pair(
                    prompt_length,
                    prefix_ratio,
                    config,
                    trial,
                )

                # Prime the cache if there is a shared prefix.
                if prefix_ratio > 0 and len(prime_tokens) > 0:
                    await client.generate_tokens(
                        prime_tokens,
                        max_tokens=config.max_output_tokens,
                        temperature=0.0,
                    )

                # Measure TTFT.
                wall_ttft, engine_ttft, num_cached = await measure_ttft(
                    client,
                    measure_tokens,
                    max_tokens=config.max_output_tokens,
                )

                results.append(
                    ProfilingResult(
                        prompt_tokens=prompt_length,
                        prefix_ratio=prefix_ratio,
                        trial=trial,
                        ttft=wall_ttft,
                        engine_ttft=engine_ttft,
                        num_cached_tokens=num_cached,
                    )
                )

                done += 1
                logger.info(
                    "[%d/%d] prompt=%d prefix=%.0f%% trial=%d ttft=%.4fs cached=%d",
                    done,
                    total,
                    prompt_length,
                    prefix_ratio * 100,
                    trial,
                    wall_ttft,
                    num_cached,
                )

    return results


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------


def save_results_csv(results: list[ProfilingResult], path: Path) -> None:
    """Write profiling results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # Replace None with empty string for CSV.
            row = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(row)
    logger.info("Saved %d results to %s", len(results), path)


def load_results_csv(path: Path) -> list[ProfilingResult]:
    """Load profiling results from a CSV file."""
    results: list[ProfilingResult] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                ProfilingResult(
                    prompt_tokens=int(row["prompt_tokens"]),
                    prefix_ratio=float(row["prefix_ratio"]),
                    trial=int(row["trial"]),
                    ttft=float(row["ttft"]),
                    engine_ttft=(
                        float(row["engine_ttft"]) if row["engine_ttft"] else None
                    ),
                    num_cached_tokens=int(row["num_cached_tokens"]),
                )
            )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def async_main(
    host: str,
    port: int,
    config: ProfilingConfig,
    engine_args: list[str],
    server_timeout: float,
    client_timeout: float,
) -> None:
    """Spawn the server, run profiling, and save results."""
    proc = start_server_process(host, port, engine_args)
    try:
        client = VLLMClient(host=host, port=port, timeout=client_timeout)
        await wait_for_server(client, proc, timeout=server_timeout)
        results = await run_profiling(client, config)
        save_results_csv(results, config.output_path)
    finally:
        stop_server_process(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TTFT profiling for vLLM (via UDP server)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default="128,256,512,1024,2048,4096",
        help="Comma-separated list of prompt lengths in tokens",
    )
    parser.add_argument(
        "--prefix-ratios",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated list of prefix sharing ratios",
    )
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/profiling/ttft_traces.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the server to start",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds",
    )

    # Everything not recognised by *this* parser is forwarded to the server
    # (e.g. --model, --enable-prefix-caching, --max-num-seqs, etc.).
    args, engine_args = parser.parse_known_args()

    setup_logging(args.log_level)
    logger.info("Profiling args: %s", args)
    logger.info("Engine args (forwarded to server): %s", engine_args)

    config = ProfilingConfig(
        prompt_lengths=[int(x) for x in args.prompt_lengths.split(",")],
        prefix_ratios=[float(x) for x in args.prefix_ratios.split(",")],
        num_trials=args.num_trials,
        max_output_tokens=args.max_output_tokens,
        output_path=Path(args.output_path),
        seed=args.seed,
    )

    asyncio.run(
        async_main(
            host=args.host,
            port=args.port,
            config=config,
            engine_args=engine_args,
            server_timeout=args.server_timeout,
            client_timeout=args.client_timeout,
        )
    )
