# KVSwitch: Accelerating Distributed LLM Inference with In-Network Prefix-Aware Routing

## Running Experiments

Experiments run inside Docker with a pre-built Mininet + BMv2 image. **No
local dependency installation is required** — Docker pulls the image
automatically on first run.

### Prerequisites

- Docker (with `--privileged` support)
- Compiled P4 artifacts in `build/p4/kvswitch/` (included in the repo)
- ShareGPT dataset at `data/ShareGPT_V3_unfiltered_cleaned_split.json`
  ([download](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json))
- HuggingFace model cache (for tokenizer; downloaded on first run)

### Run all experiments

```bash
bash exp/run_exp.sh
```

Results are saved to `results/exp/`.

### Run specific experiments

```bash
bash exp/run_exp.sh 1          # Microbenchmark: routing overhead
bash exp/run_exp.sh 2          # End-to-end: rate sweep
bash exp/run_exp.sh 3a         # Ablation: ECMP vs pinning
bash exp/run_exp.sh 3b         # Ablation: warm-up impact
bash exp/run_exp.sh 4a         # Sensitivity: prefix sharing ratio
bash exp/run_exp.sh 4b         # Sensitivity: resource constraints
bash exp/run_exp.sh 2 4b       # Multiple experiments
```

### Run a single evaluation

```bash
bash exp/run_eval.sh --baselines l4_ecmp,l7_rr,l7_pa,kvswitch \
  --num-requests 200 --request-rate 10
```

### Rebuild the Docker image

Pass `--build` to `run_eval.sh` to recompile the P4 program and rebuild
the Docker image:

```bash
bash exp/run_eval.sh --build --baselines kvswitch --num-requests 50
```

### Recompile P4 artifacts

Pre-compiled artifacts are committed in the repo for zero-setup
reproduction. If you modify `p4/`, recompile and recommit:

```bash
bash scripts/compile_p4.sh p4/kvswitch.p4 build/p4/kvswitch
```

The script uses a locally installed `p4c` if available, otherwise falls
back to a `p4c` Docker image (built from [p4lang/p4c](https://github.com/p4lang/p4c),
tagged as `p4c:latest`).

## Development Setup

Local installation is only needed for development (editing code, running
tests, profiling). **This is not required to run experiments**, which use
Docker.

```bash
bash scripts/install.sh
```

This creates a Python 3.12 virtual environment in `.venv`, installs all
dependencies (including vLLM for GPU profiling), and may take a long time
due to the vLLM build.

### Run tests

```bash
uv run pytest tests/ -q
```

### Lint and format

```bash
uv run bash scripts/format.sh --all
```
