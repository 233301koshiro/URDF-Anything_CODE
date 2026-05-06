# CPU Docker Verification — URDF-Anything

This document records the CPU-only Docker verification performed for the repository and the workarounds applied so the project runs end-to-end on CPU.

## Overview
- Image tag used: `urdf_any_cpu:latest` (built by `./docker_build.sh`).
- Verification script: `./docker_run_test.sh` which runs `test_llava_cpu.py` inside the container and saves logs to `logs/`.
- Final verification output includes the marker `Verification Complete`.

## Commands
Build and run the verification (single command):

```bash
./docker_build.sh && ./docker_run_test.sh
```

To run only verification with an existing image:

```bash
./docker_run_test.sh
```

To open an interactive shell inside the verification container (for debugging):

```bash
./docker_run.sh --shell
```

## Where logs are saved
All test runs write logs to the `logs/` folder in the repository root. Example:

- `logs/docker_test_20260501_171351.log` (contains successful run and `Verification Complete`).

## Workarounds & reasons
During CPU verification the following pragmatic workarounds were implemented to allow reliable CPU-only execution and fast iterative debugging. They are recorded here so maintainers can decide whether to keep, refine or remove them.

1. `pointnet2_ops` wrapper
   - Problem: original project expects a compiled C++/CUDA extension `pointnet2_ops` which is not trivially available on CPU-only environments.
   - Action: added a lightweight Python package `pointnet2_ops` that imports the repository's pure-PyTorch `pointnet2_utils` implementation.
   - Status: works for CPU verification. Long-term: consider compiling the native extension in CI or upstreaming the pure-PyTorch fallback.

2. Hugging Face registration guards
   - Problem: repeated runs re-register a custom HF config named `llava`, causing `ValueError` on duplicate registration in long-lived processes or test loops.
   - Action: wrapped `AutoConfig.register` and `AutoModelForCausalLM.register` calls in a `try/except ValueError` to avoid crashes during iterative development.
   - Status: safe as a development workaround; consider removing once registration is centralized.

3. `transformers` pin
   - Problem: API mismatches (internal helpers like `_expand_mask`) between different `transformers` versions caused import errors.
   - Action: pinned `transformers==4.31.0` in `requirements.cpu.txt` used by the Docker image.

4. Symlink for `ReConV2`
   - Problem: some scripts expect `ReConV2/` relative to the project root while code lives in `model/ReConV2/`.
   - Action: created `ReConV2 -> model/ReConV2` symlink at repository root.

## Recommendations
- Keep `docs/VERIFICATION.md` updated with any further changes to the verification flow.
- If you plan CI runs, either build the `pointnet2_ops` native extension there or vendor the pure-PyTorch fallback upstream.
- Once CI is stable, remove the `try/except` registration guards or guard them behind an initialization flag.

## Reproduce locally (quick)
1. Build the image and run verification:

```bash
./docker_build.sh && ./docker_run_test.sh
```

2. Inspect the latest log file in `logs/` for `Verification Complete`.

---

Generated on 2026-05-01.
