# Prompt P02 — uv + RTX 5090/PyTorch environment validation

Goal:
Validate the pg3d uv environment on an RTX 5090 workstation.

Context to read first:
- `AGENTS.md`
- `docs/runbooks/setup_workstation.md`
- `docs/runbooks/commands.md`
- `docs/adr/0006-python-uv-cuda-stack.md`
- `pyproject.toml`
- `scripts/check_gpu.py`

Constraints:
- Do not downgrade Python or PyTorch without proposing an ADR update first.
- Do not let DP3 install an older torch build.
- Do not install system packages without asking.

Tasks:
1. Run/inspect `uv sync --extra cu129 --group dev`.
2. Run `make gpu-check`, `make test`, `make lint`, and `make smoke`.
3. If PyTorch is CPU-only or CUDA is unavailable, diagnose whether the issue is uv index config, Python version, driver, or torch version.
4. Update `docs/runbooks/setup_workstation.md` and `docs/runbooks/commands.md` with the working commands.
5. Update `docs/status.md` and add a worklog entry.

Done when:
- GPU smoke test succeeds, or the blocker is documented with next steps.
