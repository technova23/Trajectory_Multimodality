# pg3d Codex Starter Pack

This directory contains a Codex-ready documentation and scaffold pack for the `pg3d` project.

Copy the files into the root of a fresh private GitHub repository named `pg3d`, then fill in organization-specific details such as GitHub org/user names, local paths, and CoppeliaSim install paths.

Recommended first command after copying:

```bash
uv sync --extra cu129 --group dev
make test
make gpu-check
```

The pack intentionally focuses on documentation, prompts, and lightweight scaffold files. It does not vendor DP3/RLBench/PyRep code; use the mirroring/submodule instructions in `docs/runbooks/dependency_mirroring.md`.
