from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from pg3d.policies.dp3 import ReachDatasetConfig, ReachSequenceDataset, SimpleDP3
from pg3d.policies.dp3.normalizer import LinearNormalizer
from pg3d.policies.dp3.reach_dataset import reach_shape_meta
from pg3d.policies.dp3.utils import dict_apply


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    dataset = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=args.dataset,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_train_episodes=args.max_train_episodes,
        ),
        split="train",
    )
    if len(dataset) == 0:
        raise RuntimeError("training dataset has no sequences")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    policy_kwargs = _policy_kwargs(args, shape_meta=dataset.shape_meta)
    policy = SimpleDP3(**policy_kwargs)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = _init_wandb(args, dataset=dataset, policy_kwargs=policy_kwargs)

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            step += 1
            batch = _batch_to(batch, device)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.compute_loss(batch)
            loss.backward()
            grad_norm = _grad_norm(policy)
            optimizer.step()
            metrics = {
                "train/bc_loss": float(loss_dict["bc_loss"]),
                "train/grad_norm": grad_norm,
                "train/action_rms": float(batch["action"].detach().pow(2).mean().sqrt().cpu()),
                "train/point_cloud_mean": float(
                    batch["obs"]["point_cloud"].detach().mean().cpu()
                ),
                "train/step": step,
            }
            print(
                f"step={step} bc_loss={metrics['train/bc_loss']:.6f} "
                f"grad_norm={metrics['train/grad_norm']:.6f}"
            )
            if run is not None:
                _wandb_log(run, metrics, batch=batch, step=step, log_histograms=args.log_histograms)
            if step >= args.max_steps:
                break

    if args.checkpoint_out is not None:
        _save_checkpoint(args.checkpoint_out, policy, optimizer, policy_kwargs, args)
        print(f"saved checkpoint: {args.checkpoint_out}")
    if run is not None:
        run.finish()
    print(
        "summary: "
        + json.dumps(
            {
                "dataset": str(args.dataset),
                "num_sequences": len(dataset),
                "num_episodes": dataset.num_episodes,
                "max_steps": args.max_steps,
                "device": str(device),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke-scale pg3d-native DP3 training loop on a reach Zarr dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--encoder-output-dim", type=int, default=32)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="pg3d")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-required", action="store_true")
    parser.add_argument("--log-histograms", action="store_true")
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    return args


def _policy_kwargs(
    args: argparse.Namespace,
    *,
    shape_meta: dict[str, dict[str, dict[str, list[int]]]] | None = None,
) -> dict[str, Any]:
    shape_meta = shape_meta or reach_shape_meta()
    return {
        "shape_meta": shape_meta,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "encoder_output_dim": args.encoder_output_dim,
        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
        "down_dims": tuple(args.down_dims),
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "pointcloud_encoder_cfg": {
            "out_channels": args.encoder_output_dim,
            "use_layernorm": True,
            "final_norm": "layernorm",
        },
    }


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is false")
    return torch.device(value)


def _batch_to(batch: Any, device: torch.device) -> Any:
    return dict_apply(batch, lambda tensor: tensor.to(device=device, dtype=torch.float32))


def _grad_norm(policy: torch.nn.Module) -> float:
    total = 0.0
    for param in policy.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def _init_wandb(
    args: argparse.Namespace,
    *,
    dataset: ReachSequenceDataset,
    policy_kwargs: dict[str, Any],
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": str(args.dataset),
                "num_sequences": len(dataset),
                "num_episodes": dataset.num_episodes,
                "policy": _jsonable(policy_kwargs),
                "command": "scripts/train_dp3_reach.py",
            },
        )
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B init failed, continuing without W&B: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _wandb_log(
    run: Any,
    metrics: dict[str, float],
    *,
    batch: Any,
    step: int,
    log_histograms: bool,
) -> None:
    if log_histograms:
        import wandb

        metrics = {
            **metrics,
            "viz/action_hist": wandb.Histogram(batch["action"].detach().cpu().numpy()),
            "viz/agent_pos_hist": wandb.Histogram(
                batch["obs"]["agent_pos"].detach().cpu().numpy()
            ),
            "viz/point_cloud_hist": wandb.Histogram(
                batch["obs"]["point_cloud"].detach().cpu().numpy()
            ),
        }
    run.log(metrics, step=step)


def _save_checkpoint(
    path: Path,
    policy: SimpleDP3,
    optimizer: torch.optim.Optimizer,
    policy_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        key: value.detach().cpu()
        for key, value in policy.state_dict().items()
        if not key.startswith("normalizer.")
    }
    torch.save(
        {
            "model": model_state,
            "normalizer": {
                key: value.detach().cpu() for key, value in policy.normalizer.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "policy_kwargs": _jsonable(policy_kwargs),
            "args": vars(args),
        },
        path,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def load_reach_policy_from_checkpoint(path: Path, *, device: torch.device) -> SimpleDP3:
    """Load a DP3 reach checkpoint written by this smoke trainer."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = SimpleDP3(**checkpoint["policy_kwargs"])
    policy.set_normalizer(LinearNormalizer.from_state_dict(checkpoint["normalizer"]))
    policy.load_state_dict(checkpoint["model"], strict=False)
    policy.to(device)
    policy.eval()
    return policy


if __name__ == "__main__":
    raise SystemExit(main())
