"""Hydra entry: Libero LEWM embedding prediction evaluation."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

_LEWM_ROOT = Path(__file__).resolve().parent.parent
if str(_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEWM_ROOT))

from eval.libero_episodes import (
    apply_transform_to_episode,
    encode_pixels_in_chunks,
    episode_dict_to_batch,
    list_episode_indices,
    load_raw_episode,
)
from eval.metrics import all_metrics
from eval.model_builder import build_jepa_model, load_jepa_checkpoint
from eval.rollout import rollout_full_episode_to_last, rollout_n_steps_open_loop
from train import build_dataset
from utils import get_column_normalizer, get_img_preprocessor


def _setup_libero_dataset(cfg: DictConfig):
    dataset = build_dataset(cfg.data.dataset)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    return dataset


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text_summary(
    path: Path,
    full_ep: Dict[str, float],
    n_step: Dict[int, Dict[str, float]],
    meta: Dict[str, Any],
) -> None:
    lines = [
        "LEWM Libero embedding evaluation",
        f"episodes_used: {meta.get('episodes_used')}",
        f"checkpoint: {meta.get('checkpoint')}",
        "",
        "[full_episode] pred vs encode(last_frame)",
    ]
    for k in ("mse", "mae", "l2", "cosine_dist"):
        lines.append(f"  {k}: {full_ep.get(k, float('nan'))}")
    lines.append("")
    lines.append("[n_step_open_loop] mean over (episode, start_t) pairs")
    for n in sorted(n_step.keys()):
        lines.append(f"  n={n}")
        for k in ("mse", "mae", "l2", "cosine_dist"):
            lines.append(f"    {k}: {n_step[n].get(k, float('nan'))}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _zero_metric_dict() -> Dict[str, float]:
    return {"mse": 0.0, "mae": 0.0, "l2": 0.0, "cosine_dist": 0.0}


def _resolved_n_step_ns(cfg: DictConfig) -> List[int]:
    """任务2 的开环长度 n 列表：n_step_list 非空则用其整数；否则 1..n_step_max。"""
    raw = cfg.get("n_step_list")
    if raw is None:
        nmx = int(cfg.get("n_step_max", 10))
        return list(range(1, nmx + 1))
    resolved = OmegaConf.to_container(raw, resolve=True)
    if resolved is None or resolved == [] or resolved == ():
        nmx = int(cfg.get("n_step_max", 10))
        return list(range(1, nmx + 1))
    nums = sorted({int(x) for x in resolved})
    if not nums or min(nums) < 1:
        raise ValueError("n_step_list must contain positive integers.")
    return nums


@hydra.main(version_base=None, config_path="../config/eval", config_name="lewm_libero")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_s = cfg.get("device", "cuda")
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")

    out_root = Path(cfg.output_dir).resolve()
    _ensure_output_dir(out_root)
    with open(out_root / "run_config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    dataset = _setup_libero_dataset(cfg)
    model = build_jepa_model(cfg).to(device)
    model.eval()

    ckpt = str(cfg.checkpoint)
    load_jepa_checkpoint(model, ckpt, device)

    eval_task1 = bool(cfg.get("eval_task1", True))
    eval_task2 = bool(cfg.get("eval_task2", True))
    if not eval_task1 and not eval_task2:
        raise ValueError("eval_task1 与 eval_task2 不能同时为 false")

    h = int(cfg.wm.history_size)
    n_step_ns = _resolved_n_step_ns(cfg)
    chunk_t = int(cfg.encode_chunk_t)
    n_step_max_cfg = int(cfg.n_step_max)

    episodes = list_episode_indices(dataset)
    if cfg.get("max_episodes") is not None:
        episodes = episodes[: int(cfg.max_episodes)]

    print(
        "[lewm_eval] Libero parquet 评测：LEWM/JEPA "
        "图像序列编码 + 动力学 rollout，对 embedding 做回归指标。"
    )
    print(
        f"  split={cfg.data.dataset.split}  root={cfg.data.dataset.root}  "
        f"episodes={len(episodes)}  history_size={h}  "
        f"encode_chunk_t={chunk_t}"
    )
    print(
        f"  eval_task1={eval_task1}  eval_task2={eval_task2}  "
        f"n_step_ns={n_step_ns}  (n_step_max={n_step_max_cfg}，仅当 n_step_list=null 时用于生成 1..n)"
    )
    print(
        "  任务1 full_episode: 从历史窗口自回归 rollout 到整条 trajectory 的最后一帧，"
        "pred 与 encode(最后一帧) 对比。"
    )
    print(
        "  任务2 n_step_open_loop: 对每个 episode 的起点 t，开环 n 步预测，"
        "与对应时刻 embedding 对比；均值在 (episode, start_t) 对上；"
        f"max_starts_per_episode={cfg.get('max_starts_per_episode')}"
    )
    progress_every = bool(cfg.get("progress_every_episode", True))
    print(
        f"  进度：progress_every_episode={progress_every}；"
        "全部结束后 stdout 再打 JSON 汇总，明细见 output_dir。"
    )

    def mean_dict(acc: Dict[str, float], c: int) -> Dict[str, float]:
        if c <= 0:
            return {k: float("nan") for k in _zero_metric_dict()}
        return {k: acc[k] / c for k in acc}

    # --- accumulators ---
    full_ep_sum: Dict[str, float] = _zero_metric_dict()
    full_ep_count = 0

    nstep_sum: Dict[int, Dict[str, float]] = {
        n: _zero_metric_dict() for n in n_step_ns
    } if eval_task2 else {}
    nstep_count: Dict[int, int] = {n: 0 for n in n_step_ns} if eval_task2 else {}

    per_episode_full: List[Dict[str, Any]] = []
    n_eps = len(episodes)
    completed_eps = 0

    for ep_rank, ep_idx in enumerate(episodes, start=1):
        raw = load_raw_episode(dataset, ep_idx)
        tens = apply_transform_to_episode(raw, dataset.transform)
        batch = episode_dict_to_batch(tens, device)
        pixels = batch["pixels"]
        actions = batch["action"]
        l = pixels.size(1)
        if l < h:
            continue

        with torch.inference_mode():
            emb, act_emb = encode_pixels_in_chunks(model, pixels, actions, chunk_t)

        m: Dict[str, float] = {}
        if eval_task1:
            with torch.inference_mode():
                pred_last = rollout_full_episode_to_last(model, emb, act_emb, h)
                gt_last = emb[:, -1:]
                m = all_metrics(pred_last.squeeze(0), gt_last.squeeze(0))
            full_ep_count += 1
            for k in full_ep_sum:
                full_ep_sum[k] += m[k]
            per_episode_full.append(
                {"episode_index": int(ep_idx), "length": int(l), "metrics": m}
            )

        if eval_task2:
            starts = list(range(max(0, l - h)))
            if cfg.get("max_starts_per_episode") is not None:
                cap = int(cfg.max_starts_per_episode)
                if len(starts) > cap:
                    rng = random.Random(cfg.seed + ep_idx)
                    starts = rng.sample(starts, cap)

            ep_n_sum = {n: _zero_metric_dict() for n in n_step_ns}
            ep_n_cnt = {n: 0 for n in n_step_ns}

            for n in n_step_ns:
                valid_starts = [t for t in starts if t + h + n <= l]
                for t in valid_starts:
                    with torch.inference_mode():
                        pred = rollout_n_steps_open_loop(model, emb, act_emb, h, t, n)
                        gt = emb[:, t + h + n - 1 : t + h + n]
                        mm = all_metrics(pred.squeeze(0), gt.squeeze(0))
                    for kk in ep_n_sum[n]:
                        ep_n_sum[n][kk] += mm[kk]
                    ep_n_cnt[n] += 1
                    for kk in nstep_sum[n]:
                        nstep_sum[n][kk] += mm[kk]
                    nstep_count[n] += 1

        completed_eps += 1
        if progress_every:
            parts: List[str] = [
                f"done {completed_eps}  queued {ep_rank}/{n_eps}  "
                f"episode_index={int(ep_idx)}  len={int(l)}"
            ]
            if eval_task1:
                run_mse = full_ep_sum["mse"] / full_ep_count
                run_cos = full_ep_sum["cosine_dist"] / full_ep_count
                parts.append(
                    "task1 full_ep[last]: "
                    f"mse={m['mse']:.6g} mae={m['mae']:.6g} l2={m['l2']:.6g} "
                    f"cos_dist={m['cosine_dist']:.6g}  "
                    f"run_mean_mse={run_mse:.6g} run_mean_cos={run_cos:.6g}"
                )
            if eval_task2:
                cur_bits: List[str] = []
                run_bits: List[str] = []
                for n in n_step_ns:
                    c_ep = ep_n_cnt[n]
                    if c_ep > 0:
                        em = mean_dict(ep_n_sum[n], c_ep)
                        cur_bits.append(f"n{n}:mse={em['mse']:.4g}/cos={em['cosine_dist']:.4g}")
                    rc = nstep_count[n]
                    if rc > 0:
                        rm = mean_dict(nstep_sum[n], rc)
                        run_bits.append(f"n{n}:mse={rm['mse']:.4g}/cos={rm['cosine_dist']:.4g}")
                if cur_bits:
                    parts.append("task2 this_ep " + " ".join(cur_bits))
                if run_bits:
                    parts.append("task2 run_avg " + " ".join(run_bits))
            print("[lewm_eval] " + " | ".join(parts), flush=True)

    full_ep_mean = (
        mean_dict(full_ep_sum, full_ep_count)
        if eval_task1
        else {k: float("nan") for k in _zero_metric_dict()}
    )
    n_step_means = (
        {n: mean_dict(nstep_sum[n], nstep_count[n]) for n in n_step_ns}
        if eval_task2
        else {}
    )

    meta = {
        "checkpoint": ckpt,
        "episodes_used": completed_eps,
        "eval_task1": eval_task1,
        "eval_task2": eval_task2,
        "n_step_ns": n_step_ns,
        "split": cfg.data.dataset.split,
        "root": cfg.data.dataset.root,
    }
    report = {
        "meta": meta,
        "full_episode_mean": full_ep_mean,
        "full_episode_per_episode": per_episode_full[:200],
        "n_step_mean": {str(n): n_step_means[n] for n in n_step_ns} if eval_task2 else {},
        "n_step_counts": {str(n): nstep_count[n] for n in n_step_ns} if eval_task2 else {},
    }
    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_text_summary(
        out_root / "metrics.txt", full_ep_mean, n_step_means, meta
    )

    if cfg.wandb.enabled:
        try:
            import wandb

            wcfg = cfg.wandb
            run = wandb.init(
                project=wcfg.project,
                entity=wcfg.entity if wcfg.entity else None,
                name=wcfg.name,
                tags=list(wcfg.tags) if wcfg.get("tags") else None,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            log: Dict[str, Any] = {}
            if eval_task1:
                for k, v in full_ep_mean.items():
                    log[f"eval/full_episode/{k}"] = v
            if eval_task2:
                for n in n_step_ns:
                    for k, v in n_step_means[n].items():
                        log[f"eval/n_step/n_{n}/{k}"] = v
                    log[f"eval/n_step/n_{n}/num_pairs"] = nstep_count[n]
            log["eval/num_episodes"] = completed_eps
            wandb.log(log)
            run.finish()
        except Exception as e:
            print(f"[WARN] wandb logging failed: {e}")

    out_json: Dict[str, Any] = {}
    if eval_task1:
        out_json["full_episode_mean"] = full_ep_mean
    if eval_task2:
        out_json["n_step"] = n_step_means
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
