import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


class LiberoParquetDataset(Dataset):
    """LIBERO parquet dataset adapter for LeWM training."""

    def __init__(
        self,
        root: str,  # 数据集根目录，内部应包含 meta/ 与 data/ 子目录
        num_steps: int,  # 每个训练样本的时间步长度 T（会按 frameskip 抽帧）
        frameskip: int,  # 相邻采样时间步之间跨越的底层帧数
        keys_to_load: List[str],  # 需要加载的逻辑键名，例如 pixels/action/state
        split: str = "train",  # 使用的数据划分名，对应 info.json 里的 splits 字段
        transform=None,  # 样本后处理函数，输入/输出均为样本字典
        pixel_key: str = "image",  # parquet 中图像列名（用于映射逻辑键 pixels）
        key_mapping: Optional[Dict[str, str]] = None,  # 额外键映射：逻辑键 -> parquet 实际列名
        max_episode_cache: int = 8,  # 内存中缓存的 episode 数量上限（LRU）
    ):
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.keys_to_load = list(keys_to_load)
        self.split = split
        self.transform = transform
        self.pixel_key = pixel_key
        self.max_episode_cache = int(max_episode_cache)

        self.key_mapping = {
            "pixels": self.pixel_key,
            "action": "actions",
            "state": "state",
        }
        
        if key_mapping is not None:
            self.key_mapping.update(key_mapping)

        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0.")
        if self.frameskip <= 0:
            raise ValueError("frameskip must be > 0.")

        self.info = self._read_info()
        self.stats = self._read_stats()
        self.episodes = self._read_episodes()
        self.episode_indices = self._resolve_split_indices()
        self.sample_index = self._build_sample_index()
        self._episode_cache: "OrderedDict[int, Dict[str, list]]" = OrderedDict()
        self._column_cache: Dict[str, np.ndarray] = {}

    def _read_info(self) -> dict:
        with open(self.meta_dir / "info.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_stats(self) -> dict:
        stats_path = self.meta_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _read_episodes(self) -> Dict[int, dict]:
        episodes: Dict[int, dict] = {}
        with open(self.meta_dir / "episodes.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                episodes[int(ep["episode_index"])] = ep
        return episodes

    def _resolve_split_indices(self) -> List[int]:
        split_spec = self.info.get("splits", {}).get(self.split)
        if split_spec is None:
            raise KeyError(f"Split '{self.split}' not found in info.json.")
        start_s, end_s = split_spec.split(":")
        start, end = int(start_s), int(end_s)
        return list(range(start, end))

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        # We need enough frames for pixels/state at step offsets and enough
        # low-level actions to build a frameskip-sized action chunk per step.
        horizon = self.num_steps * self.frameskip - 1
        for ep_idx in self.episode_indices:
            length = int(self.episodes[ep_idx]["length"])
            max_start = length - 1 - horizon
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                samples.append((ep_idx, start))
        return samples

    def _episode_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        return self.data_dir / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    def _load_episode(self, episode_index: int) -> Dict[str, list]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        parquet_path = self._episode_path(episode_index)
        raw_columns = [self.key_mapping.get(k, k) for k in self.keys_to_load]
        table = pq.read_table(parquet_path, columns=raw_columns)
        episode = {col: table[col].to_pylist() for col in table.column_names}

        self._episode_cache[episode_index] = episode
        self._episode_cache.move_to_end(episode_index)
        if len(self._episode_cache) > self.max_episode_cache:
            self._episode_cache.popitem(last=False)
        return episode

    @staticmethod
    def _decode_image_bytes(image_cell: dict) -> np.ndarray:
        img_bytes = image_cell["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.sample_index[idx]
        episode = self._load_episode(ep_idx)
        offsets = [start + i * self.frameskip for i in range(self.num_steps)]

        out = {}
        for key in self.keys_to_load:
            col = self.key_mapping.get(key, key)
            if key == "action":
                # Match LeWM's expected action shape: (T, frameskip * action_dim)
                # where each step stacks the consecutive low-level actions.
                action_chunks = []
                for base in offsets:
                    chunk = episode[col][base : base + self.frameskip]
                    action_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                arr = np.stack(action_chunks, axis=0)
            else:
                values = [episode[col][i] for i in offsets]
                if key == "pixels":
                    arr = np.stack([self._decode_image_bytes(v) for v in values], axis=0)
                else:
                    arr = np.asarray(values, dtype=np.float32)
            out[key] = arr

        if self.transform is not None:
            out = self.transform(out)
        return out

    def get_dim(self, key: str) -> int:
        mapped = self.key_mapping.get(key, key)
        feature = self.info["features"].get(mapped)
        if feature is None:
            raise KeyError(f"Feature '{mapped}' not found in info.json.")
        shape = feature.get("shape", [])
        if len(shape) == 0:
            return 1
        if mapped in ("image", "wrist_image"):
            return int(shape[-1])
        return int(shape[-1])

    def get_col_data(self, key: str) -> np.ndarray:
        mapped = self.key_mapping.get(key, key)
        if mapped in self._column_cache:
            return self._column_cache[mapped]

        data = []
        for ep_idx in self.episode_indices:
            parquet_path = self._episode_path(ep_idx)
            table = pq.read_table(parquet_path, columns=[mapped])
            col_data = table[mapped].to_pylist()
            if mapped in ("image", "wrist_image"):
                continue
            data.append(np.asarray(col_data, dtype=np.float32))

        if len(data) == 0:
            raise ValueError(f"No numeric data available for key '{key}'.")
        merged = np.concatenate(data, axis=0)
        self._column_cache[mapped] = merged
        return merged