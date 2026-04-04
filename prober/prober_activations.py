from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformer_lens import utils

ROOT_DIR = Path(__file__).resolve().parents[1]
V2_DIR = ROOT_DIR / "dataset"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from metadata import FIM_TOKENS_BY_MODEL  # noqa: E402


def _resolve_act_name(act: str, layer: Optional[int]) -> str:
    if layer is None:
        return act
    if act == "resid_pre":
        return utils.get_act_name("resid_pre", layer)
    if act == "resid_post":
        return utils.get_act_name("resid_post", layer)
    if act == "attn_out":
        return utils.get_act_name("attn_out", layer)
    if act == "mlp_out":
        return utils.get_act_name("mlp_out", layer)
    if act == "z":
        return utils.get_act_name("z", layer)
    return utils.get_act_name(act, layer)


def _pool_activation(
    act: torch.Tensor, pool: str, token_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if act.ndim == 2:
        return act
    if pool == "avg":
        if act.ndim == 3:
            return act.mean(dim=1)
        if act.ndim == 4:
            pooled = act.mean(dim=1)
            return pooled.reshape(pooled.shape[0], -1)
        raise ValueError(f"Unsupported activation shape for avg pooling: {act.shape}")
    if pool == "last":
        if act.ndim == 3:
            return act[:, -1, :]
        if act.ndim == 4:
            last = act[:, -1, :, :]
            return last.reshape(last.shape[0], -1)
        raise ValueError(f"Unsupported activation shape for last pooling: {act.shape}")
    if pool == "fim":
        if token_indices is None:
            raise ValueError("FIM pooling requires token indices.")
        idx = token_indices.to(act.device)
        if act.ndim == 3:
            return act[torch.arange(act.shape[0], device=act.device), idx, :]
        if act.ndim == 4:
            last = act[torch.arange(act.shape[0], device=act.device), idx, :, :]
            return last.reshape(last.shape[0], -1)
        raise ValueError(f"Unsupported activation shape for fim pooling: {act.shape}")
    raise ValueError(f"Unknown pooling strategy: {pool}")


def _find_last_subsequence_indices(toks: torch.Tensor, seq: List[int], pad_id: Optional[int]) -> torch.Tensor:
    batch_size, seq_len = toks.shape
    seq_len_pat = len(seq)
    if seq_len_pat == 0:
        return torch.zeros(batch_size, dtype=torch.long, device=toks.device)
    indices = []
    for b in range(batch_size):
        row = toks[b].tolist()
        last_idx = None
        for i in range(0, seq_len - seq_len_pat + 1):
            if row[i : i + seq_len_pat] == seq:
                last_idx = i + seq_len_pat - 1
        if last_idx is None:
            if pad_id is None:
                last_idx = seq_len - 1
            else:
                non_pad = [i for i, t in enumerate(row) if t != pad_id]
                last_idx = non_pad[-1] if non_pad else 0
        indices.append(last_idx)
    return torch.tensor(indices, dtype=torch.long, device=toks.device)


def _fim_token_indices(*, toks: torch.Tensor, tokenizer, model_name: str) -> torch.Tensor:
    fim = FIM_TOKENS_BY_MODEL.get(model_name)
    if fim is None:
        fim = next(iter(FIM_TOKENS_BY_MODEL.values()))
    middle = fim["middle"]
    seq = tokenizer.encode(middle, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id
    return _find_last_subsequence_indices(toks, seq, pad_id)
