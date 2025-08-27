import torch
import torch.nn as nn
import pointops

class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )

def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
    return torch.repeat_interleave(
        torch.arange(len(offset), device=offset.device), counts
    ).long()

@torch.no_grad()
def build_inst_knn_idx_by_sort(
    xyz: torch.Tensor,          # (N,3) float32 CUDA
    offset: torch.Tensor,       # (B,)   int (cumulative) CUDA
    inst_label: torch.Tensor,   # (N,)   int  (>=0 valid, -1 ignore)
    nsample: int = 16
) -> torch.Tensor:
    """
    Return (N, nsample) int32 CUDA: kNN indices within the *same instance*.
    Logic mirrors the old dataset-side inst_knn_same but works for batched N.

    - Points are lex-sorted by (batch, inst), invalid insts (-1) dropped.
    - Build per-(batch,inst) offsets and call pointops.knn_query.
    - Map gathered indices back to the original ordering.
    """
    device = xyz.device
    N = xyz.shape[0]
    out = torch.full((N, nsample), -1, device=device, dtype=torch.int32)
    if inst_label is None:
        return out

    inst = inst_label.to(device=device, dtype=torch.long)
    valid = inst >= 0
    if not torch.any(valid):
        return out

    bid = _offset2batch(offset).to(device)      # (N,)

    # make a stable lexicographic key without giant constants
    inst_clip = inst.clamp_min(0)
    R = int(inst_clip.max().item()) + 1
    key = bid * R + inst_clip                   # unique per (batch, inst)

    # sort and keep only valid
    order = torch.argsort(key)
    order = order[valid[order]]

    xyz_s   = xyz[order].contiguous()
    bid_s   = bid[order]
    inst_s  = inst_clip[order]

    # boundaries where (batch,inst) changes
    change = ((bid_s[1:] != bid_s[:-1]) | (inst_s[1:] != inst_s[:-1])).nonzero(as_tuple=False).squeeze(1) + 1
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), change])
    counts = torch.diff(torch.cat([starts, torch.tensor([order.numel()], device=device, dtype=torch.long)]))

    # cumulative counts -> offsets for pointops
    grp_offset = counts.to(torch.int32).cumsum(0)

    # kNN within each (batch,inst) group
    idx_s, _ = pointops.knn_query(nsample, xyz_s, grp_offset, xyz_s, grp_offset)  # (Nv, nsample)

    # map indices back into original indexing space
    back = order[idx_s.long()]                # (Nv, nsample)
    out[order] = back.to(torch.int32)
    return out

@torch.no_grad()
def build_inst_knn_idx_gpu(xyz: torch.Tensor,
                           offset: torch.Tensor,
                           inst_label: torch.Tensor,
                           nsample: int = 16) -> torch.Tensor:
    """
    xyz        : (N,3) float32 CUDA
    offset     : (B,)  int32/64 CUDA (cumulative counts)
    inst_label : (N,)  int32/64 (>=0 valid, -1 ignore)
    return     : (N, nsample) int32 CUDA with neighbours *within the same instance* (−1 padded)
    """
    device = xyz.device
    N = xyz.shape[0]
    out = torch.full((N, nsample), -1, device=device, dtype=torch.int32)
    if inst_label is None:
        return out

    inst = inst_label.to(device=device, dtype=torch.int64)
    valid = inst >= 0
    if not torch.any(valid):
        return out

    bid = _offset2batch(offset)                  # (N,)
    BIG = 1_000_003                              # separate groups by (batch, inst)
    gid = bid * BIG + inst

    sel = torch.nonzero(valid, as_tuple=False).squeeze(1)
    xyz_v = xyz[sel]
    gid_v = gid[sel]

    order = torch.argsort(gid_v)                 # contiguous per (batch, inst)
    xyz_s = xyz_v[order]
    gid_s = gid_v[order]

    change = torch.nonzero(gid_s[1:] != gid_s[:-1], as_tuple=False).squeeze(1) + 1
    starts = torch.cat([torch.tensor([0], device=device), change])
    counts = torch.diff(torch.cat([starts, torch.tensor([xyz_s.shape[0]], device=device)]))
    grp_offset = torch.cumsum(counts, dim=0).to(torch.int32)   # (G,)

    idx_s, _ = pointops.knn_query(nsample, xyz_s, grp_offset, xyz_s, grp_offset)  # (Nv, nsample)
    idx_global = sel[order][idx_s.long()]                    # map back into original indices

    out[sel[order]] = idx_global.to(torch.int32)
    return out

@torch.no_grad()
def build_inst_knn_idx_cpu(xyz, offset, inst_label, nsample=16):
    """
    CPU fallback (slow) using pairwise distances.
    xyz: (N,3) torch.FloatTensor on CPU
    """
    import numpy as np
    assert xyz.device.type == "cpu"
    N = xyz.shape[0]
    out = np.full((N, nsample), -1, dtype=np.int32)
    inst_np = inst_label.cpu().numpy()
    for b in range(offset.numel()):
        s = 0 if b == 0 else int(offset[b-1])
        e = int(offset[b])
        block_idx = np.arange(s, e)
        for iid in np.unique(inst_np[s:e]):
            if iid < 0:
                continue
            sel = block_idx[inst_np[s:e] == iid]
            if sel.size == 0:
                continue
            P = xyz[sel]                          # (m,3)
            D = torch.cdist(P, P)                 # (m,m)
            k = min(nsample, P.shape[0])
            nn_local = torch.topk(-D, k=k, dim=1).indices.numpy()  # include self
            out[sel, :k] = sel[nn_local]
    return torch.from_numpy(out)

