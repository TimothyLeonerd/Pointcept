# pointcept/models/point_transformer/point_transformer_cls.py
import torch
import torch.nn as nn
import pointops

from .point_transformer_seg import TransitionDown, Bottleneck
from pointcept.models.builder import MODELS


class PointTransformerCls(nn.Module):
    def __init__(self, block, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])

        self.cls = nn.Sequential(
            nn.Linear(planes[4], 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        inst = data_dict.get("instance", None)   # (N,) long

        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)

        # enc1[0] is stride-1 TransitionDown → positions unchanged
        p1, x1, o1 = self.enc1[0]([p0, x0, o0])

        # Build same-instance kNN only for stage-1 bottleneck (if present)
        idx_override = None
        if len(self.enc1) > 1 and inst is not None:
            idx_override = self._build_inst_knn_batch(p1, inst, o1, nsample=self.enc1[1].transformer.nsample)

        if len(self.enc1) > 1:
            p1, x1, o1 = self.enc1[1]([p1, x1, o1], idx_override=idx_override)

        # remaining stages (standard kNN inside)
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # global average per cloud
        outs = []
        for i in range(o5.shape[0]):
            s = 0 if i == 0 else o5[i-1].item()
            e = o5[i].item()
            outs.append(x5[s:e].mean(0, keepdim=True))
        x = torch.cat(outs, dim=0)
        x = self.cls(x)
        return x

    @staticmethod
    def _build_inst_knn_batch(p, inst, offset, nsample=16):
        """
        p      : (N,3) float (cuda)
        inst   : (N,)  long  (cpu or cuda)  — per-point instance ids
        offset : (B,)  int cumulative counts
        return : (N, nsample) int (cuda) indices, neighbors within same instance (per cloud)
        """
        device = p.device
        inst = inst.to(device).long()
        N = p.shape[0]
        out_idx = torch.empty((N, nsample), dtype=torch.int, device=device)

        start = 0
        for b in range(offset.shape[0]):
            end = offset[b].item()
            p_b = p[start:end].contiguous()
            i_b = inst[start:end].contiguous()

            # sort points by instance id so each instance is contiguous
            order = torch.argsort(i_b)
            p_s   = p_b[order]
            i_s   = i_b[order]

            # counts per instance (on CUDA)
            uniq, counts = torch.unique(i_s, return_counts=True)
            # build int32 offset for knn_query
            off = torch.cumsum(counts, dim=0).int()

            # run CUDA kNN within instances (search==query, offset==off)
            idx_s, _ = pointops.knn_query(nsample, p_s, off, p_s, off)  # (Nb, nsample)

            # map back to original indices within this cloud
            idx_b = order[idx_s]              # (Nb, nsample)
            out_idx[start:end] = idx_b + start
            start = end

        return out_idx
        

@MODELS.register_module("PointTransformer-Cls26")
class PointTransformerCls26(PointTransformerCls):
    def __init__(self, **kwargs):
        super().__init__(Bottleneck, [1, 1, 1, 1, 1], **kwargs)


@MODELS.register_module("PointTransformer-Cls38")
class PointTransformerCls38(PointTransformerCls):
    def __init__(self, **kwargs):
        super().__init__(Bottleneck, [2, 2, 2, 2, 2], **kwargs)


# ===== features-only backbones for DefaultClassifier =====

@MODELS.register_module()
class PTv1Cls38_Features(PointTransformerCls38):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.cls  # backbone only

    # ----- helper: run one encoder (enc1..enc5) with instance-aware KNN on its first block
    @staticmethod
    def _run_stage_with_inst_knn(enc: nn.Sequential, p, x, o, inst):
        """
        enc   : nn.Sequential [ TransitionDown, Block1, Block2, ... ]
        p,x,o : inputs of this stage
        inst  : per-point instance ids aligned with 'p' (or None)

        returns: (p_new, x_new, o_new, inst_new)
        """
        # 1) always run the stage's TransitionDown first
        td = enc[0]
        p_new, x_new, o_new = td([p, x, o])

        # 2) propagate instance labels through FPS mapping
        inst_new = None
        if inst is not None:
            # make sure 'inst' is on the same device and is long
            inst = inst.to(p_new.device).to(torch.long)
            if td.last_down_idx is None:
                # stride == 1 (e.g., enc1): positions unchanged → labels unchanged
                inst_new = inst
            else:
                # copy labels via FPS indices (global indices)
                inst_new = inst[td.last_down_idx.long()]

        # 3) instance-aware KNN for the FIRST block only (if it exists and we have labels)
        if len(enc) > 1 and inst_new is not None:
            first_block = enc[1]
            ns = first_block.transformer.nsample  # nsample setting of this block
            # build same-instance neighbors under current batch partition 'o_new'
            idx_override = PointTransformerCls._build_inst_knn_batch(
                p_new, inst_new, o_new, nsample=ns
            )
            p_new, x_new, o_new = first_block([p_new, x_new, o_new], idx_override=idx_override)
            # run any remaining blocks normally
            for i in range(2, len(enc)):
                p_new, x_new, o_new = enc[i]([p_new, x_new, o_new])
        else:
            # no labels or no blocks beyond TD → run remaining blocks (if any) normally
            for i in range(1, len(enc)):
                p_new, x_new, o_new = enc[i]([p_new, x_new, o_new])

        return p_new, x_new, o_new, inst_new

    def forward(self, data_dict):
        # inputs
        p0 = data_dict["coord"]                 # (N,3)
        x0 = data_dict["feat"]                  # (N,C)  (often zeros if using xyz only)
        o0 = data_dict["offset"].int()          # (B,)
        inst0 = data_dict.get("instance", None) # (N,) long/int per-point instance id

        # features: xyz-only if in_channels==3, else concat (unchanged behavior)
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)

        # enc1..enc5 with instance-aware KNN applied to the first block of each encoder
        p1, x1, o1, inst1 = self._run_stage_with_inst_knn(self.enc1, p0, x0, o0, inst0)
        p2, x2, o2, inst2 = self._run_stage_with_inst_knn(self.enc2, p1, x1, o1, inst1)
        p3, x3, o3, inst3 = self._run_stage_with_inst_knn(self.enc3, p2, x2, o2, inst2)
        p4, x4, o4, inst4 = self._run_stage_with_inst_knn(self.enc4, p3, x3, o3, inst3)
        _,  x5, o5, _     = self._run_stage_with_inst_knn(self.enc5, p4, x4, o4, inst4)

        # global pooling per batch (unchanged)
        feats = torch.stack([
            x5[(o5[i-1] if i else 0):o5[i]].mean(0)
            for i in range(o5.shape[0])
        ])
        return feats
