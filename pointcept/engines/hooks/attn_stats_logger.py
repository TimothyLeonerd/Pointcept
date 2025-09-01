# pointcept/engines/hooks/attn_stats_logger.py
import torch
import torch.distributed as dist

import pointcept.utils.comm as comm
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class AttnStatsLogger(HookBase):
    """
    Logs per-encoder cross-instance neighbor % (enc1..enc5) to TensorBoard and train.log.
    Requires backbone.get_attn_stats(reset=True) to be available.
    """

    def __init__(self, every_n_iters=None, every_n_epochs=1, tag_prefix="attn"):
        self.every_n_iters = every_n_iters   # e.g., 200 (or None to disable per-iter)
        self.every_n_epochs = every_n_epochs  # e.g., 1 (or None to disable per-epoch)
        self.tag_prefix = tag_prefix
        self._last_logged_epoch = -1

    def _get_backbone(self):
        m = self.trainer.model
        if hasattr(m, "module"):
            m = m.module
        # DefaultClassifier holds .backbone; if you run the backbone directly, fall back
        return getattr(m, "backbone", m)

    def _reduce_across_ranks(self, stats_dict):
        """
        Sum mismatch/valid across ranks so pct = total_mismatch/total_valid.
        """
        if comm.get_world_size() == 1:
            return stats_dict

        reduced = {}
        for lvl, d in stats_dict.items():
            t = torch.tensor([d["mismatch"], d["valid"]],
                             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                             dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mismatch = int(t[0].item())
            valid    = int(t[1].item())
            pct = (100.0 * mismatch / valid) if valid > 0 else float("nan")
            reduced[lvl] = dict(mismatch=mismatch, valid=valid, pct=pct, calls=d.get("calls", 0))
        return reduced

    def _emit(self, step, when="iter"):
        backbone = self._get_backbone()
        if not hasattr(backbone, "get_attn_stats"):
            return

        # collect + reset counters on each rank
        stats = backbone.get_attn_stats(reset=True)
        if not stats:
            return

        # DDP: sum across ranks (so we log global numbers)
        stats = self._reduce_across_ranks(stats)

        # Log to TensorBoard (if available) and to train.log
        writer = getattr(self.trainer, "writer", None)
        if writer is not None:
            for lvl, d in stats.items():
                writer.add_scalar(f"{self.tag_prefix}/{lvl}_outside_pct", d["pct"], step)
                writer.add_scalar(f"{self.tag_prefix}/{lvl}_valid",       d["valid"], step)
                writer.add_scalar(f"{self.tag_prefix}/{lvl}_mismatch",    d["mismatch"], step)

        msg = " | ".join([f"{lvl}: {d['pct']:.2f}% (m={d['mismatch']}, v={d['valid']})"
                          for lvl, d in stats.items()])
        self.trainer.logger.info(f"[{when} {step}] cross-instance neighbors → {msg}")

    # HookBase API used by your trainer:
    def after_step(self):
        if self.every_n_iters is None:
            return
        # per-epoch iteration index is in trainer.comm_info["iter"] (0-based)
        it = self.trainer.comm_info.get("iter", None)
        if it is None:
            return
        if (it + 1) % self.every_n_iters == 0:
            # Convert to a global step (like InformationWriter does)
            epoch_base = self.trainer.epoch * len(self.trainer.train_loader)
            global_step = epoch_base + (it + 1)
            self._emit(global_step, when="iter")

    def after_epoch(self):
        if self.every_n_epochs is None:
            return
        if (self.trainer.epoch + 1) % self.every_n_epochs == 0 and self.trainer.epoch > self._last_logged_epoch:
            self._emit(self.trainer.epoch + 1, when="epoch")
            self._last_logged_epoch = self.trainer.epoch
