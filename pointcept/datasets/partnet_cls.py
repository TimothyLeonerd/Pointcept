# pointcept/datasets/partnet_cls.py
import os
import glob
import json
import h5py
import numpy as np

from .defaults import DefaultDataset
from .builder  import DATASETS


@DATASETS.register_module()
class PartNetClsDataset(DefaultDataset):
    """
    PartNet (ins_seg_h5)  ▸  shape‑level classification loader.

    Output dict
    -----------
    coord    : (N,3) float32
    category : (1,) int64   – semantic class id (0…15)
    """

    def __init__(
        self,
        split="train",
        data_root="data/ins_seg_h5/ins_seg_h5",
        class_names=None,
        transform=None,
        cache=False,
        **kwargs,          # absorbs loop, num_points, etc.
    ):
        # ---------- consume custom keys -----------------------------------
        if class_names is None:
            raise ValueError("`class_names` list must be provided")
        self.class_names = class_names

        # discard keys DefaultDataset doesn't recognise
        kwargs.pop("num_points", None)        # not used (we keep full 10 k)

        # ---------- delegate to parent -----------------------------------
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            cache=cache,
            **kwargs,                         # now only accepted keys remain
        )

    # ------------------------------------------------------------------ helpers
    def get_data_list(self):
        """
        Build [(h5_path, row_idx, cat_id), ...] for split = train / val / test.
        """
        data_list = []
        for cat in sorted(os.listdir(self.data_root)):
            cat_dir = os.path.join(self.data_root, cat)
            if not os.path.isdir(cat_dir) or cat not in self.class_names:
                continue
            cat_id = self.class_names.index(cat)

            for js_path in glob.glob(os.path.join(cat_dir, f"{self.split}-*.json")):
                h5_path = os.path.splitext(js_path)[0] + ".h5"
                if not os.path.isfile(h5_path):
                    continue

                rows = json.load(open(js_path))

                # Case A: entries contain an explicit "row"
                if rows and isinstance(rows[0], dict) and "row" in rows[0]:
                    for rec in rows:
                        data_list.append((h5_path, int(rec["row"]), cat_id))
                # Case B: plain list – index is the row
                else:
                    for idx_in_batch, _ in enumerate(rows):
                        data_list.append((h5_path, idx_in_batch, cat_id))

        if not data_list:
            raise RuntimeError(f"No samples for split '{self.split}' in {self.data_root}")
        return data_list
    
    def build_same_inst_idx(self, inst, nsample=16):
        N = inst.shape[0]
        out = np.full((N, nsample), fill_value=-1, dtype=np.int32)
        for i_id in np.unique(inst):
            if i_id < 0:                 # skip padding label, if any
                continue
            pts = np.where(inst == i_id)[0]
            for p in pts:
                nbrs = np.random.choice(pts, size=min(nsample, len(pts)),
                                        replace=len(pts) < nsample)
                out[p, :len(nbrs)] = nbrs
        return out

    # ------------------------------------------------------------------ API
    def get_data(self, idx):
        h5_path, row, cat_id = self.data_list[idx % len(self.data_list)]
        with h5py.File(h5_path, "r") as f:
            coord = f["pts"][row]                 # (10000,3)
            inst  = f["label"][row]        # (10000,) int32

        inst_idx = self.build_same_inst_idx(inst) # (N,16)
        return dict(
            coord    = coord.astype(np.float32),
            category = np.array([cat_id], dtype=np.int64),
            inst_idx = inst_idx,   
        )
