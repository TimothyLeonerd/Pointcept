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
    PartNet (ins_seg_h5) → shape-level classification.

    Output dict
    -----------
    coord     : (N,3) float32
    category  : (1,)  int64
    instance  : (N,)  int64  per-point instance id (no CUDA work here)
    """

    def __init__(
        self,
        split="train",
        data_root="data/ins_seg_h5/ins_seg_h5",
        class_names=None,
        transform=None,
        cache=False,
        **kwargs,                      # absorbs loop, num_points, etc.
    ):
        if class_names is None:
            raise ValueError("`class_names` list must be provided")
        self.class_names = class_names

        # DefaultDataset doesn't take num_points; discard if present
        kwargs.pop("num_points", None)

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            cache=cache,
            **kwargs,
        )

    # --------------------------------------------------------------- helpers
    def get_data_list(self):
        """
        Build [(h5_path, row_idx, cat_id), ...] for split = train / val / test.
        Compatible with:
          • [{'row': 17, ...}, ...]
          • [ {...}, {...}, ... ]  (row = list index)
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

                if rows and isinstance(rows[0], dict) and "row" in rows[0]:
                    for rec in rows:
                        data_list.append((h5_path, int(rec["row"]), cat_id))
                else:
                    for idx_in_batch, _ in enumerate(rows):
                        data_list.append((h5_path, idx_in_batch, cat_id))

        if not data_list:
            raise RuntimeError(
                f"No samples for split '{self.split}' in {self.data_root}"
            )
        return data_list

    # ------------------------------------------------------------------- API
    def get_data(self, idx):
        h5_path, row, cat_id = self.data_list[idx % len(self.data_list)]
        with h5py.File(h5_path, "r") as f:
            coord = f["pts"][row]     # (N,3) float
            inst  = f["label"][row]   # (N,)  int

        return dict(
            coord    = coord.astype(np.float32),
            category = np.array([cat_id], dtype=np.int64),
            instance = inst.astype(np.int64),
        )
