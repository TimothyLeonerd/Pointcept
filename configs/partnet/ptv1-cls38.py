# --------------------------------------------------------------
#  Point Transformer v1  ▸  PartNet‑24 shape‑level classification
# --------------------------------------------------------------
_base_ = ["../_base_/default_runtime.py"]   # keep relative path as in other configs
seed = 58241877                             # explicit so "cfg.seed" always exists

# ------------- misc --------------------------------------------------------
batch_size         = 8     # total across all GPUs (adjust to VRAM)
batch_size_val     = 16
batch_size_test    = 16
num_worker         = 4
empty_cache        = False
enable_amp         = False     # AMP off for first experiments

# ------------- model -------------------------------------------------------
model = dict(
    type="DefaultClassifier",
    num_classes=24,
    backbone_embed_dim=512,
    backbone=dict(
        type="PTv1Cls38_Features",    # backbone file exists in Pointcept
        in_channels=3                 # xyz only
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ------------- optimiser & scheduler --------------------------------------
epoch      = 10
eval_epoch = epoch
optimizer  = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler  = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# ------------- dataset -----------------------------------------------------
dataset_type = "PartNetClsDataset"
data_root    = "data/ins_seg_h5/ins_seg_h5"      # adjust if you move the bundle

class_names = [
    "Bag", "Bed", "Bottle", "Bowl", "Chair", "Clock",
    "Dishwasher", "Display", "Door", "Earphone", "Faucet", "Hat",
    "Keyboard", "Knife", "Lamp", "Laptop", "Microwave", "Mug",
    "Refrigerator", "Scissors", "StorageFurniture", "Table", "TrashCan", "Vase",
]

data = dict(
    num_classes=24,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        loop=1,
        num_points=1024,
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            #dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True,),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category", "inst_idx"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        loop=1,
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            #dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True,),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category", "inst_idx"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
)

# ------------- runtime hooks -----------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
