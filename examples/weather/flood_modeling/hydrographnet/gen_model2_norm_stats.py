"""One-off: compute & persist Model_2 train norm stats.

Model_2 training wrote norm stats only to the ephemeral $TMPDIR copy, so the
persistent data dir lacks static_norm_stats.json / dynamic_norm_stats.json and
test-split inference fails. Instantiating the dataset in `train` mode against
the persistent dir recomputes and saves them (deterministic over the same train
events, so identical to what training used). use_1d=True to match the trained
model's stat keys (depth_1d / base_area_1d / per-type vol).
"""
import sys

from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/woody/iwi5/iwi5416h/urbanflood/data"

ds = UrbanFloodDataset(
    data_dir=DATA_DIR,
    model_name="Model_2",
    split="train",
    use_1d=True,
    num_samples=69,
    compute_boundary_mask=True,
)
print(f"[gen] Model_2 train dataset built: {ds.num_2d_nodes} 2D nodes, "
      f"{ds.num_1d_nodes} 1D nodes, {len(ds.event_ids)} events.")
print(f"[gen] static + dynamic norm stats written under {DATA_DIR}/Model_2/train/")
