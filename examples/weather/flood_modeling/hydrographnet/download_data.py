from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
import os

# Get $WORK directory
work_dir = os.environ.get('WORK')
data_dir = f'{work_dir}/hydrographnet/data'

print(f'Downloading data to: {data_dir}')

# This triggers the Zenodo download
dataset = HydroGraphDataset(
    data_dir=data_dir,
    prefix='M80',
    num_samples=10,
    split='train',
    n_time_steps=2,
    return_physics=True
)

print(f'Download complete. Samples: {len(dataset)}')
print(f'Graph: {dataset[0][0]}')