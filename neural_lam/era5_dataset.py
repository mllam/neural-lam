import os
import glob
import torch
import numpy as np
import datetime as dt

from neural_lam import utils, constants

class ERA5UKDataset(torch.utils.data.Dataset):
    """
    ERA5 UK dataset
    
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    N_x = 268 (width)
    N_y = 238 (height)
    N_grid = 268x238 = 63784 (total number of grid nodes)
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    def __init__(
        self,
        dataset_name, 
        pred_length=1, 
        split="train", 
        subsample_step=3,
        standardize=True,
        subset=False,
        control_only=False
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        # member_file_regexp = "nwp*mbr000.npy" if control_only else "nwp*mbr*.npy"
        member_file_regexp = "*.npy"
        sample_paths = glob.glob(os.path.join(self.sample_dir_path, member_file_regexp))
        self.sample_names = [os.path.basename(path) for path in sample_paths]
        # self.sample_names = [path.split("/")[-1][4:-4] for path in sample_paths]

        if subset:
            self.sample_names = self.sample_names[:50] # Limit to 50 samples

        self.sample_length = pred_length + 2 # 2 init states
        # self.subsample_step = subsample_step
        # self.original_sample_length = 65//self.subsample_step # 21 for 3h steps
        # assert self.sample_length <= self.original_sample_length, (
        #         "Requesting too long time series samples")

        # # Set up for standardization
        # self.standardize = standardize
        # if standardize:
        #     ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
        #     self.data_mean, self.data_std, self.flux_mean, self.flux_std =\
        #         ds_stats["data_mean"], ds_stats["data_std"], ds_stats["flux_mean"], \
        #         ds_stats["flux_std"]


        # # If subsample index should be sampled (only duing training)
        # self.random_subsample = split == "train"

    def __len__(self):
        # two states needed to make prediction
        return len(self.sample_names) - self.sample_length + 1
    
    def _get_sample(self, sample_name):
        sample_path = os.path.join(self.sample_dir_path, f"{sample_name}")
        try:
            full_sample = torch.tensor(np.load(sample_path),
                    dtype=torch.float32) # (N_vars, N_levels, N_lat, N_lon)
        except ValueError:
            print(f"Failed to load {sample_path}")

        full_sample = full_sample.permute(3,2,0,1) # (N_lon, N_lat, N_vars, N_levels)
        full_sample = full_sample.reshape(full_sample.shape[0]*full_sample.shape[1], -1) # (N_lon*N_lat, N_vars*N_levels)        
        return full_sample

    def __getitem__(self, idx):
        # === Sample ===
        prev_prev_state = self._get_sample(self.sample_names[idx])        
        prev_state = self._get_sample(self.sample_names[idx+1])        
        target_state = self._get_sample(self.sample_names[idx+2])
        
        # N_grid = N_x * N_y; d_features = N_vars * N_levels
        init_states = torch.stack((prev_prev_state, prev_state), dim=0) # (2, N_grid, d_features)
        target_states = target_state.unsqueeze(0) # (1, N_grid, d_features)

        return init_states, target_states, torch.Tensor(), torch.Tensor()
