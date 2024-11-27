import datetime
import torch
from torch.nn import LSTM
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from pathlib import Path


class ConfigCore():
    def __init__(self, data_path='./'):
        # network
        self.device = 'cuda:0'
        self.SEED_NUMBER = 0
        self.FC_SIZE = 300
            
        # RL
        self.GAMMA = 0.98
        self.TAU = 0.005
        self.POLICY_FREQ = 2
        self.policy_noise = 0.05
        self.policy_noise_clip = 0.1
        
        # optimzer
        self.optimizer = Adam
        self.lr = 3e-4
        self.eps = 1.5e-4
        self.decayed_lr = 5e-5
        
        # environment
        self.STATE_DIM = 5
        self.ACTION_DIM = 2
        self.POS_DIM = 3
        self.OBS_DIM = 2
        self.TARGET_DIM = 2
        self.TERMINAL_ACTION = 0.1
        self.DT = 0.1 # s
        self.EPISODE_TIME = 10 # s
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DT)
        self.REWARD_SCALE = 10
        self.LINEAR_SCALE = 400 # cm/unit
        self.goal_radius_range = np.array([65, 65]) / self.LINEAR_SCALE
        self.initial_radius_range = np.array([100, 400]) / self.LINEAR_SCALE
        self.relative_angle_range = np.deg2rad([-35, 35])
        self.process_gain_default = torch.tensor([200 / self.LINEAR_SCALE, torch.deg2rad(torch.tensor(90.))])
        self.target_fullon = False
        self.target_offT = 3 # steps
        
        # others
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = data_path
        
        # For model-free belief
        self.BATCH_SIZE = 16
        self.MEMORY_SIZE = int(1e5)
        self.RNN = LSTM
        self.RNN_SIZE = 128
             
        # For model-based belief
        self.EKF_STATE_DIM = 13
        self.EKF_BATCH_SIZE = 256
        self.EKF_MEMORY_SIZE = int(1.6e6)
        
    def save(self):
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, self.data_path / f'{self.filename}_arg.pkl')
        
    def load(self, filename):
        self.__dict__ = torch.load(self.data_path / f'{filename}_arg.pkl')
        self.filename = filename
       
        
class ConfigGain(ConfigCore):
    def __init__(self, data_path='./', gain_distribution='uniform', exclude_gain=None):
        super().__init__(data_path)
        self.task = 'gain'
        self.gain_distribution = gain_distribution
        self.process_gain_range = [1, 1]
        self.exclude_gain = exclude_gain
        self.pro_noise_range = [0.2, 0.2]
        self.obs_noise_range = [0.1, 0.1]
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        