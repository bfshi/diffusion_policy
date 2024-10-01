import random

import numpy as np
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_bimanual_image_range_normalizer

from mvp.bimanual_bc.dataset import Bimanual_Dataset

class BimanualImageDataset(BaseImageDataset):
    def __init__(self, **kwargs):
        self.dataset = Bimanual_Dataset(**kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_normalizer(self, mode='limits', **kwargs):
        ids = random.choices(range(len(self.dataset)), k=1000)
        states, actions = [], []
        for id in ids:
            states.append(self.dataset[id][1])
            actions.append(self.dataset[id][3])
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)

        data = {
            'action': actions,
            'state': states,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['rgb_left'] = get_bimanual_image_range_normalizer()
        normalizer['rgb_head'] = get_bimanual_image_range_normalizer()
        normalizer['rgb_right'] = get_bimanual_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx):
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        ims, pi_obs, _, pi_act = self.dataset[idx][:4]
        data = {
            'obs': {
                'rgb_left': ims[0],  # T, 3, 224, 224
                'rgb_head': ims[1],  # T, 3, 224, 224
                'rgb_right': ims[2],  # T, 3, 224, 224
                'state': pi_obs,  # T, 24
            },
            'action': pi_act  # T, 24
        }

        return data
