import numpy as np
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class DummyImageDataset(BaseImageDataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self) -> int:
        return 1000

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': np.random.rand(100, 32, 24),
            'state': np.random.rand(100, 32, 24),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['rgb_left'] = get_image_range_normalizer()
        normalizer['rgb_head'] = get_image_range_normalizer()
        normalizer['rgb_right'] = get_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx):
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        data = {
            'obs': {
                'rgb_left': np.random.rand(32, 3, 224, 224),  # T, 3, 224, 224
                'rgb_head': np.random.rand(32, 3, 224, 224),  # T, 3, 224, 224
                'rgb_right': np.random.rand(32, 3, 224, 224),  # T, 3, 224, 224
                'state': np.random.rand(32, 24),  # T, 24
            },
            'action': np.random.rand(32, 24)  # T, 24
        }

        return data
