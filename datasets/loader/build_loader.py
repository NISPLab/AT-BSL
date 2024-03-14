from functools import partial
from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import GroupSampler, DistributedSampler, ClassAwareSampler, MixSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (12288, rlimit[1]))

def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu=2,
                     num_gpus=1,
                     dist=True,
                     sampler=None,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)

    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
        kwargs.update(shuffle=False)
    else:
        if sampler is not None:
            shuffle = False
            if 'ClassAware' in sampler:
                sampler = ClassAwareSampler(data_source=dataset)
            elif 'Mix' in sampler:
                sampler = MixSampler(data_source=dataset)
            elif 'Group' in sampler:
                sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
            else:
                raise NameError
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
        kwargs.update(shuffle=shuffle)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        drop_last=True,
        **kwargs)

    return data_loader
