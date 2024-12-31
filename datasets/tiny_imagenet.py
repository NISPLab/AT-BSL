import os
import torch

import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision.transforms as v2
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import random

DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}


class TinyImagenet(VisionDataset):
    """ TinyImagenet Dataset.
    Note: We download TinyImagenet dataset from <http://cs231n.stanford.edu/tiny-imagenet-200.zip>, then repack it as `.npz` format. 

    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'val'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``v2.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/train.npz",
            "train.npz",
            "db414016436353892fdf00cb30b9ee57",
        ],
        "val": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/val.npz",
            "val.npz",
            "7762694b6217fec8ba1a7be3c20ef218",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # reading(loading) npz file as array
        loaded_npz = np.load(os.path.join(self.root, self.filename))
        self.data = loaded_npz['image']
        self.targets = loaded_npz["label"].tolist()
        print(split+' images size:', self.data.shape)
        print(split+' labels size:', len(self.targets))

        if split == 'train':
            np.random.seed(0)
            random.seed(0)
            self.img_num_list = self.get_img_num_per_cls(200, 'exp', 0.1)
            self.gen_imbalanced_data(self.img_num_list)
            print(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = 100000 / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        idsx = []
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # print(the_img_num, end=' ')
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            temp_data = []
            for i in range(len(selec_idx)):
                temp_data.append(self.data[selec_idx[i]])
            new_data.append(temp_data) #[100000, 3, 64, 64]
            new_targets.extend([the_class, ] * the_img_num)
            idsx.append(selec_idx)
        # print()

        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        print(fpath)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)



def load_tinyimagenet(data_dir, aug, batch_size, test_batch_size):
    if aug == 'ra':
        transform_aug = [v2.RandAugment(2,8)]
    elif aug == 'none':
        transform_aug = []
    train_transform = v2.Compose(transform_aug + [
        v2.RandomCrop(64, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
    ])

    train_dataset = TinyImagenet(root=data_dir, split='train', download=False, transform=train_transform)
    test_dataset = TinyImagenet(root=data_dir, split='val', download=False, transform=test_transform)
    samples_per_cls = train_dataset.img_num_list

    kwargs = {'num_workers': 4, 'pin_memory': False}    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_dataloader, test_dataloader, samples_per_cls
