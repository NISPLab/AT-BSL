# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from torchvision.utils import save_image
import torch


random_seed = 0
train_sampler_type = 'default'
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, imbalance_ratio=0.02, root='../data', train=True, imb_type='exp',
                 transform=None, target_transform=None, download=True):
        mode = "train" if train else "evaluation"
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.train = train
        rand_number = random_seed
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            imb_factor = imbalance_ratio
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(self.img_num_list)
            self.transform = transform
        else:
            self.transform = transform
        if train_sampler_type == "weighted sampler" and self.train:
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
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

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

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
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            idsx.append(selec_idx)
        # print()

        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets


    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class SMALLCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, existing_ratio=0.02, root='../data', train=True, imb_type='exp',
                 transform=None, target_transform=None, download=True):
        mode = "train" if train else "evaluation"
        super(SMALLCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.train = train
        rand_number = random_seed
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, existing_ratio)
            self.gen_imbalanced_data(self.img_num_list)
            self.transform = transform
        else:
            self.transform = transform
        if train_sampler_type == "weighted sampler" and self.train:
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, cls_num, existing_ratio):
        img_max = int(len(self.data) / cls_num * existing_ratio)
        img_num_per_cls = [img_max] * cls_num
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # print(the_img_num, end=' ')
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        # print()
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets


    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class SMALLCIFAR100(SMALLCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
