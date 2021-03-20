import logging

import torch

from torchvision import transforms
from data.datasets import DefaultDataset
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data_path = "/lab/vislab/DATA/CUB/metric_learning"
    trainset = DefaultDataset("{}/train".format(data_path), transform_train)
    testset = DefaultDataset("{}/val".format(data_path), transform_test)

    train_sampler = samplers.MPerClassSampler(trainset.targets, args.m, length_before_new_iter=len(trainset))
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
