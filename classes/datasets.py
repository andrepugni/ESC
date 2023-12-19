import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, FakeData
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TabularDataset(Dataset):
    def __init__(
        self,
        dataset: str or pd.DataFrame,
        atts: list,
        root: str = "data/clean",
        target: str = "TARGET",
        set: str = "train",
        split: bool = False,
        test_perc: float = 0.1,
        device: str = "cuda:0",
    ):
        super(TabularDataset, self).__init__()
        if type(dataset) == str:
            self.path = os.path.join(root, dataset, dataset + "_" + set + ".csv")
            self.df = pd.read_csv(self.path)
        elif type(dataset) == pd.DataFrame:
            self.df = dataset
        if split:
            self.df, self.val = train_test_split(
                self.df, test_size=test_perc, random_state=42, stratify=self.df[target]
            )
        self.cat_atts = list(
            self.df[atts].select_dtypes(include=["object", "category"]).columns
        )
        self.cont_atts = list(
            self.df[atts].select_dtypes(exclude=["object", "category"]).columns
        )
        for col in self.cat_atts:
            self.df[col] = self.df[col].astype("category").cat.codes
        for col in self.cont_atts:
            self.df[col] = self.df[col].astype(float)
        self.x_num = torch.from_numpy(self.df[self.cont_atts].values).float().to(device)
        self.x_cat = torch.from_numpy(self.df[self.cat_atts].values).to(device).long()
        self.y = torch.from_numpy(self.df[target].values).to(device)
        self.targets = self.df[target].values
        self.data = torch.cat([self.x_num, self.x_cat], dim=1)
        self.classes = np.unique(self.df[target])

    def __getitem__(self, index):
        return self.x_num[index], self.x_cat[index], self.y[index], index

    def __len__(self):
        return self.y.shape[0]


class ImgFolder(ImageFolder):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index


class FkeData(FakeData):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index


def transformation(dataset: str, also_inp_size: bool = False):
    if dataset == "catsdogs":
        input_size = 64
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset in ["cifar10", "cifar100"]:
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif dataset in [
        "organamnist",
        "organcmnist",
        "chestmnist",
        "breastmnist",
        "dermamnist",
        "bloodmnist",
        "pneumoniamnist",
        "tissuemnist",
        "octmnist",
        "pathmnist",
        "retinamnist",
        "organsmnist",
        "FashionMNIST",
    ]:
        input_size = 28
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset == "SVHN":
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(15),
                transforms.RandomCrop(input_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif dataset == "MNIST":
        input_size = 28
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
    elif dataset == "waterbirds":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        # assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             (
        #                 int(target_resolution[0] * scale),
        #                 int(target_resolution[1] * scale),
        #             )
        #         ),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "stanfordcars":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "food101":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "xray":
        input_size = 224
        transform_test = transforms.Compose(
            [
                transforms.Resize(size=(input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(
                    20
                ),  # Randomly rotate the image within a range of (-20, 20) degrees
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),  # Randomly flip the image horizontally with 50% probability
                transforms.RandomResizedCrop(
                    size=(input_size, input_size), scale=(0.8, 1.0)
                ),
                # Randomly crop the image and resize it
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                # Randomly change the brightness, contrast, saturation, and hue
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5
                ),
                # Randomly apply affine transformations with translation
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.2)], p=0.5
                ),
                # Randomly apply perspective transformations
                transforms.Resize(size=(input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif dataset == "celeba":
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        input_size = 224
        assert input_size is not None
        transform_train = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "oxfordpets":
        input_size = 224
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    else:
        raise NotImplementedError(
            "The transformations for this dataset have not been defined yet!"
        )
    if also_inp_size:
        return transform_train, transform_test, input_size
    else:
        return transform_train, transform_test


class TabularDatasetLGBM(Dataset):
    def __init__(
        self,
        dataset: str or pd.DataFrame,
        atts: list,
        root: str = "data/clean",
        target: str = "TARGET",
        set: str = "train",
        split: bool = False,
        test_perc: float = 0.1,
        device: str = "cuda:0",
    ):
        super(TabularDatasetLGBM, self).__init__()
        if type(dataset) == str:
            self.path = os.path.join(root, dataset, dataset + "_" + set + ".csv")
            self.df = pd.read_csv(self.path)
        elif type(dataset) == pd.DataFrame:
            self.df = dataset
        if split:
            self.df, self.val = train_test_split(
                self.df, test_size=test_perc, random_state=42, stratify=self.df[target]
            )
        self.cat_atts = list(
            self.df[atts].select_dtypes(include=["object", "category"]).columns
        )
        self.cont_atts = list(
            self.df[atts].select_dtypes(exclude=["object", "category"]).columns
        )
        for col in self.cat_atts:
            self.df[col] = self.df[col].astype("category")
        for col in self.cont_atts:
            self.df[col] = self.df[col].astype(float)
        self.targets = self.df[target].values
        self.classes = np.unique(self.df[target])

    def __len__(self):
        return self.targets.shape[0]


def onlytransformation(dataset: str, also_inp_size: bool = False):
    if dataset == "catsdogs":
        input_size = 64
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif dataset in ["cifar10", "cifar100"]:
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif dataset in [
        "organamnist",
        "organcmnist",
        "chestmnist",
        "breastmnist",
        "dermamnist",
        "bloodmnist",
        "pneumoniamnist",
        "tissuemnist",
        "octmnist",
        "pathmnist",
        "retinamnist",
        "organsmnist",
        "FashionMNIST",
    ]:
        input_size = 28
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif dataset == "SVHN":
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(15),
                transforms.RandomCrop(input_size, padding=4),
            ]
        )

    elif dataset == "MNIST":
        input_size = 28
        transform_train = transforms.Compose([])

    elif dataset == "waterbirds":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )
    elif dataset == "stanfordcars":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
            ]
        )

    elif dataset == "food101":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
            ]
        )
    elif dataset == "xray":
        input_size = 224
        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(
                    20
                ),  # Randomly rotate the image within a range of (-20, 20) degrees
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),  # Randomly flip the image horizontally with 50% probability
                transforms.RandomResizedCrop(
                    size=(input_size, input_size), scale=(0.8, 1.0)
                ),
                # Randomly crop the image and resize it
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                # Randomly change the brightness, contrast, saturation, and hue
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5
                ),
                # Randomly apply affine transformations with translation
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.2)], p=0.5
                ),
                # Randomly apply perspective transformations
                transforms.Resize(size=(input_size, input_size)),
            ]
        )
    elif dataset == "celeba":
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        input_size = 224
        assert input_size is not None
        transform_train = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(input_size),
            ]
        )
    elif dataset == "oxfordpets":
        input_size = 224
        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=6),
                transforms.RandomHorizontalFlip(),
            ]
        )

    else:
        raise NotImplementedError(
            "The transformations for this dataset have not been defined yet!"
        )
    if also_inp_size:
        return transform_train
    else:
        return transform_train


def normalization(dataset: str, also_inp_size: bool = False):
    if dataset == "catsdogs":
        input_size = 64
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset in ["cifar10", "cifar100"]:
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif dataset in [
        "organamnist",
        "organcmnist",
        "chestmnist",
        "breastmnist",
        "dermamnist",
        "bloodmnist",
        "pneumoniamnist",
        "tissuemnist",
        "octmnist",
        "pathmnist",
        "retinamnist",
        "organsmnist",
        "FashionMNIST",
    ]:
        input_size = 28
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset == "SVHN":
        input_size = 32
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif dataset == "MNIST":
        input_size = 28
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
    elif dataset == "waterbirds":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        # assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             (
        #                 int(target_resolution[0] * scale),
        #                 int(target_resolution[1] * scale),
        #             )
        #         ),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "stanfordcars":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "food101":
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        input_size = 224
        assert target_resolution is not None
        # transform_train = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.CenterCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # transform_test = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "xray":
        input_size = 224
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif dataset == "celeba":
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        input_size = 224
        assert input_size is not None
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif dataset == "oxfordpets":
        input_size = 224
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    else:
        raise NotImplementedError(
            "The transformations for this dataset have not been defined yet!"
        )
    if also_inp_size:
        return transform_train, input_size
    else:
        return transform_train
