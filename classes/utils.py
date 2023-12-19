# -*- coding: utf-8 -*-
import torch.utils.data
import torchvision.models
from tqdm import tqdm
from classes.modules import *
from classes.losses import *
from classes.datasets import *
from torchvision.models.resnet import BasicBlock, Bottleneck
import random
import sklearn.metrics as skm
import torch.optim as optim
import copy
from time import time
import scipy

cfg = {
    "D": [
        64,
        0.3,
        64,
        "M",
        128,
        0.4,
        128,
        "M",
        256,
        0.4,
        256,
        0.4,
        256,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        0.5,
    ]
}


# set seed function
def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2**32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(f"Random seed {seed} has been set.")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_layers(cfg, batch_norm=False):
    """ "
    VGG module is taken from https://github.com/LayneH/SAT-selective-cls/blob/main/models/cifar/vgg.py implementation
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v) == int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v) == float:
            layers += [nn.Dropout(v)]
    return nn.Sequential(*layers)


def vgg16(**kwargs):
    """

    VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg["D"]), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg["D"], batch_norm=True), **kwargs)
    return model


def buildSelNet(
    model_type: str,
    body_params: dict,
    head_params: dict,
):
    """
    Args:
        model_type: str
            The main body type. Possible choices are:
            - 'Resnet50'
            - 'Resnet34'
            - 'TabResnet'
            - 'TabFTTransformer'
            - 'VGG16'
            - 'VGG16bn'
        body_params: dict
            A dictionary containing the body parameters for the network
        head_params:
            A dictionary containing the head parameters for the network
    Returns:
        torch.Module:
            A SelNet architecture, with a final HeadSelectiveNet
    """
    if model_type == "TabResnet":
        model = TabCatResNet(**body_params)
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "resnet"
            ), "Check the head is configured for a ResNet architecture"
        else:
            head_params["main_body"] = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.resnet.blocks[-1].linear_second.out_features
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["d_in"] = model.resnet.blocks[-1].linear_second.out_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadSelectiveNet(**head_params)
        model.resnet.head = head
    elif model_type == "TabFTTransformer":
        model = TabFTTransformer(**body_params)
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "transformer"
            ), "Check the head is configured for a transformer architecture"
        else:
            head_params["main_body"] = "transformer"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.ftt.transformer.blocks[-1].ffn_normalization.normalized_shape[
                    0
                ]
            ), "Check the input of the head corresponds to the last one of the transformer"
        else:
            head_params["d_in"] = model.ftt.transformer.blocks[
                -1
            ].ffn_normalization.normalized_shape[0]
        if "batch_norm" in head_params.keys():
            assert (
                head_params["batch_norm"] == "layer_norm"
            ), "Check the head has set a layer_norm parameter"
        else:
            head_params["batch_norm"] = "layer_norm"
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadSelectiveNet(**head_params)
        model.ftt.transformer.head = head
    elif model_type == "VGG16":
        if "main_body" in body_params:
            assert (
                body_params["main_body"] == "VGG"
            ), "Please check your body type is consistent with the model"
        else:
            body_params["main_body"] = "VGG"
            head_params["main_body"] = "VGG"
        model = vgg16(**body_params)
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.classifier[0].in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.classifier[0].in_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == False
            ), "Check the input of the head is not normalized"
        else:
            head_params["pre_norm"] = False
        print(head_params)
        head = HeadSelectiveNet(**head_params)
        model.classifier = head
    elif model_type == "VGG16bn":
        if "main_body" in body_params:
            assert (
                body_params["main_body"] == "VGG"
            ), "Please check your body type is consistent with the model"
        else:
            body_params["main_body"] = "VGG"
            head_params["main_body"] = "VGG"
        model = vgg16_bn(**body_params)
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.classifier[0].in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.classifier[0].in_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == False
            ), "Check the input of the head is not normalized"
        else:
            head_params["pre_norm"] = False
        print(head_params)
        head = HeadSelectiveNet(**head_params)
        model.classifier = head
    elif model_type == "Resnet34":
        if (body_params["block"] == BasicBlock) and (
            body_params["layers"] == [3, 4, 6, 3]
        ):
            model = torchvision.models.ResNet(**body_params)
        else:
            model = torchvision.models.ResNet(
                block=BasicBlock, layers=[3, 4, 6, 3], **body_params
            )
        model.body_type = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadSelectiveNet(**head_params)
        model.fc = head
    elif model_type == "Resnet18":
        if (body_params["block"] == BasicBlock) and (
            body_params["layers"] == [2, 2, 2, 2]
        ):
            model = torchvision.models.ResNet(**body_params)
        else:
            model = torchvision.models.ResNet(
                block=BasicBlock, layers=[2, 2, 2, 2], **body_params
            )
        model.body_type = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadSelectiveNet(**head_params)
        model.fc = head
    elif model_type == "Resnet50":
        if (body_params["block"] == Bottleneck) and (
            body_params["layers"] == [3, 4, 6, 3]
        ):
            model = torchvision.models.ResNet(**body_params)
        else:
            model = torchvision.models.ResNet(
                block=Bottleneck, layers=[3, 4, 6, 3], **body_params
            )
        model.body_type = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadSelectiveNet(**head_params)
        model.fc = head
    else:
        model = None
    return model


def disable_dropout(model, verbose=False):
    # Freeze also BN running average parameters
    for layer in model.named_modules():
        if "Dropout" in layer[1]._get_name():
            print("deactivating dropout")
            layer[1].eval()
    for layer in model.named_modules():
        print(layer[1].training)
    return model


def buildConfidNet(
    model_type: str,
    orig_model: torch.nn.Module,
    head_params: dict,
    deactivate: bool = True,
):
    """
    Args:
        model_type: str
            The main body type. Possible choices are:
            - 'Resnet50'
            - 'Resnet34'
            - 'TabResnet'
            - 'TabFTTransformer'
            - 'VGG16'
            - 'VGG16bn'
        orig_model: torch.Module
            The original model to build the confidnet
        head_params:
            A dictionary containing the head parameters for the network
        deactivate: bool
            A boolean to deactivate the dropout layers
    Returns:
        torch.Module:
            A SelNet architecture, with a final HeadConfidNet
    """
    model = copy.deepcopy(orig_model)
    if deactivate:
        model = disable_dropout(model)
    if model_type == "TabResnet":
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "resnet"
            ), "Check the head is configured for a ResNet architecture"
        else:
            head_params["main_body"] = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.resnet.blocks[-1].linear_second.out_features
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["d_in"] = model.resnet.blocks[-1].linear_second.out_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadConfidNet(**head_params)
        model.resnet.head = head
    elif model_type == "TabFTTransformer":
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "transformer"
            ), "Check the head is configured for a transformer architecture"
        else:
            head_params["main_body"] = "transformer"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.ftt.transformer.blocks[-1].ffn_normalization.normalized_shape[
                    0
                ]
            ), "Check the input of the head corresponds to the last one of the transformer"
        else:
            head_params["d_in"] = model.ftt.transformer.blocks[
                -1
            ].ffn_normalization.normalized_shape[0]
        if "batch_norm" in head_params.keys():
            assert (
                head_params["batch_norm"] == "layer_norm"
            ), "Check the head has set a layer_norm parameter"
        else:
            head_params["batch_norm"] = "layer_norm"
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadConfidNet(**head_params)
        model.ftt.transformer.head = head
    elif model_type == "VGG16":
        head_params["main_body"] = "VGG"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.classifier[0].in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.classifier[0].in_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == False
            ), "Check the input of the head is not normalized"
        else:
            head_params["pre_norm"] = False
        print(head_params)
        head = HeadConfidNet(**head_params)
        model.classifier = head
    elif model_type == "VGG16bn":
        head_params["main_body"] = "VGG"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.classifier[0].in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.classifier[0].in_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == False
            ), "Check the input of the head is not normalized"
        else:
            head_params["pre_norm"] = False
        print(head_params)
        head = HeadConfidNet(**head_params)
        model.classifier = head
    elif model_type == "Resnet34":
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadConfidNet(**head_params)
        model.fc = head
    elif model_type == "Resnet18":
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadConfidNet(**head_params)
        model.fc = head
    elif model_type == "Resnet50":
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"] == model.fc.in_features
            ), "Check that the size of the last features layer corresponds with the initial of the head"
        else:
            head_params["d_in"] = model.fc.in_features
        # if "pre_norm" in head_params.keys():
        #     assert (
        #         head_params["pre_norm"] == False
        #     ), "Check the input of the head is not normalized"
        # else:
        #     head_params["pre_norm"] = False
        print(head_params)
        head = HeadConfidNet(**head_params)
        model.fc = head
    return model


def build_model_tabular(trial, model_type, train_=None, meta="plugin"):
    """
    Args:
        trial: the Optuna trial
        model_type: str
            A string for the model type selected
        train_: TabularDataset
            A Tabular dataset. Default is None.
        meta: str
            The kind of model to build.
    Returns:
        object: torch.Module
        A neural network with parameters extracted from the trial
    """
    try:
        cat_dim = (train_.x_cat.max(dim=0).values + 1).tolist()
    except:
        cat_dim = []
    n_classes = len(train_.classes)
    if meta in ["sat", "dg", "sat_te"]:
        n_classes += 1
    if model_type == "TabFTTransformer":
        body = {
            "d_in": train_.x_num.shape[1],
            "cat_cardinalities": cat_dim,
            "d_out": n_classes,
        }
        body["n_blocks"] = trial.suggest_int("n_blocks", 1, 4, 1)
        body["d_token"] = trial.suggest_int("d_token", 64, 512, 64)
        body["n_blocks"] = trial.suggest_int("n_blocks", 1, 4, 1)
        body["attention_dropout"] = trial.suggest_float(
            "attention_dropout", 0, 0.5, step=0.05
        )
        res_drop = trial.suggest_categorical("res_drop", [0, 1])
        if res_drop == 0:
            body["residual_dropout"] = 0
        else:
            body["residual_dropout"] = trial.suggest_float(
                "residual_dropout", 0, 0.2, step=0.05
            )
        body["ffn_dropout"] = trial.suggest_float("ffn_dropout", 0, 0.5, step=0.05)
        ffn_factor = trial.suggest_float("ffn_factor", 2 / 3, 8 / 3, step=1 / 3)
        body["ffn_d_hidden"] = int(ffn_factor * body["d_token"])
        if body["ffn_d_hidden"] % 2 == 1:
            body["ffn_d_hidden"] += 1
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {
                "main_body": "transformer",
                "d_out": n_classes,
                "pre_norm": True,
            }
            model = buildSelNet(
                model_type="TabFTTransformer", body_params=body, head_params=head
            )
        else:
            model = TabFTTransformer(**body)
    elif model_type == "TabResnet":
        body = {
            "d_in": train_.x_num.shape[1],
            "cat_cardinalities": cat_dim,
            "d_out": n_classes,
        }
        body["n_blocks"] = trial.suggest_int("n_blocks", 1, 4, 1)
        body["d_token"] = trial.suggest_int("d_token", 64, 512, 64)
        body["d_main"] = trial.suggest_int("d_main", 64, 512, 64)
        hidden_factor = trial.suggest_int("hidden_factor", 1, 4, 1)
        body["d_hidden"] = int(hidden_factor * body["d_main"])
        body["dropout_first"] = trial.suggest_float("dropout_first", 0, 0.5, step=0.05)
        body["dropout_second"] = trial.suggest_float(
            "dropout_second", 0, 0.5, step=0.05
        )
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes, "pre_norm": True}
            model = buildSelNet(
                model_type="TabResnet", body_params=body, head_params=head
            )
        else:
            model = TabCatResNet(**body)
    else:
        model = None
    return model


def build_model_image(trial, model_type, train_=None, meta="plugin", inp=64):
    """

    Args:
        trial: the Optuna trial
        model_type: str
            A string for the model type selected.
            Possible values are:
                - 'VGG';
                - 'Resnet34';
                - 'Resnet50'
        train_: ImgFolder
            An Image dataset. Default is None.
        meta: str
            The kind of model to build.
        inp: int
            The size of the input. Used if VGG is selected as a model_type

    Returns:
        object: torch.Module
        A neural network with parameters extracted from the trial
    """
    if type(train_.classes) == list:
        n_classes = len(train_.classes)
    elif type(train_.classes) == int:
        n_classes = train_.classes
    else:
        n_classes = None
    if meta in ["sat", "dg", "sat_em", "sat_te"]:
        n_classes += 1
    if model_type == "VGG":
        body = {"d_out"}
        body = {"input_size": inp, "d_out": n_classes}
        batch_norm = trial.suggest_categorical("b_norm", [True, False])
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "VGG", "d_out": n_classes, "pre_norm": False}
            if batch_norm:
                model = buildSelNet(
                    model_type="VGG16bn", body_params=body, head_params=head
                )
            else:
                model = buildSelNet(
                    model_type="VGG16", body_params=body, head_params=head
                )

        else:
            if batch_norm:
                model = vgg16_bn(**body)
            else:
                model = vgg16(**body)
    elif model_type == "Resnet34":
        body = {"block": BasicBlock, "layers": [3, 4, 6, 3], "num_classes": n_classes}
        body["zero_init_residual"] = trial.suggest_categorical(
            "zero_init_residual", [True, False]
        )

        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet34", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    elif model_type == "Resnet18":
        body = {"block": BasicBlock, "layers": [2, 2, 2, 2], "num_classes": n_classes}
        body["zero_init_residual"] = trial.suggest_categorical(
            "zero_init_residual", [True, False]
        )

        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet18", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    elif model_type == "Resnet50":
        body = {"block": Bottleneck, "layers": [3, 4, 6, 3], "num_classes": n_classes}
        body["zero_init_residual"] = trial.suggest_categorical(
            "zero_init_residual", [True, False]
        )

        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet50", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    else:
        model = None
    return model


def get_num_classes(training_set):
    """

    Args:
        training_set: ImgFolder or TabularDataset

    Returns:
        int:
            the number of classes for a specific dataset
    """
    if hasattr(training_set, "classes"):
        n_classes = len(training_set.classes)
    elif hasattr(training_set, "datasets"):
        n_classes = len(training_set.datasets[1].classes)
    elif hasattr(training_set, "dataset"):
        if hasattr(training_set.dataset, "classes"):
            n_classes = len(training_set.dataset.classes)
        elif hasattr(training_set.dataset, "datasets"):
            n_classes = len(training_set.dataset.datasets[1].classes)
    else:
        n_classes = "NotFound"
    return n_classes


def get_datatype(training_set):
    """

    Args:
        training_set: ImgFolder or FkeData or TabularDataset

    Returns:
        bool True if the dataset is tabular, False otherwise
    """
    if hasattr(training_set, "classes"):
        if type(training_set) == ImgFolder:
            tab = False
        else:
            tab = True
    elif hasattr(training_set, "datasets"):
        if type(training_set.datasets[1]) == ImgFolder:
            tab = False
        elif type(training_set.datasets[1]) == TabularDataset:
            tab = True
    elif hasattr(training_set, "dataset"):
        if type(training_set.dataset) == FakeData:
            tab = False
        else:
            if hasattr(training_set.dataset, "classes"):
                if type(training_set.dataset) == ImgFolder:
                    tab = False

                elif type(training_set.dataset) == TabularDataset:
                    tab = True
            elif hasattr(training_set.dataset, "datasets"):
                if type(training_set.dataset.datasets[1]) == ImgFolder:
                    tab = False
                elif type(training_set.dataset.datasets[1]) == TabularDataset:
                    tab = True
    else:
        tab = None
    if type(training_set) == FkeData:
        tab = False
    return tab


def compute_loss_sat(
    y, outputs, indices, num_examples, n_classes, pretrain, momentum, epoch
):
    if epoch > pretrain:
        loss_f = SelfAdaptiveTraining(
            num_examples=num_examples,
            num_classes=n_classes,
            mom=momentum,
        )
        loss = loss_f(y, outputs, indices)
    else:
        loss = compute_loss_ce(y, outputs[:, :-1])
    return loss


def compute_loss_satte(
    y, outputs, indices, num_examples, n_classes, pretrain, momentum, epoch, beta
):
    if epoch > pretrain:
        loss = compute_loss_sat(
            y, outputs, indices, num_examples, n_classes, pretrain, momentum, epoch
        )
        # if criterion == "sat_te":
        loss_en = entropy_term(outputs[:, :-1])
        loss += beta * loss_en
    else:
        loss = compute_loss_ce(y, outputs[:, :-1])
    return loss


def compute_loss_selnet(y, hg, aux, lamda, coverage, alpha):
    loss1 = cross_entropy_selection_vectorized(y, hg, lamda=lamda, c=coverage)
    loss2 = cross_entropy_vectorized(y, aux)
    loss = (alpha * loss1) + ((1 - alpha) * loss2)
    return loss


def compute_loss_selnette(y, hg, aux, lamda, coverage, alpha, beta):
    loss = compute_loss_selnet(y, hg, aux, lamda, coverage, alpha)
    loss_en = entropy_term(hg[:, :-1])
    loss += beta * loss_en
    return loss


def compute_loss_ce(y, outputs):
    loss = torch.nn.functional.cross_entropy(outputs, y)
    return loss


def train(
    model: nn.Module,
    device: str,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: str,
    train_dl: torch.utils.data.DataLoader,
    lamda: int = 32,
    alpha: float = 0.5,
    coverage: float = 0.9,
    beta: float = 0.01,
    pretrain: int = 0,
    reward: float = 2.0,
    td: bool = True,
    gamma: float = 0.5,
    epochs_lr: list = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299],
    momentum: float = 0.99,
    verbose: bool = True,
    seed: int = 42,
    save_interm: bool = True,
    path_interm: str = "default",
    model_base_confnet: nn.Module = None,
):
    """

    Args:
        model: torch.Module
            The model to train.
        device: str
            The string for the device to use during training.
        epochs: int
            The number of epochs for the training.
        optimizer: torch.optim
            The optimizer to use during the training.
        criterion: str
            A string to indicate which loss to use during the training.
            Possible values are:
                - 'ce' for cross-entropy
                - 'selnet' for SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' Deep Gamblers loss (Liu et al., 2023)
        train_dl: torch.utils.data.DataLoader
            A dataloader for training dataset
        lamda: int
            The value of parameter $ \lambda $ in Selective Net loss
        alpha: float [0;1]
            The value of parameter $ \alpha $ in Selective Net loss
        coverage: float [0;1]
            The value of parameter $c$ in Selective Net loss
        beta: float
            The value of parameter $ \beta $ for additional entropy term
        pretrain: int
            The value of pretraining using cross-entropy for SAT.
        reward: float
            The value of reward for DeepGamblers
        td: bool
            If True, time decay is applied in all the epochs specified in epochs_lr parameter.
        gamma: float
            The value that decreases learning rate if td is True.
        epochs_lr: list
            The list of epochs when time decay decreases learning rate
        momentum: float
            The value of $ \gamma $ in SAT loss
        verbose: bool
            If True, it prints epochs training.

    Returns:
        model: torch.Module
            The trained model.
    """
    model.train()
    model.to(device)
    n = len(train_dl.dataset)
    n_classes = get_num_classes(train_dl.dataset)
    print("Number of dataset classes is: {}".format(n_classes))
    tabular = get_datatype(train_dl.dataset)
    if criterion in ["sat", "sat_em", "sat_te", "dg"]:
        n_classes += 1
    print("Number of classes for training is: {}".format(n_classes))
    print("\n criterion is {} \n".format(criterion))
    set_seed(seed)
    for epoch in range(1, epochs + 1):
        if td:
            if epoch in epochs_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= gamma
                    print("\n: lr now is: {}\n".format(param_group["lr"]))
        running_loss = 0
        if verbose:
            if tabular:
                with tqdm(train_dl, unit="batch") as tbatch:
                    for i, batch in enumerate(tbatch):
                        tbatch.set_description(
                            "Epoch {} - dev {}".format(epoch, device)
                        )
                        x_num, x_cat, y, indices = batch
                        x_num, x_cat, y = (
                            x_num.to(device),
                            x_cat.to(device),
                            y.to(device),
                        )
                        if len(y) == 1:
                            pass
                        else:
                            optimizer.zero_grad()
                            if "selnet" in criterion:
                                hg, aux = model.forward(x_num, x_cat)
                                loss1 = cross_entropy_selection_vectorized(
                                    y, hg, lamda=lamda, c=coverage
                                )
                                loss2 = cross_entropy_vectorized(y, aux)
                                loss = (alpha * loss1) + ((1 - alpha) * loss2)
                                if criterion == "selnet_te":
                                    loss3 = entropy_term(hg[:, :-1])
                                    loss += beta * loss3
                            else:
                                outputs = model.forward(x_num, x_cat)
                                if "sat" in criterion:
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    if epoch > pretrain:
                                        if (epoch == pretrain + 1) & (i == 0):
                                            print("switching to Adaptive")
                                        loss_f = SelfAdaptiveTraining(
                                            num_examples=n,
                                            num_classes=n_classes,
                                            mom=momentum,
                                        )
                                        try:
                                            loss = loss_f(y, outputs, indices)
                                        except:
                                            import pdb

                                            pdb.set_trace()
                                        if criterion == "sat_te":
                                            loss3 = entropy_term(outputs[:, :-1])
                                            loss += beta * loss3
                                    else:
                                        if (epoch == 1) & (i == 0):
                                            print(
                                                "\n training with Cross Entropy until epoch {}\n".format(
                                                    pretrain + 1
                                                )
                                            )
                                        loss = torch.nn.functional.cross_entropy(
                                            outputs[:, :-1], y
                                        )
                                elif criterion == "ce":
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    loss = torch.nn.functional.cross_entropy(outputs, y)
                                elif criterion in ["cn", "sele", "reg"]:
                                    if model_base_confnet is None:
                                        raise ValueError(
                                            "The base model is not spcified"
                                        )
                                    else:
                                        if (epoch == 1) & (i == 0):
                                            model_base_confnet.eval()
                                    outputs_cl = model_base_confnet.forward(
                                        x_num, x_cat
                                    )
                                    out = torch.cat([outputs_cl, outputs], dim=1)
                                    if criterion == "cn":
                                        loss = MSE_confid_loss(
                                            y, out, num_classes=n_classes, device=device
                                        )
                                    elif criterion == "reg":
                                        loss = reg_loss(y, out)
                                    elif criterion == "sele":
                                        loss = sele_loss(y, out)
                                elif criterion == "dg":
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    loss = deep_gambler_loss(y, outputs, reward)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            r = torch.cuda.memory_reserved(0)
                            a = torch.cuda.memory_allocated(0)
                            f = (r - a) / (1024**2)
                            tbatch.set_postfix(
                                loss=loss.item(),
                                average_loss=running_loss / (i + 1),
                                memory=f,
                            )
            else:
                with tqdm(train_dl, unit="batch") as tbatch:
                    for i, batch in enumerate(tbatch):
                        tbatch.set_description(
                            "Epoch {} - dev {}".format(epoch, device)
                        )
                        x, y, indices = batch
                        x, y = x.to(device), y.to(device)
                        if len(y) == 1:
                            pass
                        else:
                            optimizer.zero_grad()
                            if criterion in ["selnet", "selnet_te"]:
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                hg, aux = model.forward(x)
                                loss1 = cross_entropy_selection_vectorized(
                                    y, hg, lamda=lamda, c=coverage
                                )
                                loss2 = cross_entropy_vectorized(y, aux)
                                loss = (alpha * loss1) + ((1 - alpha) * loss2)
                                if criterion == "selnet_te":
                                    loss3 = entropy_term(hg[:, :-1])
                                    loss += beta * loss3
                            else:
                                outputs = model.forward(x)
                                if "sat" in criterion:
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    if epoch > pretrain:
                                        if (epoch == pretrain + 1) & (i == 0):
                                            print("switching to Adaptive")
                                        loss_f = SelfAdaptiveTraining(
                                            num_examples=n,
                                            num_classes=n_classes,
                                            mom=momentum,
                                        )
                                        loss = loss_f(y, outputs, indices)
                                        if criterion == "sat_te":
                                            loss3 = entropy_term(outputs[:, :-1])
                                            loss += beta * loss3
                                    else:
                                        if (epoch == 1) & (i == 0):
                                            print(
                                                "\n training with Cross Entropy until epoch {}\n".format(
                                                    pretrain + 1
                                                )
                                            )
                                        loss = torch.nn.functional.cross_entropy(
                                            outputs[:, :-1], y
                                        )
                                elif criterion == "ce":
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    loss = torch.nn.functional.cross_entropy(outputs, y)
                                elif criterion in ["cn", "sele", "reg"]:
                                    if model_base_confnet is None:
                                        raise ValueError(
                                            "The base model is not spcified"
                                        )
                                    else:
                                        if (epoch == 1) & (i == 0):
                                            model_base_confnet.eval()
                                    outputs_cl = model_base_confnet.forward(x)
                                    out = torch.cat([outputs_cl, outputs], dim=1)
                                    if criterion == "cn":
                                        loss = MSE_confid_loss(
                                            y, out, num_classes=n_classes, device=device
                                        )
                                    elif criterion == "reg":
                                        loss = reg_loss(y, out)
                                    elif criterion == "sele":
                                        loss = sele_loss(y, out)
                                elif criterion == "dg":
                                    if (epoch == 1) & (i == 0):
                                        print("\n criterion is {} \n".format(criterion))
                                    loss = deep_gambler_loss(y, outputs, reward)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            r = torch.cuda.memory_reserved(0)
                            a = torch.cuda.memory_allocated(0)
                            f = (r - a) / (1024**2)
                            tbatch.set_postfix(
                                loss=loss.item(),
                                average_loss=running_loss / (i + 1),
                                memory=f,
                            )
        else:
            if tabular:
                for i, batch in enumerate(train_dl):
                    x_num, x_cat, y, indices = batch
                    x_num, x_cat, y = (
                        x_num.to(device),
                        x_cat.to(device),
                        y.to(device),
                    )
                    if len(y) == 1:
                        pass
                    else:
                        optimizer.zero_grad()
                        if "selnet" in criterion:
                            if (epoch == 1) & (i == 0):
                                print("\n criterion is {} \n".format(criterion))
                            hg, aux = model.forward(x_num, x_cat)
                            loss1 = cross_entropy_selection_vectorized(
                                y, hg, lamda=lamda, c=coverage
                            )
                            loss2 = cross_entropy_vectorized(y, aux)
                            loss = (alpha * loss1) + ((1 - alpha) * loss2)
                            if criterion == "selnet_te":
                                loss3 = entropy_term(hg[:, :-1])
                                loss += beta * loss3
                        else:
                            outputs = model.forward(x_num, x_cat)
                            if "sat" in criterion:
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                if epoch > pretrain:
                                    if (epoch == pretrain + 1) & (i == 0):
                                        print("switching to Adaptive")
                                    loss_f = SelfAdaptiveTraining(
                                        num_examples=n,
                                        num_classes=n_classes,
                                        mom=momentum,
                                    )
                                    loss = loss_f(y, outputs, indices)
                                    if criterion == "sat_te":
                                        loss3 = entropy_term(outputs[:, :-1])
                                        loss += beta * loss3
                                else:
                                    if (epoch == 1) & (i == 0):
                                        print(
                                            "\n training with Cross Entropy until epoch {}\n".format(
                                                pretrain + 1
                                            )
                                        )
                                    loss = torch.nn.functional.cross_entropy(
                                        outputs[:, :-1], y
                                    )
                            elif criterion in ["cn", "sele", "reg"]:
                                if model_base_confnet is None:
                                    raise ValueError("The base model is not spcified")
                                else:
                                    if (epoch == 1) & (i == 0):
                                        model_base_confnet.eval()
                                outputs_cl = model_base_confnet.forward(x_num, x_cat)
                                out = torch.cat([outputs_cl, outputs], dim=1)
                                if criterion == "cn":
                                    loss = MSE_confid_loss(
                                        y, out, num_classes=n_classes, device=device
                                    )
                                elif criterion == "reg":
                                    loss = reg_loss(y, out)
                                elif criterion == "sele":
                                    loss = sele_loss(y, out)
                            elif criterion == "ce":
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                loss = torch.nn.functional.cross_entropy(outputs, y)
                            elif criterion == "dg":
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                loss = deep_gambler_loss(y, outputs, reward)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if (epoch % 50 == 0) & (i == len(train_dl) - 1):
                            print(running_loss / len(train_dl))
            else:
                for i, batch in enumerate(train_dl):
                    x, y, indices = batch
                    x, y = x.to(device), y.to(device)
                    if len(y) == 1:
                        pass
                    else:
                        optimizer.zero_grad()
                        if criterion in ["selnet", "selnet_em", "selnet_te"]:
                            if (epoch == 1) & (i == 0):
                                print("\n criterion is {} \n".format(criterion))
                            hg, aux = model.forward(x)
                            loss1 = cross_entropy_selection_vectorized(
                                y, hg, lamda=lamda, c=coverage
                            )
                            loss2 = cross_entropy_vectorized(y, aux)
                            loss = (alpha * loss1) + ((1 - alpha) * loss2)
                            # if criterion == "selnet_em":
                            #     loss3 = entropy_loss(hg[:, :-1])
                            #     loss += beta * loss3
                            if criterion == "selnet_te":
                                loss3 = entropy_term(hg[:, :-1])
                                loss += beta * loss3
                        else:
                            outputs = model.forward(x)
                            if "sat" in criterion:
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                if epoch > pretrain:
                                    if (epoch == pretrain + 1) & (i == 0):
                                        print("switching to Adaptive")
                                    loss_f = SelfAdaptiveTraining(
                                        num_examples=n,
                                        num_classes=n_classes,
                                        mom=momentum,
                                    )
                                    loss = loss_f(y, outputs, indices)
                                    if criterion == "sat_te":
                                        loss3 = entropy_term(outputs[:, :-1])
                                        loss += beta * loss3

                                else:
                                    if (epoch == 1) & (i == 0):
                                        print(
                                            "\n training with Cross Entropy until epoch {}\n".format(
                                                pretrain + 1
                                            )
                                        )
                                    loss = torch.nn.functional.cross_entropy(
                                        outputs[:, :-1], y
                                    )
                            elif criterion == "ce":
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                loss = torch.nn.functional.cross_entropy(outputs, y)
                            elif criterion in ["cn", "sele", "reg"]:
                                if model_base_confnet is None:
                                    raise ValueError("The base model is not spcified")
                                else:
                                    if (epoch == 1) & (i == 0):
                                        model_base_confnet.eval()
                                outputs_cl = model_base_confnet.forward(x)
                                out = torch.cat([outputs_cl, outputs], dim=1)
                                if criterion == "cn":
                                    loss = MSE_confid_loss(
                                        y, out, num_classes=n_classes, device=device
                                    )
                                elif criterion == "reg":
                                    loss = reg_loss(y, out)
                                elif criterion == "sele":
                                    loss = sele_loss(y, out)
                            elif criterion == "dg":
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                loss = deep_gambler_loss(y, outputs, reward)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if (epoch % 50 == 0) & (i == len(train_dl)):
                            print(running_loss / len(train_dl))
    if (epoch in [50, 100, 150, 200, 250, 300]) & (save_interm != False):
        path_to_save = path_interm + "_ep{}.pt".format(epoch)
        torch.save(model.state_dict(), path_to_save)

    return model


# def train_cross():
#


def predict_proba(
    model: nn.Module, device: str, dataloader: torch.utils.data.DataLoader, meta: str
):
    """

    Args:
        model: torch.Module
            The model used for prediction
        device: str
            The device used by pytorch
        dataloader: torch.utils.data.DataLoader
            The dataloader for the set where we want to make predictions
        meta:
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
    Returns:
        scores: np.array

    """
    model.eval()
    # model.to(device)
    y_hat_ = []
    tab = get_datatype(dataloader.dataset)
    if tab:
        for i, batch in enumerate(dataloader):
            x_num, x_cat, y, indices = batch
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            if "selnet" in meta:
                hg, aux = model(x_num, x_cat)
                y_hat_batch = (
                    torch.nn.functional.softmax(hg[:, :-1], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                out = model(x_num, x_cat)
                if meta in ["plugin", "cn", "sele", "reg"]:
                    y_hat_batch = torch.nn.functional.softmax(out, dim=1).detach().cpu()
                else:
                    y_hat_batch = (
                        torch.nn.functional.softmax(out[:, :-1], dim=1).detach().cpu()
                    )
            y_hat_.append(y_hat_batch)
    elif tab == False:
        print("predicting")
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print("batch {} out of {}".format(i, len(dataloader)))
            x, y, indices = batch
            x, y = x.to(device), y.to(device)
            if "selnet" in meta:
                hg, aux = model(x)
                y_hat_batch = (
                    torch.nn.functional.softmax(hg[:, :-1], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                out = model(x)
                if meta in ["plugin", "cn", "sele", "reg"]:
                    y_hat_batch = torch.nn.functional.softmax(out, dim=1).detach().cpu()
                else:
                    y_hat_batch = (
                        torch.nn.functional.softmax(out[:, :-1], dim=1).detach().cpu()
                    )

            y_hat_.append(y_hat_batch)
    return np.vstack(y_hat_)


def predict(
    model: nn.Module, device: str, dataloader: torch.utils.data.DataLoader, meta: str
):
    """
        Args:
        model: torch.Module
            The model used for prediction
        device: str
            The device used by pytorch
        dataloader: torch.utils.data.DataLoader
            The dataloader for the set where we want to make predictions
        meta:
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
    Returns:
        scores: np.array
    """
    model.eval()
    model.to(device)
    return np.argmax(predict_proba(model, device, dataloader, meta), axis=1)


def kl_divergence(p, q):
    """
    method to compute the KL-divergence
    between two probability distributions
    """

    # normalize probabilities if they are not for rounding reasons
    if np.sum(p.sum(axis=1) != 1) > 0:
        p = p / ((p.sum(axis=1)).reshape(-1, 1))
    if np.sum(q.sum(axis=1) != 1) > 0:
        q = q / ((q.sum(axis=1)).reshape(-1, 1))
    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = scipy.stats.entropy(p, q, axis=1)
    #     # compute the Jensen Shannon Distance
    #     distance = np.sqrt(divergence)
    divergence = np.where(divergence < 0, 0, divergence)
    return divergence


def get_confidence_ensemble(scores_all_models, avg_score):
    vals = np.sum(
        [
            kl_divergence(scores_all_models[l], avg_score)
            for l in range(len(scores_all_models))
        ],
        axis=0,
    )
    return vals


def predict_conf(
    model: nn.Module, device: str, dataloader: torch.utils.data.DataLoader, meta: str
):
    """

    Args:
        model: torch.Module
            The model used for prediction
        device: str
            The device used by pytorch
        dataloader: torch.utils.data.DataLoader
            The dataloader for the set where we want to make predictions
        meta:
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'cn' for a network trained using ConfidNet (Corbière et al., 2020)
                - 'sele' for a network trained using SELE (Franc et al., 2023)
                - 'reg' for a network trained using REG (Franc et al., 2023)
    Returns:
        confidences: np.array

    """
    model.eval()
    # model.to(device)
    sel_ = []
    if type(dataloader.dataset) == TabularDataset:
        for i, batch in enumerate(dataloader):
            x_num, x_cat, y, indices = batch
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            if "selnet" in meta:
                hg, aux = model(x_num, x_cat)
                if meta in ["selnet", "selnet_em", "selnet_te"]:
                    sel_batch = hg[:, -1].detach().cpu().numpy().reshape(-1, 1)
                elif meta in ["selnet_em_sr", "selnet_te_sr", "selnet_sr"]:
                    sel_batch = (
                        torch.max(
                            torch.nn.functional.softmax(hg[:, :-1], dim=1), dim=1
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 1)
                    )

            else:
                out = model(x_num, x_cat)
                if meta == "plugin":
                    sel_batch = (
                        torch.max(
                            torch.nn.functional.softmax(out, dim=1).detach().cpu(),
                            dim=1,
                        )[0]
                        .numpy()
                        .reshape(-1, 1)
                    )
                elif meta in ["cn", "sele", "reg"]:
                    sel_batch = torch.sigmoid(out).cpu().detach().numpy()
                else:
                    softout = torch.nn.functional.softmax(out, dim=1).detach().cpu()
                    if meta in ["sat_sr", "sat_em_sr", "sat_te_sr"]:
                        sel_batch = (
                            torch.max(softout[:, :-1], dim=1)[0].numpy().reshape(-1, 1)
                        )
                    else:
                        sel_batch = softout[:, -1].numpy().reshape(-1, 1)
            sel_.append(sel_batch)
    elif (type(dataloader.dataset) == ImgFolder) or (
        type(dataloader.dataset) == FkeData
    ):
        for i, batch in enumerate(dataloader):
            x, y, indices = batch
            x, y = x.to(device), y.to(device)
            if "selnet" in meta:
                hg, aux = model(x)
                if meta in ["selnet", "selnet_em", "selnet_te"]:
                    sel_batch = hg[:, -1].detach().cpu().numpy().reshape(-1, 1)
                elif meta in ["selnet_em_sr", "selnet_te_sr", "selnet_sr"]:
                    sel_batch = (
                        torch.max(
                            torch.nn.functional.softmax(hg[:, :-1], dim=1), dim=1
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 1)
                    )
            else:
                out = model(x)
                if meta == "plugin":
                    sel_batch = (
                        torch.max(
                            torch.nn.functional.softmax(out, dim=1).detach().cpu(),
                            dim=1,
                        )[0]
                        .numpy()
                        .reshape(-1, 1)
                    )
                elif meta in ["cn", "sele", "reg"]:
                    sel_batch = torch.sigmoid(out).cpu().detach().numpy()
                else:
                    softout = torch.nn.functional.softmax(out, dim=1).detach().cpu()
                    if meta in ["sat_sr", "sat_em_sr", "sat_te_sr"]:
                        sel_batch = (
                            torch.max(softout[:, :-1], dim=1)[0].numpy().reshape(-1, 1)
                        )
                    else:
                        sel_batch = softout[:, -1].numpy().reshape(-1, 1)
            sel_.append(sel_batch)
    return np.vstack(sel_).flatten()


def calibrate(
    model: nn.Module or list[nn.Module],
    device: str,
    dataloader: torch.utils.data.DataLoader,
    meta: str,
    coverages=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
):
    """

    Args:
        model: torch.Module
            The model used for prediction
        device: str
            The device used by pytorch
        dataloader: torch.utils.data.DataLoader
            The dataloader for the set where we want to make predictions
        meta:  str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'cn' for a network trained using ConfidNet (Corbière et al., 2020)
                - 'sele' for a network trained using SELE (Franc et al., 2023)
                - 'reg' for a network trained using REG (Franc et al., 2023)
        coverages: list
            The list with desired target coverages $c$. The default is [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    Returns:
        thetas: list
        List of thresholds over the confidence values for every target coverage provided
    """
    covs = sorted(coverages, reverse=True)
    if "selnet" in meta:
        thetas = []
        print(len(model))
        print(len(covs))
        assert (type(model) == list) or (
            type(model) == dict
        ), "Please check you have provided a list of selective nets"
        assert len(model) == len(
            covs
        ), "Please check that you have the same numbers of selective nets and coverages."
        if type(model) == list:
            for i, cov in enumerate(covs):
                model[i].eval()
                model[i].to(device)
                confs = predict_conf(model[i], device, dataloader, meta)
                theta = np.quantile(confs, 1 - cov)
                thetas.append(theta)
        else:
            for i, cov in enumerate(covs):
                model[cov].eval()
                model[cov].to(device)
                confs = predict_conf(model[cov], device, dataloader, meta)
                theta = np.quantile(confs, 1 - cov)
                thetas.append(theta)
    else:
        model.eval()
        model.to(device)
        if meta in ["plugin", "sat_sr", "sat_em_sr", "sat_te_sr", "cn"]:
            confs = predict_conf(model, device, dataloader, meta)
            thetas = []
            for i, cov in enumerate(covs):
                theta = np.quantile(confs, 1 - cov)
                thetas.append(theta)
        elif meta in ["sat", "sat_em", "dg", "sat_te", "sele", "reg"]:
            confs = predict_conf(model, device, dataloader, meta)
            thetas = []
            for i, cov in enumerate(covs):
                theta = np.quantile(confs, cov)
                thetas.append(theta)
    return thetas


def qband(
    model: nn.Module or list[nn.Module],
    device: str,
    testloader: torch.utils.data.DataLoader,
    calloader: torch.utils.data.DataLoader,
    meta: str,
    coverages=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
):
    """
    Function to predict intervals of instances to reject depending on target coverages
     model: torch.Module
            The model used for prediction or a list of selective net models
        device: str
            The device used by pytorch
        testloader: torch.utils.data.DataLoader
            The dataloader for the set where we want to make predictions
        calloader: torch.utils.data.DataLoader
            The dataloader for the set where we use to calibrate confidences
        meta:  str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'cn' for a network trained using ConfidNet (Corbière et al., 2020)
                - 'sele' for a network trained using SELE (Franc et al., 2023)
                - 'reg' for a network trained using REG (Franc et al., 2023)
                - 'pluginauc' for a network trained using PlugInAUC (Pugnana and Ruggieri, 2023b)
        coverages: list
            The list with desired target coverages $c$. The default is [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    """
    covs = sorted(coverages, reverse=True)
    if "selnet" in meta:
        assert (type(model) == list) or (
            type(model) == dict
        ), "Please check you have provided a list or a dictionary of selective nets"
        thetas = calibrate(model, device, calloader, meta, coverages=coverages)
        selected = np.zeros(len(testloader.dataset), dtype=int)
        if type(model) == list:
            for i, cov in enumerate(covs):
                model[i].eval()
                model[i].to(device)
                confs = predict_conf(model[i], device, testloader, meta)
                selected = np.where(confs > thetas[i], 1 + i, selected)
        else:
            for i, cov in enumerate(covs):
                model[cov].eval()
                model[cov].to(device)
                confs = predict_conf(model[cov], device, testloader, meta)
                selected = np.where(confs > thetas[i], 1 + i, selected)
    else:
        model.eval()
        model.to(device)
        if meta in ["plugin", "sat_sr", "sat_te_sr", "cn"]:
            thetas = calibrate(model, device, calloader, meta, coverages=coverages)
            confs = predict_conf(model, device, testloader, meta)
            selected = np.digitize(confs, sorted(thetas), right=False)
        elif meta in ["sat", "dg", "sat_te", "sele", "reg"]:
            thetas = calibrate(model, device, calloader, meta, coverages=coverages)
            confs = predict_conf(model, device, testloader, meta)
            selected = np.zeros(len(testloader.dataset), dtype=int)
            for i, cov in enumerate(covs):
                selected = np.where(confs <= thetas[i], 1 + i, selected)
        elif meta == "pluginauc":
            y_scores = predict_proba(model, device, calloader, "plugin")[:, 1]
            y = np.array(calloader.dataset.targets)
            if len(np.unique(y)) > 2:
                raise NotImplementedError(
                    "PlugInAUC not implemented for non binary classification."
                )
            auc_roc = skm.roc_auc_score(y, y_scores)
            n, npos = len(y), np.sum(y)
            pneg = 1 - np.mean(y)
            u_pos = int(auc_roc * pneg * n)
            pos_sorted = np.argsort(y_scores)
            if isinstance(y, pd.Series):
                tp = np.cumsum(y.iloc[pos_sorted[::-1]])
            else:
                tp = np.cumsum(y[pos_sorted[::-1]])
            l_pos = n - np.searchsorted(tp, auc_roc * npos + 1, side="right")
            quantiles = [1 - cov for cov in coverages]
            pos = (u_pos + l_pos) / 2
            thetas = []
            for q in quantiles:
                delta = int(n * q / 2)
                t1 = y_scores[pos_sorted[max(0, round(pos - delta))]]
                t2 = y_scores[pos_sorted[min(round(pos + delta), n - 1)]]
                thetas.append([t1, t2])
            m = len(quantiles)
            confs = predict_proba(model, device, testloader, "plugin")[:, 1]
            selected = np.zeros(len(confs)) + m
            for i, t in enumerate(reversed(thetas)):
                t1, t2 = t[0], t[1]
                selected[((t1 <= confs) & (confs <= t2))] = m - i - 1
    return selected


def get_metrics(
    params: dict,
    num_classes: 2,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_scores: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: int,
    dataset,
    perc_train: float,
):
    """

    Args:
        params: dict
            The dictionry of parameters that were used to train the model.
        num_classes: int
            The number of classes in the test set
        coverages: list
            A list of desired coverages for evaluating the selective classifiers
        true: np.array
            The array with true labels
        selected: np.array
            The array provided by qband function, with instances labelled according to increasing levels of coverage
        y_scores: np.array
            The array for the scores
        y_preds: np.array
            The array for predictions
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'cn' for a network trained using ConfidNet (Corbière et al., 2020)
                - 'sele' for a network trained using SELE (Franc et al., 2023)
                - 'reg' for a network trained using REG (Franc et al., 2023)
        trial_num: int
            The number of trial
        dataset: str
            The dataset name
        perc_train:
            The percentage of positive class in the training set
    Returns:

    """
    cols = [col for col in params.keys()]
    parameters = [el for el in params.values()]
    results = pd.DataFrame()
    violations = []
    losses = []
    cover = sorted(coverages, reverse=True)
    for i, cov in enumerate(cover):
        if "selnet" in meta:
            if type(y_scores) == list:
                scores = y_scores[i]
                preds = y_preds[i]
            else:
                scores = y_scores[cov]
                preds = y_preds[cov]
        else:
            scores = y_scores
            preds = y_preds
        if (scores[selected > i].shape[0] == 0) | (scores[selected <= i].shape[0] == 0):
            coverage = -1
            viol = 1
            loss = 1
            acc = -1
            print(acc)
            tmp = pd.DataFrame([parameters], columns=cols)
            if num_classes == 2:
                auc = -1
                pos_rate = -1
                brier = 1
                bss = -1
                auc_rej = -1
                pos_rate_rej = -1
                brier_rej = 1
                bss_denom_rej = -1
                bss_rej = -1
            if num_classes >= 2:
                aucmu = -1
                aucmu_rej = -1
            err_rejected = 1
            acc_rejected = -1
        else:
            coverage = len(selected[selected > i]) / len(selected)
            viol = abs((len(selected[selected > i]) / len(selected) - cov))
            try:
                loss = skm.log_loss(
                    true[selected > i],
                    scores[selected > i],
                    labels=[i for i in range(num_classes)],
                )
            except ValueError:
                loss = 1
            acc = skm.accuracy_score(true[selected > i], preds[selected > i])
            if num_classes == 2:
                try:
                    auc = skm.roc_auc_score(
                        true[selected > i], scores[:, 1][selected > i]
                    )
                except ValueError:
                    auc = -1
                try:
                    auc_rej = skm.roc_auc_score(
                        true[selected <= i], scores[:, 1][selected <= i]
                    )
                except ValueError:
                    auc_rej = -1
                pos_rate = np.sum(true[selected > i]) / len(true[selected > i])
                try:
                    brier = skm.brier_score_loss(
                        true[selected > i], scores[:, 1][selected > i]
                    )
                    bss_denom = skm.brier_score_loss(
                        true[selected > i],
                        np.repeat(perc_train, len(true))[selected > i],
                    )
                except ValueError:
                    brier = 1
                    bss_denom = 1
                try:
                    brier_rej = skm.brier_score_loss(
                        true[selected <= i], scores[:, 1][selected <= i]
                    )
                    bss_denom_rej = skm.brier_score_loss(
                        true[selected <= i],
                        np.repeat(perc_train, len(true))[selected <= i],
                    )
                except ValueError:
                    brier_rej = 1
                    bss_denom_rej = 1
                bss = 1 - brier / bss_denom
                pos_rate_rej = np.sum(true[selected <= i]) / len(true[selected <= i])
                bss_rej = 1 - brier / bss_denom_rej
            elif num_classes > 2:
                try:
                    aucmu = auc_mu(true[selected > i], scores[selected > i])
                except:
                    aucmu = -1
                try:
                    aucmu_rej = auc_mu(true[selected <= i], scores[selected <= i])
                except:
                    aucmu_rej = -1
            try:
                err_rejected = skm.log_loss(
                    true[selected <= i],
                    scores[selected <= i],
                    labels=[i for i in range(num_classes)],
                )
                acc_rejected = skm.accuracy_score(
                    true[selected <= i], preds[selected <= i]
                )
            except ValueError:
                err_rejected = 1
                acc_rejected = -1
        tmp = pd.DataFrame([parameters], columns=cols)
        tmp["accuracy"] = acc
        tmp["ce_loss"] = loss
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["accuracy"] = acc
        tmp["ce_loss"] = loss
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["reject_rate"] = 1 - coverage
        tmp["ce_loss_rej"] = err_rejected
        tmp["accuracy_rej"] = acc_rejected
        if num_classes == 2:
            tmp["auc"] = auc
            tmp["auc_rej"] = auc_rej
            tmp["pos_rate"] = pos_rate
            tmp["bss"] = bss
            tmp["brier"] = brier
            tmp["bss_rej"] = bss_rej
            tmp["brier_rej"] = brier_rej
            tmp["pos_rate_rej"] = pos_rate_rej
        elif num_classes > 2:
            tmp["auc"] = aucmu
            tmp["auc_rej"] = aucmu_rej

        results = pd.concat([results, tmp], axis=0)
        violations.append(viol)
        losses.append(loss)
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    return results, violations, losses


def get_metrics_test(
    num_classes: 2,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_scores: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: str or int,
    dataset: str,
    arch: str,
    most_common_class: int = 0,
):
    """
    Function to get results over different coverages (so we store everything)
        num_classes: int
            The number of classes in the test set
        coverages: list
            A list of desired coverages for evaluating the selective classifiers
        true: np.array
            The array with true labels
        selected: np.array
            The array provided by qband function, with instances labelled according to increasing levels of coverage
        y_scores: np.array
            The array for the scores
        y_preds: np.array
            The array for predictions
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'cn' for a network trained using ConfidNet (Corbière et al., 2020)
                - 'sele' for a network trained using SELE (Franc et al., 2023)
                - 'reg' for a network trained using REG (Franc et al., 2023)
        trial_num: int
            The number of trial
        dataset: str
            The dataset name
        arch: str
            The base network architecture
        most_common_class: str
            The label of most common class in the training set

    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    for i, cov in enumerate(cover):
        if "selnet" in meta:
            if type(y_scores) == list:
                scores = y_scores[i]
                preds = y_preds[i]
            else:
                scores = y_scores[cov]
                preds = y_preds[cov]
        else:
            scores = y_scores
            preds = y_preds
        if scores[selected > i].shape[0] == 0:
            coverage = -1
            viol = 1
            acc = -1
            acc_stupid1 = 1
            print(acc)
            if num_classes == 2:
                auc = -1
                pos_rate = -1
                auc_rej = -1
        else:
            coverage = len(selected[selected > i]) / len(selected)
            viol = abs((len(selected[selected > i]) / len(selected) - cov))
            acc = skm.accuracy_score(true[selected > i], preds[selected > i])
            acc_stupid1 = skm.accuracy_score(
                true[selected > i],
                np.repeat(most_common_class, len(true))[selected > i],
            )
            if num_classes == 2:
                try:
                    auc = skm.roc_auc_score(
                        true[selected > i], scores[:, 1][selected > i]
                    )
                except ValueError:
                    auc = -1
                try:
                    auc_rej = skm.roc_auc_score(
                        true[selected <= i], scores[:, 1][selected <= i]
                    )
                except ValueError:
                    auc_rej = -1
                pos_rate = np.sum(true[selected > i]) / len(true[selected > i])
        tmp = pd.DataFrame()
        tmp["accuracy"] = [acc]
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["accuracy"] = acc
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["reject_rate"] = 1 - coverage
        tmp["viol"] = viol
        tmp["acc_stupid1"] = acc_stupid1
        # tmp["relative_accuracy1"] = 1 - (acc / acc_stupid1)
        # tmp["relative_error1"] = 1 - (1 - acc) / (1 - acc_stupid1)
        if num_classes == 2:
            tmp["auc"] = auc
            tmp["auc_rej"] = auc_rej
            tmp["pos_rate"] = pos_rate
            tmp["true_pos"] = np.sum(true)
            tmp["num_positives"] = np.sum(true[selected > i])
            tmp["num_acc"] = len(true[selected > i])
            tmp["false_positives_acc"] = np.sum(
                true[(preds != true) & (true == 1) & (selected > i)]
            )
            tmp["false_negatives_acc"] = np.sum(
                true[(preds != true) & (true == 0) & (selected > i)]
            )
            tmp["false_positives_rej"] = np.sum(
                true[(preds != true) & (true == 1) & (selected <= i)]
            )
            tmp["false_negatives_rej"] = np.sum(
                true[(preds != true) & (true == 0) & (selected <= i)]
            )
        elif num_classes > 2:
            tmp["num_acc"] = len(true[selected > i])
        for label in np.unique(true):
            try:
                tmp["accepted_class{}".format(int(label))] = len(
                    true[(selected > i) & (true == label)]
                )
            except:
                tmp["accepted_class{}".format(int(label))] = 0
            try:
                tmp["total_class{}".format(int(label))] = len(true[true == label])
            except:
                tmp["total_class{}"] = -1
        results = pd.concat([results, tmp], axis=0)
    tmp["max_accepted"] = tmp[
        [cols for cols in tmp.columns if "accepted_class" in cols]
    ].max(axis=1)
    tmp["min_accepted"] = tmp[
        [cols for cols in tmp.columns if "accepted_class" in cols]
    ].min(axis=1)
    tmp["min_max_ratio_accepted"] = np.where(
        tmp["max_accepted"] > 0, tmp["min_accepted"] / tmp["max_accepted"], 0
    )
    tmp["max_total"] = tmp[[cols for cols in tmp.columns if "total_class" in cols]].max(
        axis=1
    )
    tmp["min_total"] = tmp[[cols for cols in tmp.columns if "total_class" in cols]].min(
        axis=1
    )
    tmp["min_max_ratio_total"] = np.where(
        tmp["max_total"] > 0, tmp["min_total"] / tmp["max_total"], 0
    )
    tmp["error_rate"] = 1 - tmp["accuracy"]
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    results["architecture"] = arch
    return results


def get_metrics_test_fast(
    num_classes: 2,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: str or int,
    dataset: str,
    arch: str,
    most_common_class: int = 0,
):
    """
    Function to get results over different coverages (so we store everything)
        num_classes: int
            The number of classes in the test set
        coverages: list
            A list of desired coverages for evaluating the selective classifiers
        true: np.array
            The array with true labels
        selected: np.array
            The array provided by qband function, with instances labelled according to increasing levels of coverage
        y_scores: np.array
            The array for the scores
        y_preds: np.array
            The array for predictions
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        trial_num: int
            The number of trial
        dataset: str
            The dataset name
        arch: str
            The base network architecture
        most_common_class: str
            The label of most common class in the training set

    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    for i, cov in enumerate(cover):
        if "selnet" in meta:
            if type(preds) == list:
                preds = y_preds[i]
            else:
                preds = y_preds[cov]
        else:
            preds = y_preds
        if preds[selected > i].shape[0] == 0:
            coverage = -1
            viol = 1
            acc = -1
            acc_stupid1 = 1
            if num_classes == 2:
                pos_rate = -1
        else:
            coverage = len(selected[selected > i]) / len(selected)
            viol = abs((len(selected[selected > i]) / len(selected) - cov))
            acc = skm.accuracy_score(true[selected > i], preds[selected > i])
            acc_stupid1 = skm.accuracy_score(
                true[selected > i],
                np.repeat(most_common_class, len(true))[selected > i],
            )
            if num_classes == 2:
                pos_rate = np.sum(true[selected > i]) / len(true[selected > i])
        tmp = pd.DataFrame()
        tmp["accuracy"] = [acc]
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["accuracy"] = acc
        tmp["desired_coverage"] = cov
        tmp["coverage"] = coverage
        tmp["reject_rate"] = 1 - coverage
        tmp["viol"] = viol
        tmp["acc_stupid1"] = acc_stupid1
        # tmp["relative_accuracy1"] = 1 - (acc / acc_stupid1)
        # tmp["relative_error1"] = 1 - (1 - acc) / (1 - acc_stupid1)
        if num_classes == 2:
            tmp["pos_rate"] = pos_rate
            tmp["true_pos"] = np.sum(true)
            tmp["num_positives"] = np.sum(true[selected > i])
            tmp["num_acc"] = len(true[selected > i])
            tmp["false_positives_acc"] = np.sum(
                true[(preds != true) & (true == 1) & (selected > i)]
            )
            tmp["false_negatives_acc"] = np.sum(
                true[(preds != true) & (true == 0) & (selected > i)]
            )
            tmp["false_positives_rej"] = np.sum(
                true[(preds != true) & (true == 1) & (selected <= i)]
            )
            tmp["false_negatives_rej"] = np.sum(
                true[(preds != true) & (true == 0) & (selected <= i)]
            )
        results = pd.concat([results, tmp], axis=0)

    results["error_rate"] = 1 - results["accuracy"]
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    results["architecture"] = arch
    return results


def get_metrics_test_selnet(
    num_classes: int,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_scores: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: int,
    dataset: str,
    arch: str,
    true_cov: float,
    most_common_class: int = 0,
):
    """
    Function to get results over different coverages (so we store everything)
         num_classes: int
             The number of classes in the test set
         coverages: list
             A list of desired coverages for evaluating the selective classifiers
         true: np.array
             The array with true labels
         selected: np.array
             The array provided by qband function, with instances labelled according to increasing levels of coverage
         y_scores: np.array
             The array for the scores
         y_preds: np.array
             The array for predictions
         meta: str
             The type of selective classifier. Possible values are:
                 - 'plugin' for a simple network
                 - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                 - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                 - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                 - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                 - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
         trial_num: int
             The number of trial
         dataset: str
             The dataset name
         arch: str
             The base network architecture
         true_cov: float
             The coverage used to train the original SelNet
         most_common_class: str
             The label of most common class in the training set
    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    scores = y_scores
    preds = y_preds
    if scores[selected > 0].shape[0] == 0:
        coverage = -1
        viol = 1
        loss = 1
        acc = -1
        print(acc)
        tmp = pd.DataFrame()
        if num_classes == 2:
            auc = -1
    else:
        coverage = len(selected[selected > 0]) / len(selected)
        viol = abs((len(selected[selected > 0]) / len(selected) - true_cov))
        acc = skm.accuracy_score(true[selected > 0], preds[selected > 0])
        acc_stupid1 = skm.accuracy_score(
            true[selected > 0], np.repeat(most_common_class, len(true))[selected > 0]
        )

        if num_classes == 2:
            try:
                auc = skm.roc_auc_score(true[selected > 0], scores[:, 1][selected > 0])
            except ValueError:
                auc = -1
    tmp = pd.DataFrame()
    tmp["accuracy"] = [acc]
    tmp["desired_coverage"] = true_cov
    tmp["coverage"] = coverage
    tmp["reject_rate"] = 1 - coverage
    tmp["viol"] = viol
    tmp["relative_accuracy1"] = 1 - (acc / acc_stupid1)
    tmp["relative_error1"] = 1 - (1 - acc) / (1 - acc_stupid1)
    tmp["acc_stupid1"] = acc_stupid1
    if num_classes == 2:
        tmp["auc"] = auc
        tmp["true_pos"] = np.sum(true)
        tmp["num_positives"] = np.sum(true[selected > 0])
        tmp["num_acc"] = len(true[selected > 0])
        tmp["false_positives_acc"] = np.sum(
            true[(preds != true) & (true == 1) & (selected > 0)]
        )
        tmp["false_negatives_acc"] = np.sum(
            true[(preds != true) & (true == 0) & (selected > 0)]
        )
        tmp["false_positives_rej"] = np.sum(
            true[(preds != true) & (true == 1) & (selected <= 0)]
        )
        tmp["false_negatives_rej"] = np.sum(
            true[(preds != true) & (true == 0) & (selected <= 0)]
        )
    elif num_classes > 2:
        tmp["num_acc"] = len(true[selected > 0])
    for label in np.unique(true):
        try:
            tmp["accepted_class{}".format(int(label))] = len(
                true[(selected > 0) & (true == label)]
            )
        except:
            tmp["accepted_class{}".format(int(label))] = 0
        try:
            tmp["total_class{}".format(int(label))] = len(true[true == label])
        except:
            tmp["total_class{}"] = -1
    tmp["max_accepted"] = tmp[
        [cols for cols in tmp.columns if "accepted_class" in cols]
    ].max(axis=1)
    tmp["min_accepted"] = tmp[
        [cols for cols in tmp.columns if "accepted_class" in cols]
    ].min(axis=1)
    tmp["min_max_ratio_accepted"] = np.where(
        tmp["max_accepted"] > 0, tmp["min_accepted"] / tmp["max_accepted"], 0
    )
    tmp["max_total"] = tmp[[cols for cols in tmp.columns if "total_class" in cols]].max(
        axis=1
    )
    tmp["min_total"] = tmp[[cols for cols in tmp.columns if "total_class" in cols]].min(
        axis=1
    )
    tmp["min_max_ratio_total"] = np.where(
        tmp["max_total"] > 0, tmp["min_total"] / tmp["max_total"], 0
    )
    tmp["error_rate"] = 1 - tmp["accuracy"]
    # print(tmp)
    results = pd.concat([results, tmp], axis=0)
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    results["architecture"] = arch
    return results


def get_metrics_test_selnet_fast(
    num_classes: int,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: int,
    dataset: str,
    arch: str,
    true_cov: float,
    most_common_class: int = 0,
):
    """
    Function to get results over different coverages (so we store everything)
         num_classes: int
             The number of classes in the test set
         coverages: list
             A list of desired coverages for evaluating the selective classifiers
         true: np.array
             The array with true labels
         selected: np.array
             The array provided by qband function, with instances labelled according to increasing levels of coverage
         y_scores: np.array
             The array for the scores
         y_preds: np.array
             The array for predictions
         meta: str
             The type of selective classifier. Possible values are:
                 - 'plugin' for a simple network
                 - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                 - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                 - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                 - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                 - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
         trial_num: int
             The number of trial
         dataset: str
             The dataset name
         arch: str
             The base network architecture
         true_cov: float
             The coverage used to train the original SelNet
         most_common_class: str
             The label of most common class in the training set
    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    preds = y_preds
    if preds[selected > 0].shape[0] == 0:
        coverage = -1
        viol = 1
        loss = 1
        acc = -1
        print(acc)
        tmp = pd.DataFrame()
        if num_classes == 2:
            auc = -1
    else:
        coverage = len(selected[selected > 0]) / len(selected)
        viol = abs((len(selected[selected > 0]) / len(selected) - true_cov))
        acc = skm.accuracy_score(true[selected > 0], preds[selected > 0])
        acc_stupid1 = skm.accuracy_score(
            true[selected > 0], np.repeat(most_common_class, len(true))[selected > 0]
        )
    tmp = pd.DataFrame()
    tmp["accuracy"] = [acc]
    tmp["desired_coverage"] = true_cov
    tmp["coverage"] = coverage
    tmp["reject_rate"] = 1 - coverage
    tmp["viol"] = viol
    tmp["relative_accuracy1"] = 1 - (acc / acc_stupid1)
    tmp["relative_error1"] = 1 - (1 - acc) / (1 - acc_stupid1)
    tmp["acc_stupid1"] = acc_stupid1
    if num_classes == 2:
        tmp["true_pos"] = np.sum(true)
        tmp["num_positives"] = np.sum(true[selected > 0])
        tmp["num_acc"] = len(true[selected > 0])
        tmp["false_positives_acc"] = np.sum(
            true[(preds != true) & (true == 1) & (selected > 0)]
        )
        tmp["false_negatives_acc"] = np.sum(
            true[(preds != true) & (true == 0) & (selected > 0)]
        )
        tmp["false_positives_rej"] = np.sum(
            true[(preds != true) & (true == 1) & (selected <= 0)]
        )
        tmp["false_negatives_rej"] = np.sum(
            true[(preds != true) & (true == 0) & (selected <= 0)]
        )

    tmp["error_rate"] = 1 - tmp["accuracy"]
    # print(tmp)
    results = pd.concat([results, tmp], axis=0)
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    results["architecture"] = arch
    return results


def get_metrics_test_ood(
    num_classes: 2,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_scores: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: str or int,
    dataset: str,
    arch: str,
    perc_train: float,
):
    """
    Function to get results over different coverages for the OOD experiment
       num_classes: int
           The number of classes in the test set
       coverages: list
           A list of desired coverages for evaluating the selective classifiers
       true: np.array
           The array with true labels
       selected: np.array
           The array provided by qband function, with instances labelled according to increasing levels of coverage
       y_scores: np.array
           The array for the scores
       y_preds: np.array
           The array for predictions
       meta: str
           The type of selective classifier. Possible values are:
               - 'plugin' for a simple network
               - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
               - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
               - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
               - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
               - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
       trial_num: int
           The number of trial
       dataset: str
           The dataset name
       arch: str
           The base network architecture
       most_common_class: str
           The label of most common class in the training set
    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    for i, cov in enumerate(cover):
        if "selnet" in meta:
            if type(y_scores) == list:
                scores = y_scores[i]
                preds = y_preds[i]
            else:
                scores = y_scores[cov]
                preds = y_preds[cov]
        else:
            scores = y_scores
            preds = y_preds
        try:
            coverage = len(scores[selected > i]) / len(scores)
        except:
            coverage = 0
        tmp = pd.DataFrame()
        tmp["coverage"] = [coverage]
        tmp["desired_coverage"] = cov
        results = pd.concat([results, tmp], axis=0)
    results["meta"] = meta
    results["dataset"] = dataset
    results["trial"] = trial_num
    results["architecture"] = arch
    return results


def get_metrics_test_ood_selnet(
    num_classes: int,
    coverages: list,
    true: np.array,
    selected: np.array,
    y_scores: np.array,
    y_preds: np.array,
    meta: str,
    trial_num: int,
    dataset: str,
    arch: str,
    perc_train: float,
    true_cov: float,
):
    """
    Function to get results over different coverages for the OOD experiment for SelNet
       num_classes: int
           The number of classes in the test set
       coverages: list
           A list of desired coverages for evaluating the selective classifiers
       true: np.array
           The array with true labels
       selected: np.array
           The array provided by qband function, with instances labelled according to increasing levels of coverage
       y_scores: np.array
           The array for the scores
       y_preds: np.array
           The array for predictions
       meta: str
           The type of selective classifier. Possible values are:
               - 'plugin' for a simple network
               - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
               - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
               - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
               - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
               - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
       trial_num: int
           The number of trial
       dataset: str
           The dataset name
       arch: str
           The base network architecture
       most_common_class: str
           The label of most common class in the training set
    """
    results = pd.DataFrame()
    cover = sorted(coverages, reverse=True)
    for i, cov in enumerate(cover):
        if cov != true_cov:
            continue
        else:
            scores = y_scores
            preds = y_preds
            try:
                coverage = len(scores[selected > i]) / len(scores)
            except:
                coverage = 0
            tmp = pd.DataFrame()
            tmp["desired_coverage"] = [cov]
            tmp["coverage"] = coverage
            results = pd.concat([results, tmp], axis=0)
        results["meta"] = meta
        results["dataset"] = dataset
        results["trial"] = trial_num
        results["architecture"] = arch
    return results


def objective(
    trial,
    meta,
    arch,
    iterations,
    device,
    dataset,
    training_set,
    holdout_set,
    calibration_set,
    root_models,
    root_results,
    sub,
    bsize,
    verb=False,
):
    """

    Args:
        trial: Optuna.trial
            The Optuna trial for optimizing results
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        arch: str
            The architecture for the neural network.
        iterations: int
            The number of epochs to train the model
        device: str
            The device used by pytorch
        dataset: str
            The name of the dataset
        training_set: TabularDataset or ImgFolder
            The training set
        holdout_set: TabularDataset or ImgFolder
            The holdout set
        calibration_set: TabularDataset or ImgFolder
            The calibration set
        root_models: str
            The folder to store models
        root_results: str
            The folder to store results
        sub: bool
            If True, we subsample the dataset. In all the experiments this was set to False.
        bsize: int
            The batch size
        verb: bool
            If True, the training prints evolution

    Returns:

    """
    seed = 42
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    td_ = trial.suggest_categorical("time_decay", [False, True])
    exp_wd = trial.suggest_int("exp_wd", -6, -3, step=1)
    wd_ = 10**exp_wd
    torch.manual_seed(seed)
    set_seed(42)
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    if type(training_set) == ImgFolder:
        tabular = False
    else:
        tabular = True
    if sub:
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed)
        g3 = torch.Generator().manual_seed(seed)
        s1 = torch.utils.data.RandomSampler(
            training_set, num_samples=1000, generator=g1
        )
        s2 = torch.utils.data.RandomSampler(holdout_set, num_samples=300, generator=g2)
        s3 = torch.utils.data.RandomSampler(
            calibration_set, num_samples=300, generator=g3
        )
    else:
        s1 = None
        s2 = None
        s3 = None
    if meta in ["sele", "reg"]:
        train_idx, unc_idx, y_train, y_unc = train_test_split(
            np.arange(len(training_set.targets)),
            training_set.targets,
            stratify=training_set.targets,
            test_size=0.5,
            random_state=42,
        )
        training_set_cl = torch.utils.data.Subset(training_set, train_idx)
        training_set_unc = torch.utils.data.Subset(training_set, unc_idx)
        if tabular:
            if sub:
                trainloader = torch.utils.data.DataLoader(
                    training_set_cl, batch_size=bsize, sampler=s1
                )
                trainloader_unc = torch.utils.data.DataLoader(
                    training_set_unc, batch_size=bsize, sampler=s1
                )

            else:
                trainloader = torch.utils.data.DataLoader(
                    training_set_cl,
                    batch_size=bsize,
                    shuffle=True,
                )
                trainloader_unc = torch.utils.data.DataLoader(
                    training_set_unc,
                    batch_size=bsize,
                    shuffle=True,
                )

            holdloader = torch.utils.data.DataLoader(
                holdout_set,
                batch_size=int(bsize / 4),
                shuffle=False,
            )
            calloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=int(bsize / 4),
                shuffle=False,
            )
        else:
            if sub:
                trainloader = torch.utils.data.DataLoader(
                    training_set_cl,
                    batch_size=bsize,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                    sampler=s1,
                )
                trainloader_unc = torch.utils.data.DataLoader(
                    training_set_unc,
                    batch_size=bsize,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                    sampler=s1,
                )
            else:
                trainloader = torch.utils.data.DataLoader(
                    training_set_cl,
                    batch_size=bsize,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                )
                trainloader_unc = torch.utils.data.DataLoader(
                    training_set_unc,
                    batch_size=bsize,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                )

            holdloader = torch.utils.data.DataLoader(
                holdout_set, batch_size=int(bsize / 4), shuffle=False, pin_memory=True
            )
            calloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=int(bsize / 4),
                shuffle=False,
                pin_memory=True,
            )
    else:
        if tabular:
            if sub:
                trainloader = torch.utils.data.DataLoader(
                    training_set, batch_size=bsize, sampler=s1
                )
            else:
                trainloader = torch.utils.data.DataLoader(
                    training_set,
                    batch_size=bsize,
                    shuffle=True,
                )
            holdloader = torch.utils.data.DataLoader(
                holdout_set,
                batch_size=int(bsize / 4),
                shuffle=False,
            )
            calloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=int(bsize / 4),
                shuffle=False,
            )
        else:
            if sub:
                trainloader = torch.utils.data.DataLoader(
                    training_set,
                    batch_size=bsize,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                    sampler=s1,
                )
            else:
                trainloader = torch.utils.data.DataLoader(
                    training_set,
                    batch_size=bsize,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                )
            holdloader = torch.utils.data.DataLoader(
                holdout_set, batch_size=int(bsize / 4), shuffle=False, pin_memory=True
            )
            calloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=int(bsize / 4),
                shuffle=False,
                pin_memory=True,
            )
    # set fixed parameters
    coverages = [0.99, 0.85, 0.7]
    n_classes = len(training_set.classes)
    # Training of the model.
    if meta == "plugin":
        crit = "ce"
    elif meta == "dg":
        crit = "dg"
        reward_ = trial.suggest_float("reward", 1, n_classes, step=0.2)
    elif meta == "sat":
        crit = "sat"
    elif meta == "selnet":
        crit = "selnet"
    elif meta == "sat_te":
        crit = "sat_te"
    elif meta == "selnet_te":
        crit = "selnet_te"
    if tabular == False:
        if dataset in ["cifar10", "SVHN", "cifar100"]:
            input_size = 32
        elif dataset in [
            "MNIST",
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
        elif dataset in ["catsdogs"]:
            input_size = 64
        elif dataset in ["food101", "waterbirds", "celeba", "stanfordcars"]:
            input_size = 224
        else:
            input_size = 128
            if arch != "Resnet34":
                raise ValueError(
                    "Use resnet architecture for {} dataset".format(dataset)
                )
    if "_te" in meta:
        exp_beta = trial.suggest_int("exp_beta", -4, -1, step=1)
        beta_ = 10**exp_beta
    if "selnet" in meta:
        exp_lamda = trial.suggest_int("exp_lamda", 3, 6, step=1)
        lamda_ = 2**exp_lamda
        alpha_ = trial.suggest_float("alpha", 0.25, 0.75, step=0.05)
    if "sat" in meta:
        pretrain_ = trial.suggest_int("pretrain", 0, 60, step=15)
        momentum_ = trial.suggest_float("momentum", 0.9, 0.99, step=0.01)
    if meta == "cn":
        deactiv = True
        optimizer_name_unc = trial.suggest_categorical(
            "optimizer_name_unc", ["AdamW", "Adam", "SGD"]
        )
        if optimizer_name_unc == "SGD":
            momentum_sgd_unc = trial.suggest_float(
                "momentum_sgd_unc", 0.9, 0.99, step=0.01
            )
            exp_lr_unc = trial.suggest_int("exp_lr_unc", -8, -4, step=1)
            lr_unc = 10**exp_lr_unc
            nesterov_unc = trial.suggest_categorical("nesterov_unc", [True, False])
        elif "Adam" in optimizer_name_unc:
            exp_lr_unc = trial.suggest_int("exp_lr_unc", -8, -4, step=1)
            lr_unc = 10**exp_lr_unc
            nesterov_unc = False
    elif meta in ["sele", "reg"]:
        deactiv = False
        optimizer_name_unc = trial.suggest_categorical(
            "optimizer_name_unc", ["AdamW", "Adam", "SGD"]
        )
        if optimizer_name_unc == "SGD":
            momentum_sgd_unc = trial.suggest_float(
                "momentum_sgd_unc", 0.9, 0.99, step=0.01
            )
            exp_lr_unc = trial.suggest_int("exp_lr_unc", -6, -3, step=1)
            lr_unc = 10**exp_lr_unc
            nesterov_unc = trial.suggest_categorical("nesterov_unc", [True, False])
        elif "Adam" in optimizer_name_unc:
            exp_lr_unc = trial.suggest_int("exp_lr_unc", -7, -3, step=1)
            lr_unc = 10**exp_lr_unc
            nesterov_unc = False
    true = holdout_set.targets
    if type(true) == list:
        true = np.array(true)
    if optimizer_name == "SGD":
        momentum_sgd_ = trial.suggest_float("momentum_sgd", 0.9, 0.99, step=0.01)
        exp_lr = trial.suggest_int("exp_lr", -4, -1, step=1)
        lr_ = 10**exp_lr
        nesterov = trial.suggest_categorical("nesterov", [True, False])
    elif "Adam" in optimizer_name:
        exp_lr = trial.suggest_int("exp_lr", -5, -2, step=1)
        lr_ = 10**exp_lr
        nesterov = False
    torch.cuda.empty_cache()
    failed_trial = False
    if "selnet" in meta:
        # Generate the model.
        if tabular:
            model = [
                build_model_tabular(trial, arch, train_=training_set, meta=meta)
                for _ in range(len(coverages))
            ]
        else:
            model = [
                build_model_image(
                    trial, arch, train_=training_set, meta=meta, inp=input_size
                )
                for _ in range(len(coverages))
            ]
        print(trial.params)
        scores = []
        preds = []
        st_time = time()
        times = []
        print(trial.number)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f)
        for i, cov in enumerate(coverages):
            # here we train model for every coverage
            if nesterov:
                optimizer = getattr(optim, optimizer_name)(
                    model[i].parameters(),
                    lr=lr_,
                    weight_decay=wd_,
                    nesterov=nesterov,
                    momentum=momentum_sgd_,
                )
            else:
                optimizer = getattr(optim, optimizer_name)(
                    model[i].parameters(), lr=lr_, weight_decay=wd_
                )
            path_model = "{}/{}/{}_{}_{}_tr{}_ep{}.pt".format(
                root_models, dataset, arch, meta, cov, trial.number, iterations
            )
            path_interm = "{}/{}/{}_{}_{}_tr{}".format(
                root_models, dataset, arch, meta, cov, trial.number
            )
            start_time = time()
            if os.path.exists(path_model):
                model[i].to(device)
                model[i].load_state_dict(torch.load(path_model, map_location=device))
                failed_trial = True
            else:
                if "_em" in meta:
                    model[i] = train(
                        model[i],
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        beta=beta_,
                        coverage=cov,
                        lamda=lamda_,
                        alpha=alpha_,
                        path_interm=path_interm,
                    )
                elif "_te" in meta:
                    print("Using CRITERION: {}".format(meta))
                    model[i] = train(
                        model[i],
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        beta=beta_,
                        coverage=cov,
                        lamda=lamda_,
                        alpha=alpha_,
                        path_interm=path_interm,
                    )
                else:
                    model[i] = train(
                        model[i],
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        coverage=cov,
                        lamda=lamda_,
                        alpha=alpha_,
                        path_interm=path_interm,
                    )
                torch.save(model[i].state_dict(), path_model)
            end_time = time()
            times.append(end_time - start_time)
            if i == 2:
                en_time = time()
                fi_time = st_time - en_time
            # here we append scores for each model
            y_hat = predict_proba(model[i], device, holdloader, meta)
            scores.append(y_hat)
            preds.append(np.argmax(y_hat, axis=1))
            # model[i].to("cpu")
            if os.path.exists("{}/{}".format(root_models, dataset)) == False:
                os.mkdir("{}/{}".format(root_models, dataset))
            path_model = "{}/{}/{}_{}_{}_tr{}_ep{}.pt".format(
                root_models, dataset, arch, meta, cov, trial.number, iterations
            )

        torch.cuda.empty_cache()
        # here we determine the "bands" for selection
    else:
        if tabular:
            model = build_model_tabular(trial, arch, train_=training_set, meta=meta)
        else:
            model = build_model_image(
                trial, arch, train_=training_set, meta=meta, inp=input_size
            )
        if meta in ["cn", "sele", "reg"]:
            if arch == "VGG":
                if len([el for el in model.named_modules()]) > 50:
                    model_unc = buildConfidNet("VGG16bn", model, {}, deactivate=deactiv)
                else:
                    model_unc = buildConfidNet("VGG16", model, {}, deactivate=deactiv)
            else:
                model_unc = buildConfidNet(arch, model, {}, deactivate=deactiv)
        print(trial.params)
        if nesterov:
            optimizer = getattr(optim, optimizer_name)(
                model.parameters(),
                lr=lr_,
                weight_decay=wd_,
                nesterov=nesterov,
                momentum=momentum_sgd_,
            )
        else:
            optimizer = getattr(optim, optimizer_name)(
                model.parameters(), lr=lr_, weight_decay=wd_
            )
        start_time = time()
        if os.path.exists("{}/{}".format(root_models, dataset)) == False:
            os.mkdir("{}/{}".format(root_models, dataset))
        path_model = "{}/{}/{}_{}_{}_tr{}_ep{}.pt".format(
            root_models, dataset, arch, meta, "xx", int(trial.number), iterations
        )
        path_interm = "{}/{}/{}_{}_{}_tr{}".format(
            root_models, dataset, arch, meta, "xx", int(trial.number)
        )
        if os.path.exists(path_model):
            model.load_state_dict(torch.load(path_model, map_location=device))
            model.to(device)
            failed_trial = True
            if meta in ["cn", "sele", "reg"]:
                path_model_unc = "{}/{}/unc_{}_{}_{}_tr{}_ep{}.pt".format(
                    root_models,
                    dataset,
                    arch,
                    meta,
                    "xx",
                    int(trial.number),
                    iterations,
                )
                if os.path.exists(path_model_unc):
                    model_unc.load_state_dict(
                        torch.load(path_model_unc, map_location=device)
                    )
                    failed_trial = True
                else:
                    print("training the uncertainty model")
                    if meta == "cn":
                        if arch == "VGG":
                            if len([el for el in model.named_modules()]) > 50:
                                model_unc = buildConfidNet(
                                    "VGG16bn", model, {}, deactivate=deactiv
                                )
                            else:
                                model_unc = buildConfidNet(
                                    "VGG16", model, {}, deactivate=deactiv
                                )
                        else:
                            model_unc = buildConfidNet(
                                arch, model, {}, deactivate=deactiv
                            )
                    model_unc.to(device)
                    if nesterov_unc:
                        optimizer_unc = getattr(optim, optimizer_name_unc)(
                            model_unc.parameters(),
                            lr=lr_unc,
                            weight_decay=wd_,
                            nesterov=nesterov_unc,
                            momentum=momentum_sgd_unc,
                        )
                    else:
                        optimizer_unc = getattr(optim, optimizer_name_unc)(
                            model_unc.parameters(), lr=lr_unc, weight_decay=wd_
                        )
                    if meta == "cn":
                        model_unc = train(
                            model_unc,
                            device,
                            iterations,
                            optimizer_unc,
                            "cn",
                            trainloader,
                            td=td_,
                            verbose=verb,
                            path_interm=path_interm,
                            model_base_confnet=model,
                        )
                    else:
                        model_unc = train(
                            model_unc,
                            device,
                            iterations,
                            optimizer_unc,
                            meta,
                            trainloader_unc,
                            td=td_,
                            verbose=verb,
                            path_interm=path_interm,
                            model_base_confnet=model,
                        )

                    torch.save(model_unc.state_dict(), path_model_unc)
        else:
            if meta in ["plugin", "cn", "sele", "reg"]:
                model = train(
                    model,
                    device,
                    iterations,
                    optimizer,
                    "ce",
                    trainloader,
                    td=td_,
                    verbose=verb,
                    path_interm=path_interm,
                )
                model.to(device)
                if meta in ["cn", "sele", "reg"]:
                    print("training the uncertainty model")
                    if meta == "cn":
                        if arch == "VGG":
                            if len([el for el in model.named_modules()]) > 50:
                                model_unc = buildConfidNet(
                                    "VGG16bn", model, {}, deactivate=deactiv
                                )
                            else:
                                model_unc = buildConfidNet(
                                    "VGG16", model, {}, deactivate=deactiv
                                )
                        else:
                            model_unc = buildConfidNet(
                                arch, model, {}, deactivate=deactiv
                            )
                    if nesterov_unc:
                        optimizer_unc = getattr(optim, optimizer_name_unc)(
                            model_unc.parameters(),
                            lr=lr_unc,
                            weight_decay=wd_,
                            nesterov=nesterov_unc,
                            momentum=momentum_sgd_unc,
                        )
                    else:
                        optimizer_unc = getattr(optim, optimizer_name_unc)(
                            model_unc.parameters(), lr=lr_unc, weight_decay=wd_
                        )
                    model_unc.to(device)
                    if meta == "cn":
                        model_unc = train(
                            model_unc,
                            device,
                            iterations,
                            optimizer_unc,
                            meta,
                            trainloader,
                            td=td_,
                            verbose=verb,
                            path_interm=path_interm,
                            model_base_confnet=model,
                        )
                        path_model_unc = "{}/{}/unc_{}_{}_{}_tr{}_ep{}.pt".format(
                            root_models,
                            dataset,
                            arch,
                            meta,
                            "xx",
                            int(trial.number),
                            iterations,
                        )
                    else:
                        model_unc = train(
                            model_unc,
                            device,
                            iterations,
                            optimizer_unc,
                            meta,
                            trainloader_unc,
                            td=td_,
                            verbose=verb,
                            path_interm=path_interm,
                            model_base_confnet=model,
                        )
                        path_model_unc = "{}/{}/unc_{}_{}_{}_tr{}_ep{}.pt".format(
                            root_models,
                            dataset,
                            arch,
                            meta,
                            "xx",
                            int(trial.number),
                            iterations,
                        )
                    torch.save(model_unc.state_dict(), path_model_unc)
            elif "sat" in meta:
                if "_em" in meta:
                    model = train(
                        model,
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        beta=beta_,
                        pretrain=pretrain_,
                        momentum=momentum_,
                        path_interm=path_interm,
                    )
                elif "_te" in meta:
                    model = train(
                        model,
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        beta=beta_,
                        pretrain=pretrain_,
                        momentum=momentum_,
                        path_interm=path_interm,
                    )
                else:
                    model = train(
                        model,
                        device,
                        iterations,
                        optimizer,
                        crit,
                        trainloader,
                        td=td_,
                        verbose=verb,
                        pretrain=pretrain_,
                        momentum=momentum_,
                        path_interm=path_interm,
                    )
            elif meta == "dg":
                model = train(
                    model,
                    device,
                    iterations,
                    optimizer,
                    crit,
                    trainloader,
                    td=td_,
                    verbose=verb,
                    reward=reward_,
                    path_interm=path_interm,
                )
            torch.save(model.state_dict(), path_model)
        end_time = time()
        fi_time = end_time - start_time
        times = [fi_time for _ in coverages]
        # model.to("cpu")
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = (r - a) / (1024)
        print(f)
        torch.cuda.empty_cache()
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = (r - a) / (1024)
        print(f)
        scores = predict_proba(model, device, holdloader, meta)
        preds = np.argmax(scores, axis=1)
    if meta in ["cn", "sele", "reg"]:
        selected = qband(model_unc, device, holdloader, calloader, meta, coverages)
    else:
        selected = qband(model, device, holdloader, calloader, meta, coverages)
    dict_params = trial.params
    n_trial = trial.number
    if len(training_set.classes) == 2:
        bin = True
    else:
        bin = False
    print("\n\n\n\n\n\n\n\n HERE WE GO WITH THE RESULTS \n\n\n\n\n\n")
    p_train = np.sum(training_set.targets) / len(training_set)
    results, violations, losses = get_metrics(
        params=dict_params,
        num_classes=n_classes,
        coverages=coverages,
        true=true,
        selected=selected,
        y_scores=scores,
        y_preds=preds,
        meta=meta,
        trial_num=n_trial,
        dataset=dataset,
        perc_train=p_train,
    )
    if os.path.exists("{}/{}".format(root_results, dataset)) == False:
        os.mkdir("{}/{}".format(root_results, dataset))
    results["time_to_fit"] = times
    results_filename = "{}/{}/results_{}_{}_{}_{}.csv".format(
        root_results, dataset, meta, arch, n_trial, dataset
    )
    results.to_csv(results_filename, index=False)
    if losses[0] is None:
        losses[0] = 1
    if losses[1] is None:
        losses[1] = 1
    if losses[2] is None:
        losses[2] = 1
    if violations[0] is None:
        violations[0] = 1
    if violations[1] is None:
        violations[1] = 1
    if violations[2] is None:
        violations[2] = 1
    return (
        violations[0],
        violations[1],
        violations[2],
        losses[0],
        losses[1],
        losses[2],
    )


def tabular_model(model_type, x_num, cat_dim, num_classes, meta, body_dict):
    """

    Args:
        model_type: str
            The network type. Possible values are:
                -'TabResnet' for tabular resnet
                -'TabFTTransofmrer' for tabular transformer
        x_num: int
            The number of continuous features
        cat_dim: list
            A list containing for every categorical feature the maximum values of its code.
            For instance, a dataset with two binary categorical features will require
            x_cat = [1,1]
        num_classes: int
            The number of classes
        meta:
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        body_dict:
            The features of the architectures

    Returns:
        model: torch.Module
            A neural network for tabular datasets
    """
    n_classes = num_classes
    if meta in ["sat", "dg", "sat_em", "sat_te"]:
        n_classes += 1
    body = copy.deepcopy(body_dict)
    body["d_in"] = x_num
    body["cat_cardinalities"] = cat_dim
    body["d_out"] = n_classes
    if model_type == "TabFTTransformer":
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {
                "main_body": "transformer",
                "d_out": n_classes,
                "pre_norm": True,
            }
            model = buildSelNet(
                model_type="TabFTTransformer", body_params=body, head_params=head
            )
        else:
            model = TabFTTransformer(**body)
    elif model_type == "TabResnet":
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes, "pre_norm": True}
            model = buildSelNet(
                model_type="TabResnet", body_params=body, head_params=head
            )
        else:
            model = TabCatResNet(**body)
    else:
        raise NotImplementedError("The model architecture is not implemented yet.")
    return model


def image_model(model_type, num_classes, meta, body_dict):
    """

    Args:
        model_type: str
            The network type. Possible values are:
                -'VGG' for VGG
                -'Resnet34' for Resnet34
                -'Resnet50' for Resnet50
        num_classes: int
            The number of classes
        meta:
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        body_dict:
            The features of the architectures

    Returns:
        model: torch.Module
            A neural network for image datasets
    """
    body = copy.deepcopy(body_dict)
    n_classes = num_classes
    if meta in ["sat", "dg", "sat_em", "sat_te"]:
        n_classes += 1
    if model_type == "VGG":
        #         body = {"input_size": inp, "d_out": n_classes}
        body["d_out"] = n_classes
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "VGG", "d_out": n_classes, "pre_norm": False}
            if body["b_norm"]:
                body.pop("b_norm")
                model = buildSelNet(
                    model_type="VGG16bn", body_params=body, head_params=head
                )
            else:
                body.pop("b_norm")
                model = buildSelNet(
                    model_type="VGG16", body_params=body, head_params=head
                )

        else:
            if body["b_norm"]:
                body.pop("b_norm")
                model = vgg16_bn(**body)
            else:
                body.pop("b_norm")
                model = vgg16(**body)
    elif model_type == "Resnet34":
        body["block"] = BasicBlock
        body["layers"] = [3, 4, 6, 3]
        body["num_classes"] = n_classes
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet34", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    elif model_type == "Resnet18":
        body["block"] = BasicBlock
        body["layers"] = [2, 2, 2, 2]
        body["num_classes"] = n_classes
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet18", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    elif model_type == "Resnet50":
        body["block"] = Bottleneck
        body["layers"] = [3, 4, 6, 3]
        body["num_classes"] = n_classes
        if meta in ["selnet", "selnet_em", "selnet_te"]:
            head = {"main_body": "resnet", "d_out": n_classes}
            model = buildSelNet(
                model_type="Resnet50", body_params=body, head_params=head
            )
        else:
            model = torchvision.models.ResNet(**body)
    return model


def get_best_trial(meta, filename, architecture, result_fold="results", also_num=False):
    """

    Args:
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        filename: str
            The dataset under analysis
        architecture: str
            The model type employed
        result_fold: str
            The folder path were results of tuning are stored
        also_num: bool
            If True, it reports also the number of trials and the best one. Default is False.

    Returns:
            bm: pandas.DataFrame
            A pandas Dataframe with best parameters for specific dataset and meta.

    """
    db = pd.DataFrame()
    res_folder = "{}/{}".format(result_fold, filename)
    results_tuning = [el for el in os.listdir(res_folder) if "TESTING" not in el]
    for f in results_tuning:
        tmp = pd.read_csv(res_folder + "/" + f)
        if "VGG" in f:
            tmp["model_type"] = "VGG"
        elif "Resnet34" in f:
            tmp["model_type"] = "Resnet34"
        elif "Resnet50" in f:
            tmp["model_type"] = "Resnet50"
        elif "Resnet18" in f:
            tmp["model_type"] = "Resnet18"
        elif "TabResnet" in f:
            tmp["model_type"] = "TabResnet"
        elif "TabFTTransformer" in f:
            tmp["model_type"] = "TabFTTransformer"
        db = pd.concat([db, tmp], axis=0)
    db = db[db["model_type"] == architecture].copy()
    num_trials = db["trial"].max()
    db["viol"] = np.fabs(db["coverage"] - db["desired_coverage"])
    db.reset_index(inplace=True, drop=True)
    db["max_viol_trial"] = db.groupby(["trial", "meta", "dataset", "model_type"])[
        "viol"
    ].transform(lambda x: x.max())
    # 1) check maximum violation is below .05
    print("check res shape after violation")
    res = db[db["max_viol_trial"] <= 0.05].copy()
    print(res.shape)
    # 1.1 if no trial is below .05, then take the one with the average lowest violation
    if (res.shape[0] == 0) or (meta not in res["meta"].unique()):
        db["avg_viol_trial"] = db.groupby(["trial", "meta", "dataset", "model_type"])[
            "viol"
        ].transform(lambda x: x.mean())
        min_tr_meta = db[db["meta"] == meta]["avg_viol_trial"].min()
        res = db[db["avg_viol_trial"] == min_tr_meta].copy()
    # 2) get the best in terms of average performance on the three coverages
    res["avg_perf"] = res.groupby(["trial", "meta", "dataset", "model_type"])[
        "ce_loss"
    ].transform(lambda x: x.mean())
    res["min_avg_perf"] = res.groupby(["trial", "meta", "dataset", "model_type"])[
        "avg_perf"
    ].transform(lambda x: x.min())
    print("check res shape after avg_perf")
    print(res["meta"].unique())
    bm = res[res["meta"] == meta].copy()
    print(bm.shape)
    min_perf = np.min(bm["min_avg_perf"])
    print(min_perf)
    bm = bm[bm["min_avg_perf"] == min_perf].copy()
    print("shape so far of bm")
    print(bm.shape)
    max_trial = bm["trial"].max()
    print(max_trial)
    bm = bm[bm["trial"] == max_trial].copy()
    print(bm.shape)
    if also_num:
        return bm, max_trial, num_trials
    else:
        return bm


def get_best_params(
    meta,
    filename,
    model_type,
    result_fold="results",
    params_fold="best_params",
    already_stored=True,
):
    """

    Args:
        meta: str
            The type of selective classifier. Possible values are:
                - 'plugin' for a simple network
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
        filename: str
            The dataset under analysis
        model_type: str
            The model type employed
        result_fold: str
            The folder path where results of tuning are stored. The default is "results/".
        params_fold: str
            The folder path where best parameters are stored. The default is "best_params/".
        already_stored: bool
            If True, it reads best parameters from file. The default is True.
    Returns:

    """
    params_file = "{}/{}_best_params.csv".format(params_fold, meta)
    if already_stored:
        if os.path.exists(params_file):
            bm = pd.read_csv(params_file)
            print(bm.head())
            bm = bm[bm["dataset"] == filename].copy()
            print("reading from here")
        else:
            bm = get_best_trial(meta, filename, model_type, result_fold)
    else:
        bm = get_best_trial(meta, filename, model_type, result_fold)
    params_opt = ["optimizer", "exp_wd", "exp_lr"]
    try:
        architecture = bm[bm["meta"] == meta]["model_type"].iloc[0]
    except:
        architecture = model_type
        print(architecture)
    if architecture == "TabFTTransformer":
        params_arch = [
            "n_blocks",
            "d_token",
            "attention_dropout",
            "res_drop",
            "ffn_dropout",
            "ffn_factor",
        ]
    elif architecture == "TabResnet":
        params_arch = [
            "n_blocks",
            "d_token",
            "d_main",
            "hidden_factor",
            "dropout_first",
            "dropout_second",
        ]
    elif architecture == "VGG":
        params_arch = ["b_norm"]
    elif architecture in ["Resnet34", "Resnet50", "Resnet18"]:
        params_arch = ["zero_init_residual"]
    if meta in ["selnet", "selnet_sr"]:
        params_loss = ["time_decay", "exp_lamda", "alpha"]
    elif meta in ["selnet_em", "selnet_em_sr", "selnet_te_sr", "selnet_te"]:
        params_loss = ["time_decay", "exp_lamda", "alpha", "exp_beta"]
    elif meta in ["sat", "sat_sr"]:
        params_loss = ["time_decay", "pretrain", "momentum"]
    elif meta in ["sat_em", "sat_em_sr", "sat_te", "sat_te_sr"]:
        params_loss = ["time_decay", "pretrain", "momentum", "exp_beta"]
    elif meta in ["plugin", "cn", "sele", "reg"]:
        params_loss = ["time_decay"]
    elif meta == "dg":
        params_loss = ["time_decay", "reward"]
    print(bm.head())
    dict_params_arch = bm[bm["meta"] == meta][params_arch].iloc[0, :].to_dict()
    dict_params_loss = bm[bm["meta"] == meta][params_loss].iloc[0, :].to_dict()
    dict_params_opt = bm[bm["meta"] == meta][params_opt].iloc[0, :].to_dict()
    dict_params_loss["td"] = copy.deepcopy(dict_params_loss["time_decay"])
    dict_params_loss.pop("time_decay")
    # parameters for TabFTTransformer
    if architecture == "TabFTTransformer":
        dict_params_arch["n_blocks"] = int(dict_params_arch["n_blocks"])
        dict_params_arch["d_token"] = int(dict_params_arch["d_token"])
        dict_params_arch["ffn_d_hidden"] = int(
            dict_params_arch["ffn_factor"] * dict_params_arch["d_token"]
        )
        if dict_params_arch["ffn_d_hidden"] % 2 == 1:
            dict_params_arch["ffn_d_hidden"] += 1
        dict_params_arch.pop("ffn_factor")
        if dict_params_arch["res_drop"] == 1:
            dict_params_arch["residual_dropout"] = bm[bm["meta"] == meta][
                "residual_dropout"
            ].iloc[0]
        print(dict_params_arch["ffn_d_hidden"])
        dict_params_arch.pop("res_drop")
    elif architecture == "TabResnet":
        if (dict_params_arch["hidden_factor"] % 1 > 0.3) & (
            dict_params_arch["hidden_factor"] % 1 < 0.6
        ):
            dict_params_arch["hidden_factor"] = (
                dict_params_arch["hidden_factor"] % 3
            ) / 3 + math.floor(dict_params_arch["hidden_factor"])
        elif (dict_params_arch["hidden_factor"] % 1 > 0.6) & (
            dict_params_arch["hidden_factor"] % 1 < 0.9
        ):
            dict_params_arch["hidden_factor"] = (
                dict_params_arch["hidden_factor"] % 3
            ) / 3 + math.floor(dict_params_arch["hidden_factor"])

        dict_params_arch["d_hidden"] = int(
            dict_params_arch["hidden_factor"] * dict_params_arch["d_main"]
        )
        print(dict_params_arch["d_hidden"])
        dict_params_arch["n_blocks"] = int(dict_params_arch["n_blocks"])
        dict_params_arch["d_token"] = int(dict_params_arch["d_token"])
        dict_params_arch["d_main"] = int(dict_params_arch["d_main"])
        dict_params_arch.pop("hidden_factor")
    if dict_params_opt["optimizer"] == "SGD":
        dict_params_opt["nesterov"] = bm[bm["meta"] == meta]["nesterov"].iloc[0]
        dict_params_opt["momentum_sgd"] = bm[bm["meta"] == meta]["momentum_sgd"].iloc[0]
    dict_params_opt["lr"] = 10 ** dict_params_opt["exp_lr"]
    dict_params_opt["wd"] = 10 ** dict_params_opt["exp_wd"]
    dict_params_opt.pop("exp_lr")
    dict_params_opt.pop("exp_wd")
    if meta in ["sat", "sat_em", "sat_te"]:
        dict_params_loss["pretrain"] = int(dict_params_loss["pretrain"])
    elif meta in ["selnet", "selnet_em", "selnet_te"]:
        dict_params_loss["lamda"] = int(2 ** dict_params_loss["exp_lamda"])
        dict_params_loss.pop("exp_lamda")
    if "exp_beta" in dict_params_loss.keys():
        dict_params_loss["beta"] = 10 ** dict_params_loss["exp_beta"]
        dict_params_loss.pop("exp_beta")
    trial = bm[bm["meta"] == meta]["trial"].iloc[0]
    return dict_params_arch, dict_params_loss, dict_params_opt, trial, architecture


def make_optimizer(d_opt, network):
    """

    Args:
        d_opt: dict
            A dictionary with optimizer features
        network: torch.Module
            The model we aim to optimize

    Returns: torch.optim
            The optimizer

    """
    if "nesterov" in d_opt.keys():
        optimizer = getattr(optim, d_opt["optimizer"])(
            network.parameters(),
            lr=d_opt["lr"],
            weight_decay=d_opt["wd"],
            nesterov=d_opt["nesterov"],
            momentum=d_opt["momentum_sgd"],
        )
    else:
        optimizer = getattr(optim, d_opt["optimizer"])(
            network.parameters(), lr=d_opt["lr"], weight_decay=d_opt["wd"]
        )
    return optimizer


def calibrate_aucross(ys, z, quantiles):
    """

    Args:
    A function from Pugnana and Ruggieri, 2023 to calibrate AUC-based methods.
        ys: np.array
            The true labels
        z:
            The scores of class==1
        quantiles:
            The quantiles we want to evaluate
    Returns:
        thetas: list
            A list of thetas values to build the selection function
        dict_q: dict
            A dictionary with values



    """
    thetas = []
    sc = pd.DataFrame(np.c_[ys, z], columns=["y_true", "y_scores"])
    print(sc["y_true"].unique())
    sc.sort_index(inplace=True)
    sc1, sc2 = train_test_split(
        sc, stratify=sc["y_true"], test_size=0.5, random_state=42
    )
    list_u = []
    list_l = []
    dict_q = {q: [] for q in quantiles}
    for db in [sc1, sc2, sc]:
        db = db.reset_index()
        auc_roc = roc_auc_score(db["y_true"], db["y_scores"])
        n, npos = len(db["y_true"]), np.sum(db["y_true"])
        pneg = 1 - np.mean(db["y_true"])
        u_pos = int(auc_roc * pneg * n)
        pos_sorted = np.argsort(db["y_scores"])
        if isinstance(db["y_true"], pd.Series):
            tp = np.cumsum(db["y_true"].iloc[pos_sorted[::-1]])
        else:
            tp = np.cumsum(db["y_true"][pos_sorted[::-1]])
        l_pos = n - np.searchsorted(tp, auc_roc * npos + 1, side="right")
        u = db["y_scores"][pos_sorted[u_pos]]
        l = db["y_scores"][pos_sorted[l_pos]]
        list_u.append(u)
        list_l.append(l)
    # better estimate
    tau = 1 / np.sqrt(2)
    u_star = list_u[2] * tau + (1 - tau) * (0.5 * list_u[1] + 0.5 * list_u[0])
    l_star = list_l[2] * tau + (1 - tau) * (0.5 * list_l[1] + 0.5 * list_l[0])
    pos = (u_star + l_star) * 0.5
    print(pos)
    sorted_scores = np.sort(z)
    base = np.searchsorted(sorted_scores, pos)
    for i, q in enumerate(quantiles):
        delta = int(n * q / 2)
        l_b = max(0, round(base - delta))
        u_b = min(n - 1, round(base + delta))
        t1 = sorted_scores[l_b]
        t2 = sorted_scores[u_b]
        # locallist.append( [t1, t2] )
        thetas.append([t1, t2])
        dict_q[q].append([t1, t2])
        print(t1, t2)
    return thetas, dict_q
