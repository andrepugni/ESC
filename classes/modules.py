import torch
import rtdl
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import math


def cast(typ, val):
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val


class VGG(Module):
    """ "
    VGG module is taken from https://github.com/LayneH/SAT-selective-cls/blob/main/models/cifar/vgg.py implementation
    """

    def __init__(self, features, d_out=1000, input_size=32, main_body: str = "VGG"):
        super(VGG, self).__init__()
        self.features = features
        self.main_body = main_body
        if input_size == 32:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, d_out),
            )
        elif input_size == 64:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, d_out),
            )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FTResNet(Module):
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 2,
        n_blocks: int = 2,
        d_main: int = 3,
        d_hidden: int = 4,
        dropout_first=0.25,
        dropout_second=0.00,
    ):
        """

        :param d_in: int
                the number of continuous variables
        :param cat_cardinalities: list
                a list with the number of dimensions for each categorical variable
        :param d_token: int
                the size of the embeddings
        :param d_out:  int
                the final output size
        :param n_blocks: int
                the number of Blocks
        :param d_main: int
                the input size (or, equivalently, the output size) of each Block
        :param d_hidden: int
                the output size of the first linear layer in each Block
        :param dropout_first: float
                the dropout rate of the first dropout layer in each Block.
        :param dropout_second: float
                the dropout rate of the second dropout layer in each Block.
        """
        super(FTResNet, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.FT = rtdl.FeatureTokenizer(self.d_in, self.cat_cardinalities, self.d_token)
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.d_hidden = d_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second
        self.d_out = d_out
        self.resnet = rtdl.ResNet.make_baseline(
            d_in=self.d_token,
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout_first,
            dropout_second=self.dropout_second,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        x = self.FT(x_num, x_cat)
        x = x[:, -1]
        x = self.resnet(x)
        return x


class CatResNet(Module):
    #     def __init__(
    #         self,
    #         n_num_features: int,
    #         cat_tokenizer: rtdl.CategoricalFeatureTokenizer,
    #         mlp_kwargs: Dict[str, Any],
    #     ):
    #         super().__init__()
    #         self.cat_tokenizer = cat_tokenizer
    #         self.model = rtdl.MLP.make_baseline(
    #             d_in=n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token,
    #             **mlp_kwargs,
    #         )
    #
    #     def forward(self, x_num, x_cat):
    #         return self.model(
    #             torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
    #         )
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 2,
        n_blocks: int = 2,
        d_main: int = 3,
        d_hidden: int = 4,
        dropout_first=0.25,
        dropout_second=0.00,
    ):
        """

        :param d_in: int
                the number of continuous variables
        :param cat_cardinalities:
                the
        :param d_token:
        :param d_out:
        :param n_blocks: the number of Blocks
        :param d_main: the input size (or, equivalently, the output size) of each Block
        :param d_hidden: the output size of the first linear layer in each Block
        :param dropout_first: the dropout rate of the first dropout layer in each Block.
        :param dropout_second: the dropout rate of the second dropout layer in each Block.
        """
        super(CatResNet, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.CAT = rtdl.CategoricalFeatureTokenizer(
            self.cat_cardinalities, self.d_token, True, "uniform"
        )
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.d_hidden = d_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second
        self.d_out = d_out
        self.resnet = rtdl.ResNet.make_baseline(
            d_in=self.d_in + len(self.cat_cardinalities) * self.d_token,
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout_first,
            dropout_second=self.dropout_second,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        x_cat = self.CAT(x_cat)
        x = torch.cat([x_num, x_cat.flatten(1, -1)], dim=1)
        x = self.resnet(x)
        return x


class TabCatResNet(Module):
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 2,
        n_blocks: int = 2,
        d_main: int = 3,
        d_hidden: int = 4,
        dropout_first=0.25,
        dropout_second=0.00,
    ):
        """

        :param d_in: int
                the number of continuous variables
        :param cat_cardinalities:
                the
        :param d_token:
        :param d_out:
        :param n_blocks: the number of Blocks
        :param d_main: the input size (or, equivalently, the output size) of each Block
        :param d_hidden: the output size of the first linear layer in each Block
        :param dropout_first: the dropout rate of the first dropout layer in each Block.
        :param dropout_second: the dropout rate of the second dropout layer in each Block.
        """
        super(TabCatResNet, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        if self.cat_cardinalities != []:
            self.CAT = rtdl.CategoricalFeatureTokenizer(
                self.cat_cardinalities, self.d_token, True, "uniform"
            )
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.d_hidden = d_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second
        self.d_out = d_out
        self.resnet = rtdl.ResNet.make_baseline(
            d_in=self.d_in + len(self.cat_cardinalities) * self.d_token,
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout_first,
            dropout_second=self.dropout_second,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_num, x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = torch.cat([x_num], dim=1)
        x = self.resnet(x)
        return x

    def get_features(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_num, x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = torch.cat([x_num], dim=1)
        x = self.resnet.first_layer(x)
        x = self.resnet.blocks(x)
        return x


class TabFTTransformer(Module):
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 2,
        n_blocks: int = 2,
        attention_dropout: float = 0.2,
        ffn_d_hidden: int = 192,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0,
    ):
        """

        :param d_in:
        :param cat_cardinalities:
        :param d_token:
        :param d_out:
        :param n_blocks:
        :param attention_dropout:
        :param ffn_d_hidden:
        :param ffn_dropout:
        :param residual_dropout:
        """

        super(TabFTTransformer, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token

        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.d_out = d_out
        self.ftt = rtdl.FTTransformer.make_baseline(
            n_num_features=self.d_in,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x = self.ftt(x_num, x_cat)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x = self.ftt(None, x_cat)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = self.ftt(x_num, None)
        return x

    def get_features(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x = self.ftt.feature_tokenizer(x_num, x_cat)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x = self.ftt.feature_tokenizer(None, x_cat)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = self.ftt.feature_tokenizer(x_num, None)
        x = self.ftt.cls_token(x)
        for layer_idx, layer in enumerate(self.ftt.transformer.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.ftt.transformer.last_layer_query_idx
                if layer_idx + 1 == len(self.ftt.transformer.blocks)
                else None
            )
            x_residual = self.ftt.transformer._start_residual(layer, "attention", x)
            x_residual, _ = layer["attention"](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self.ftt.transformer._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self.ftt.transformer._end_residual(layer, "attention", x, x_residual)

            x_residual = self.ftt.transformer._start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self.ftt.transformer._end_residual(layer, "ffn", x, x_residual)
            x = layer["output"](x)
        return x


class HeadSelectiveNet(Module):
    """
    Module for the head implementation of Selective Net.
    This is intended to substitute the head structure of rtdl modules
    """

    def __init__(
        self,
        d_in: int = 128,
        d_out: int = 2,
        batch_norm: str = "batch_norm",
        main_body: str = "resnet",
        pre_norm: bool = True,
    ):
        super(HeadSelectiveNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.batch_norm = batch_norm
        self.main_body = main_body
        self.dense_class = torch.nn.Linear(self.d_in, self.d_out)
        self.dense_selec_1 = torch.nn.Linear(self.d_in, int(self.d_in / 2))
        # if the model is VGG-based we apply an additional linear layer as in SelNet paper
        if self.main_body == "VGG":
            self.first_layer = nn.Sequential(
                nn.Linear(self.d_in, self.d_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.d_in),
                nn.Dropout(0.5),
            )
        # if the model is a resnet-based or a transformer we apply normalization before the head layers
        # depending on the model we change the normalization
        # if (self.main_body in ["resnet", "transformer"]) and (
        #     self.batch_norm == "batch_norm"
        # ):
        if (pre_norm) and (self.batch_norm == "batch_norm"):
            self.pre_norm = torch.nn.BatchNorm1d(self.d_in)
        elif (pre_norm) and (self.batch_norm == "layer_norm"):
            self.pre_norm = torch.nn.LayerNorm(self.d_in)
        else:
            self.pre_norm = None
        # depending on the model we change the normalization
        if self.batch_norm == "batch_norm":
            self.batch_norm = torch.nn.BatchNorm1d(int(self.d_in / 2))
            # if the model is a resnet-based or a transformer we apply normalization before the head layers
        elif self.batch_norm == "layer_norm":
            self.batch_norm = torch.nn.LayerNorm(int(self.d_in / 2))
        self.dense_selec_2 = torch.nn.Linear(int(self.d_in / 2), 1)
        self.dense_auxil = torch.nn.Linear(self.d_in, self.d_out)

    def forward(self, x):
        if self.main_body == "transformer":
            x = x[:, -1]
        if self.main_body == "VGG":
            x = self.first_layer(x)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        h = self.dense_class(x)
        # h = torch.nn.functional.softmax(h, dim=1)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil(x)
        # a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h, g], 1)
        return hg, a


class HeadConfidNet(Module):
    """
    Module for the head implementation of ConfidNet.
    This is intended to substitute the head structure of rtdl modules
    """

    def __init__(
        self,
        d_in: int = 128,
        batch_norm: str = "batch_norm",
        main_body: str = "resnet",
        pre_norm: bool = True,
    ):
        super(HeadConfidNet, self).__init__()
        self.d_in = d_in
        self.batch_norm = batch_norm
        self.main_body = main_body
        self.uncertainty_1 = torch.nn.Linear(self.d_in, 400)
        self.uncertainty_2 = torch.nn.Linear(400, 400)
        self.uncertainty_3 = torch.nn.Linear(400, 400)
        self.uncertainty_4 = torch.nn.Linear(400, 400)
        self.uncertainty_5 = torch.nn.Linear(400, 1)

        # if the model is VGG-based we apply an additional linear layer as in SelNet paper
        if self.main_body == "VGG":
            self.first_layer = nn.Sequential(
                nn.Linear(self.d_in, self.d_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.d_in),
                nn.Dropout(0.5),
            )
        # if the model is a resnet-based or a transformer we apply normalization before the head layers
        # depending on the model we change the normalization
        # if (self.main_body in ["resnet", "transformer"]) and (
        #     self.batch_norm == "batch_norm"
        # ):
        if (pre_norm) and (self.batch_norm == "batch_norm"):
            self.pre_norm = torch.nn.BatchNorm1d(self.d_in)
        elif (pre_norm) and (self.batch_norm == "layer_norm"):
            self.pre_norm = torch.nn.LayerNorm(self.d_in)
        else:
            self.pre_norm = None
        # depending on the model we change the normalization
        if self.batch_norm == "batch_norm":
            self.batch_norm = torch.nn.BatchNorm1d(int(self.d_in / 2))
            # if the model is a resnet-based or a transformer we apply normalization before the head layers
        elif self.batch_norm == "layer_norm":
            self.batch_norm = torch.nn.LayerNorm(int(self.d_in / 2))

    def forward(self, x):
        if self.main_body == "transformer":
            x = x[:, -1]
        if self.main_body == "VGG":
            x = self.first_layer(x)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        uncertainty = F.relu(self.uncertainty_1(x))
        uncertainty = F.relu(self.uncertainty_2(uncertainty))
        uncertainty = F.relu(self.uncertainty_3(uncertainty))
        uncertainty = F.relu(self.uncertainty_4(uncertainty))
        uncertainty = self.uncertainty_5(uncertainty)
        return uncertainty


def SelCalNet():
    """Implements a feed-forward MLP.
    taken from https://github.com/ajfisch/calibrated-selective-classification/blob/main/src/models.py
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
    ):
        super(SelCalNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, 1)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)
