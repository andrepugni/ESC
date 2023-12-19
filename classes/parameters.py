dict_params = {
    "SVHN": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
    },
    "catsdogs": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {"b_norm": False, "input_size": 64},
            "arch": "VGG",
        },
    },
    "cifar10": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.001},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {"b_norm": False, "input_size": 32},
            "arch": "VGG",
        },
    },
    "food101": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {},
            "arch": "Resnet34",
        },
    },
    "stanfordcars": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {},
            "arch": "Resnet34",
        },
    },
    "MNIST": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {},
            "arch": "Resnet34",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {},
            "arch": "Resnet34",
        },
    },
    "waterbirds": {
        "plugin": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True},
            "dict_arch": {},
            "arch": "Resnet50",
        },
        "sat": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99},
            "dict_arch": {},
            "arch": "Resnet50",
        },
        "selnet_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet50",
        },
        "sat_em": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "pretrain": 0, "momentum": 0.99, "beta": 0.01},
            "dict_arch": {},
            "arch": "Resnet50",
        },
        "selnet": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "lamda": 32, "alpha": 0.5},
            "dict_arch": {},
            "arch": "Resnet50",
        },
        "dg": {
            "dict_opt": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "lr": 0.1,
                "wd": 5e-04,
            },
            "dict_loss": {"td": True, "reward": 2},
            "dict_arch": {},
            "arch": "Resnet50",
        },
    },
}
dict_default_trials = {
    "SVHN": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 5,
            }
        },
    },
    "catsdogs": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 2,
            }
        },
    },
    "cifar10": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 5,
            }
        },
    },
    "cifar100": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 5,
            }
        },
    },
    "FashionMNIST": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "food101": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 50,
            }
        },
    },
    "stanfordcars": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 98,
            }
        },
    },
    "MNIST": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "waterbirds": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 2,
            }
        },
    },
    "celeba": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 2,
            }
        },
    },
    "xray": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 2,
            }
        },
    },
    "oxfordpets": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -3,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 0,
                "momentum": 0.99,
                "exp_beta": -3,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "reward": 2,
            }
        },
    },
    "organamnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "organcmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "chestmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "breastmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "dermamnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "bloodmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "pneumoniamnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "tissuemnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "octmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "pathmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "retinamnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
    "organsmnist": {
        "plugin": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "cn": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
            }
        },
        "sele": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "reg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
            }
        },
        "sat": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
            }
        },
        "selnet_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "selnet_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "exp_beta": -2,
                "zero_init_residual": False,
            }
        },
        "sat_em": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "sat_te": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "pretrain": 60,
                "momentum": 0.9,
                "zero_init_residual": False,
                "exp_beta": -2,
            }
        },
        "selnet": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "b_norm": True,
                "exp_lamda": 5,
                "alpha": 0.5,
                "zero_init_residual": False,
            }
        },
        "dg": {
            "params": {
                "optimizer": "SGD",
                "nesterov": False,
                "momentum_sgd": 0.9,
                "exp_lr": -1,
                "exp_wd": -4,
                "time_decay": True,
                "zero_init_residual": False,
                "reward": 5,
            }
        },
    },
}
