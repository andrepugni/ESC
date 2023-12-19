# -*- coding: utf-8 -*-
import os.path
from classes.attributes import *
from classes.utils import *
import argparse
from sklearn.model_selection import StratifiedKFold
from scipy import stats as st
from codecarbon import EmissionsTracker, track_emissions


def load_data(
    filename,
    device,
    dict_tabular,
    dict_atts,
    architecture,
    d_arch,
    data_fold="data/clean",
):
    """
    Load data
    :param filename:
    :param device:
    :param dict_tabular:
    :param dict_atts:
    :param architecture:
    :param d_arch:
    :param data_fold:
    :return:
    """
    input_size = None
    if dict_tabular[filename]:
        training_set = TabularDataset(
            filename, atts=dict_atts[filename], set="train", device=device
        )
        test_set = TabularDataset(
            filename, atts=dict_atts[filename], set="test", device=device
        )
        calibration_set = TabularDataset(
            filename, atts=dict_atts[filename], set="cal", device=device
        )
        x_cont = training_set.x_num.shape[1]
        try:
            cat_dim = (training_set.x_cat.max(dim=0).values + 1).tolist()
        except:
            cat_dim = []
        return training_set, test_set, calibration_set, x_cont, cat_dim
    else:
        transform_train, transform_test, input_size = transformation(
            filename, also_inp_size=True
        )
        training_set = ImgFolder(
            "{}/{}/train".format(data_fold, filename),
            transform=transform_train,
        )
        test_set = ImgFolder(
            "{}/{}/test".format(data_fold, filename),
            transform=transform_test,
        )
        calibration_set = ImgFolder(
            "{}/{}/cal".format(data_fold, filename),
            transform=transform_test,
        )
        if architecture == "vgg":
            d_arch["input_size"] = input_size
        return (
            training_set,
            test_set,
            calibration_set,
            input_size,
            transform_train,
            transform_test,
        )


def testing_torch(
    filename,
    baseline,
    device,
    coverages=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    setting="best",
    tabular_dictionary=dict_tabular,
    structure_dictionary=dict_arch,
    epochs_dictionary=dict_epochs,
    baseline_dictionary=dict_baseline,
    batch_dictionary=dict_batch,
    data_fold="data/clean",
    model_fold="models",
    result_fold="test_results",
    boot_iter=100,
    seed=42,
    verb=False,
):
    """

    Args:
        filename: str
            The name of the dataset
        baseline:
            The name of the baseline. Possible values are:
                - 'dg' for a network trained using Deep Gamblers loss (Liu et al., 2023)
                - 'sat' for a network trained using Self-Adaptive-Training loss (Huang et al., 2020)
                - 'sat_te' for a network trained using Self-Adaptive-Training loss + entropy term (Feng et al., 2023)
                - 'selnet' for a network trained using SelectiveNet loss (Geifman et al., 2019)
                - 'selnet_te' for a network trained using SelectiveNet loss + entropy term (Feng et al., 2023)
                - 'plugin' for the SR method (Geifman et al., 2017)
                - 'sat_sr' for a network trained using Self-Adaptive-Training loss, but using SR confidence
                - 'sat_te_sr' for a network trained using Self-Adaptive-Training + Entropy loss, but using SR confidence
                - 'selnet_sr' for a network trained using SelectiveNet loss, but using SR confidence
                - 'selnet_te_sr'for a network trained using SelectiveNet+ Entropy loss, but using SR confidence
                - 'scross' for a network using SCross (Pugnana and Ruggieri, 2023a)
                - 'pluginauc' for a network using PlugInAUC (Pugnana and Ruggieri, 2023b)
                - 'aucross' for a network using AUCross (Pugnana and Ruggieri, 2023b)
                - 'ensemble' for a network using Ensemble (Lakshminarayanan et al., 2017)
                - 'ensemble_sr' for a network using EnsembleSR (Lakshminarayanan et al., 2017)
                - 'sele' for a network using SELE (Franc et al., 2023)
                - 'reg' for a network using REG (Franc et al., 2023a)
                - 'cn' for a network using ConfidNet (Corbière et al., 2019)
        device: str
            The string for the device to use during training.
        coverages: list
            The list with desired target coverages $c$. The default is [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        setting: str
            Pick the setting. Default is best
        tabular_dictionary: dict
            A dictionary containing information about datasets. Every key is a dataset name and its value a boolean:
                - True, if the dataset is tabular
                - False, if not
        structure_dictionary: dict
            A dictionary containing information about architectures. Every key is a dataset name and its value the model_type
        epochs_dictionary: dict
            A dictionary containing information about epochs. Every key is a dataset name and its value the epochs
        baseline_dictionary: dict
            A dictionary containing information about hyperparameters. Every key is a baseline and its value the main hyperparameters
        batch_dictionary: dict
            A dictionary containing information about batch size. Every key is a dataset name and its value the batch sizes
        data_fold: str
            The path where data are stored
        model_fold: str
            The path where models are stored
        result_fold: str
            The path where test results are stored
        boot_iter: int
            The number of bootstrap iterations. The default is 100.
        seed: int
            The seed used for reproducibility. The default is 42.
        verb: bool
            Boolean value. If True, training prints epochs evolutions.
    """
    print(filename)
    print("\n\n\n")
    print(baseline)
    print("\n\n\n")
    if os.path.exists(result_fold) == False:
        os.mkdir(result_fold)
    if os.path.exists("{}/{}".format(result_fold, filename)) == False:
        os.mkdir("{}/{}".format(result_fold, filename))
    bsize = batch_dictionary[filename]
    meta = baseline_dictionary[baseline]
    epochs = epochs_dictionary[filename]
    tabular = tabular_dictionary[filename]
    architecture = structure_dictionary[filename]
    quantiles = [1 - cov for cov in coverages]
    if (architecture == "resnet") & (tabular == True):
        arch = "TabResnet"
    elif architecture == "transformer":
        arch = "TabFTTransformer"
    elif architecture == "vgg":
        arch = "VGG"
    elif (architecture == "resnet") & (tabular == False):
        arch = "Resnet34"
    elif architecture == "resnet50":
        arch = "Resnet50"
    elif architecture == "resnet18":
        arch = "Resnet18"
    if setting == "best":
        print("meta")
        d_arch, d_loss, d_opt, trial, arch = get_best_params(meta, filename, arch)
        setting_string = ""
        print("Best trial is number {}".format(trial))
    elif setting == "default":
        from classes.parameters import dict_params

        d_arch = dict_params[filename][meta]["dict_arch"]
        d_loss = dict_params[filename][meta]["dict_loss"]
        d_opt = dict_params[filename][meta]["dict_opt"]
        arch = dict_params[filename][meta]["arch"]
        setting_string = "DEFAULT"
        trial = "DEFAULT"
    else:
        raise NotImplementedError("The other setting not yet implemented")
    if dict_tabular[filename]:
        training_set, test_set, calibration_set, x_cont, cat_dim = load_data(
            filename, device, dict_tabular, dict_atts, architecture, d_arch
        )
    else:
        (
            training_set,
            test_set,
            calibration_set,
            input_size,
            transform_train,
            transform_test,
        ) = load_data(filename, device, dict_tabular, dict_atts, architecture, d_arch)
    n_classes = len(training_set.classes)
    counts_mostcommon = np.max(np.unique(training_set.targets, return_counts=True)[1])
    perc_train = counts_mostcommon / len(training_set)
    most_common_class = st.mode(training_set.targets)[0][0]
    y_test = np.array(test_set.targets)
    set_seed(seed)
    if baseline in ["aucross", "scross"]:
        cv = 5
        # we use cross-fitting hence no need for calibration dataset (this is used as an additional part of training set)
        trainingcal_set = torch.utils.data.ConcatDataset(
            [training_set, calibration_set]
        )
        # here we store the targets
        y_traincal = np.vstack(
            [
                np.array(trainingcal_set.datasets[0].targets).reshape(-1, 1),
                np.array(trainingcal_set.datasets[1].targets).reshape(-1, 1),
            ]
        ).flatten()
        times = []
        torch.manual_seed(seed)
        set_seed(42)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        if dict_tabular[filename]:
            print("Dataset: {}, Tabular, n classes: {}".format(filename, n_classes))
            trainloader = torch.utils.data.DataLoader(
                trainingcal_set, batch_size=dict_batch[filename], shuffle=True
            )
            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=bsize,
                shuffle=False,
            )
            modelfinal = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
        else:
            print("Dataset: {}, Image, n classes: {}".format(filename, n_classes))
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=int(bsize / 4), shuffle=False
            )
            modelfinal = image_model(arch, n_classes, meta, d_arch)
        if (baseline == "aucross") and (len(np.unique(y_traincal)) > 2):
            print("The dataset target is not binary. Skipping AUCross.")
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            splits = skf.split(np.zeros(len(y_traincal)), y_traincal)
            # here we instantiate a dictionary for each of the folds
            dict_folds = {}
            # here we define variables for confidence and indexes
            z = []
            idx = []
            ys = []
            folds = []
            # for each fold
            for k, split in enumerate(splits):
                train_idx, test_idx = split[0], split[1]
                torch.manual_seed(seed)
                set_seed(seed)
                g_seed = torch.Generator()
                g_seed.manual_seed(seed)
                if dict_tabular[filename]:
                    training_set = TabularDataset(
                        filename, atts=dict_atts[filename], set="train", device=device
                    )
                    calibration_set = TabularDataset(
                        filename, atts=dict_atts[filename], set="cal", device=device
                    )
                    trainingcal_set = torch.utils.data.ConcatDataset(
                        [training_set, calibration_set]
                    )
                    train_ = torch.utils.data.Subset(trainingcal_set, indices=train_idx)
                    hold_ = torch.utils.data.Subset(trainingcal_set, indices=test_idx)
                    trainfold_dl = torch.utils.data.DataLoader(
                        train_,
                        batch_size=int(dict_batch[filename] * (cv - 1) / cv),
                        shuffle=True,
                    )
                    holdfold_dl = torch.utils.data.DataLoader(
                        hold_, batch_size=int(dict_batch[filename] / 4), shuffle=False
                    )
                    modelfold = tabular_model(
                        arch, x_cont, cat_dim, n_classes, meta, d_arch
                    )
                    path_fold_model = "{}/{}/{}_cross_{}fold{}out{}_ep{}.pt".format(
                        model_fold, filename, setting_string, arch, k + 1, cv, epochs
                    )
                else:
                    training_set = ImgFolder(
                        "{}/{}/train".format(data_fold, filename),
                        transform=transform_train,
                    )
                    calibration_set = ImgFolder(
                        "{}/{}/cal".format(data_fold, filename),
                        transform=transform_train,
                    )
                    trainingcal_set = torch.utils.data.ConcatDataset(
                        [training_set, calibration_set]
                    )
                    train_ = torch.utils.data.Subset(trainingcal_set, indices=train_idx)
                    print(train_.dataset.datasets[0])
                    trainfold_dl = torch.utils.data.DataLoader(
                        train_,
                        batch_size=int(dict_batch[filename] * (cv - 1) / cv),
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g_seed,
                    )
                    modelfold = image_model(arch, n_classes, meta, d_arch)
                    path_fold_model = "{}/{}/best_scross_{}fold{}out{}_ep{}.pt".format(
                        model_fold, filename, setting_string, arch, k + 1, cv, epochs
                    )
                path_interm = "{}/{}/{}_cross_{}fold{}out{}".format(
                    model_fold, filename, setting_string, arch, k + 1, cv
                )
                if os.path.exists(path_fold_model):
                    modelfold.to(device)
                    modelfold.load_state_dict(
                        torch.load(path_fold_model, map_location=device)
                    )
                else:
                    optimizer = make_optimizer(d_opt, modelfold)
                    start_time = time()
                    modelfold = train(
                        modelfold,
                        device,
                        epochs,
                        optimizer,
                        "ce",
                        trainfold_dl,
                        td=d_loss["td"],
                        verbose=verb,
                        path_interm=path_interm,
                    )
                    end_time = time()
                    times.append(end_time - start_time)
                    folds.append(k + 1)
                    torch.save(modelfold.state_dict(), path_fold_model)
                    res_time = pd.DataFrame(
                        zip(folds, times), columns=["folds", "time_to_fit"]
                    )
                    res_time.to_csv(
                        result_fold
                        + "/{}/TESTING_time_CROSS_{}_{}_{}baseline.csv".format(
                            filename, filename, epochs, setting_string
                        ),
                        index=False,
                    )
                dict_folds[k] = copy.deepcopy(modelfold)
                if dict_tabular[filename] == False:
                    training_set = ImgFolder(
                        "{}/{}/train".format(data_fold, filename),
                        transform=transform_test,
                    )
                    calibration_set = ImgFolder(
                        "{}/{}/cal".format(data_fold, filename),
                        transform=transform_test,
                    )
                    trainingcal_set = torch.utils.data.ConcatDataset(
                        [training_set, calibration_set]
                    )
                    hold_ = torch.utils.data.Subset(trainingcal_set, indices=test_idx)
                    #                 hold_.dataset.datasets[0].transform = transform_test
                    #                 hold_.dataset.datasets[1].transform = transform_test
                    holdfold_dl = torch.utils.data.DataLoader(
                        hold_,
                        batch_size=int(dict_batch[filename] / 4),
                        shuffle=False,
                        pin_memory=True,
                    )
                if baseline == "aucross":
                    scores = predict_proba(
                        modelfold, device, holdfold_dl, meta="plugin"
                    )
                    conf = scores[:, 1].reshape(-1, 1)
                elif baseline == "scross":
                    conf = np.max(
                        predict_proba(modelfold, device, holdfold_dl, meta="plugin"),
                        axis=1,
                    ).reshape(-1, 1)
                z.append(conf)
                t_id = test_idx.reshape(-1, 1)
                ys.append(y_traincal[test_idx].reshape(-1, 1))
                idx.append(t_id)
            confs = np.vstack(z).flatten()
            indexes = np.vstack(idx).flatten()
            ys = np.vstack(ys).flatten()
            path_final_model = "{}/{}/{}_cross_{}final_ep{}.pt".format(
                model_fold, filename, arch, setting_string, epochs
            )
            if dict_tabular[filename] == False:
                path_final_model = "{}/{}/{}_scross_{}final_ep{}.pt".format(
                    model_fold, filename, arch, setting_string, epochs
                )
            path_final_interm = "{}/{}/{}_cross_{}final".format(
                model_fold, filename, arch, setting_string
            )
            if os.path.exists(path_final_model):
                modelfinal.to(device)
                modelfinal.load_state_dict(
                    torch.load(path_final_model, map_location=device)
                )
            else:
                optimizer = make_optimizer(d_opt, modelfinal)
                start_time = time()
                if dict_tabular[filename] == False:
                    training_set = ImgFolder(
                        "{}/{}/train".format(data_fold, filename),
                        transform=transform_train,
                    )
                    calibration_set = ImgFolder(
                        "{}/{}/cal".format(data_fold, filename),
                        transform=transform_train,
                    )
                    trainingcal_set = torch.utils.data.ConcatDataset(
                        [training_set, calibration_set]
                    )
                    trainloader = torch.utils.data.DataLoader(
                        trainingcal_set,
                        batch_size=dict_batch[filename],
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g_seed,
                    )
                modelfinal = train(
                    modelfinal,
                    device,
                    epochs,
                    optimizer,
                    "ce",
                    trainloader,
                    td=d_loss["td"],
                    verbose=verb,
                    path_interm=path_final_interm,
                )
                end_time = time()
                times.append(end_time - start_time)
                folds.append("full")
                res_time = pd.DataFrame(
                    zip(folds, times), columns=["folds", "time_to_fit"]
                )
                res_time.to_csv(
                    result_fold
                    + "/{}/TESTING_time_CROSS_{}_{}_baseline.csv".format(
                        filename, filename, epochs
                    ),
                    index=False,
                )
                torch.save(modelfinal.state_dict(), path_final_model)
            modelfinal.eval()
            if baseline == "scross":
                sub_confs_1, sub_confs_2 = train_test_split(
                    confs, test_size=0.5, random_state=42
                )
                tau = 1 / np.sqrt(2)
                thetas = [
                    (
                        tau * np.quantile(confs, q)
                        + (1 - tau)
                        * (
                            0.5 * np.quantile(sub_confs_1, q)
                            + 0.5 * np.quantile(sub_confs_2, q)
                        )
                    )
                    for q in quantiles
                ]
                dict_q = {
                    q: (
                        tau * np.quantile(confs, q)
                        + (1 - tau)
                        * (
                            0.5 * np.quantile(sub_confs_1, q)
                            + 0.5 * np.quantile(sub_confs_2, q)
                        )
                    )
                    for q in quantiles
                }
                scores = predict_proba(modelfinal, device, testloader, meta)
                preds = np.argmax(scores, axis=1)
                confs_test = np.max(scores, axis=1)
                selected = np.digitize(confs_test, sorted(thetas), right=False)
            else:
                m = len(quantiles)
                thetas, dict_q = calibrate_aucross(ys, confs, quantiles)
                scores = predict_proba(modelfinal, device, testloader, meta)
                preds = np.argmax(scores, axis=1)
                selected = np.zeros(len(scores[:, 1])) + m
                for i, t in enumerate(reversed(thetas)):
                    t1, t2 = t[0], t[1]
                    selected[((t1 <= scores[:, 1]) & (scores[:, 1] <= t2))] = m - i - 1
            results = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=y_test,
                selected=selected,
                y_scores=scores,
                y_preds=preds,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            if os.path.exists("{}/{}".format(result_fold, filename)) == False:
                os.mkdir("{}/{}".format(result_fold, filename))
            scorecols = ["classscore_{}".format(cl) for cl in range(scores.shape[1])]
            yy = pd.DataFrame(
                np.c_[y_test, preds, selected, scores],
                columns=["true", "preds", "bands"] + scorecols,
            )
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, int(trial), filename, setting
            )
            results["b_iter"] = 0
            for b in range(1, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                tmp = get_metrics_test(
                    num_classes=n_classes,
                    coverages=coverages,
                    true=db["true"].astype(int).values,
                    selected=db["bands"].values,
                    y_scores=db[scorecols].values,
                    y_preds=db["preds"].astype(int).values,
                    meta=meta,
                    trial_num=trial,
                    dataset=filename,
                    arch=arch,
                    most_common_class=most_common_class,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
    elif baseline in ["scross_ens"]:
        cv = 5
        # we use cross-fitting hence no need for calibration dataset (this is used as an additional part of training set)
        torch.manual_seed(seed)
        set_seed(42)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        if dict_tabular[filename]:
            print("Dataset: {}, Tabular, n classes: {}".format(filename, n_classes))
            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=bsize,
                shuffle=False,
            )
            calibloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=bsize,
                shuffle=False,
            )
            modelfinal = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
        else:
            print("Dataset: {}, Image, n classes: {}".format(filename, n_classes))
            calibloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=bsize,
                shuffle=False,
            )
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=int(bsize / 4), shuffle=False
            )
            modelfinal = image_model(arch, n_classes, meta, d_arch)
        if (baseline == "aucross") and (len(np.unique(training_set.targets)) > 2):
            print("The dataset target is not binary. Skipping AUCross.")
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            splits = skf.split(np.zeros(len(training_set)), training_set.targets)
            # here we instantiate a dictionary for each of the folds
            dict_folds = {}
            # here we define variables for confidence and indexes
            z = []
            idx = []
            ys = []
            folds = []
            times = []
            # for each fold
            for k, split in enumerate(splits):
                train_idx, test_idx = split[0], split[1]
                torch.manual_seed(seed)
                set_seed(seed)
                g_seed = torch.Generator()
                g_seed.manual_seed(seed)
                if dict_tabular[filename]:
                    training_set = TabularDataset(
                        filename, atts=dict_atts[filename], set="train", device=device
                    )
                    train_ = torch.utils.data.Subset(training_set, indices=train_idx)
                    hold_ = torch.utils.data.Subset(training_set, indices=test_idx)
                    trainfold_dl = torch.utils.data.DataLoader(
                        train_,
                        batch_size=int(dict_batch[filename] * (cv - 1) / cv),
                        shuffle=True,
                    )
                    holdfold_dl = torch.utils.data.DataLoader(
                        hold_, batch_size=int(dict_batch[filename] / 4), shuffle=False
                    )
                    modelfold = tabular_model(
                        arch, x_cont, cat_dim, n_classes, meta, d_arch
                    )
                else:
                    training_set = ImgFolder(
                        "{}/{}/train".format(data_fold, filename),
                        transform=transform_train,
                    )
                    train_ = torch.utils.data.Subset(training_set, indices=train_idx)
                    trainfold_dl = torch.utils.data.DataLoader(
                        train_,
                        batch_size=int(dict_batch[filename] * (cv - 1) / cv),
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g_seed,
                    )
                    modelfold = image_model(arch, n_classes, meta, d_arch)
                    path_fold_model = (
                        "{}/{}/best_scrossens_{}fold{}out{}_ep{}.pt".format(
                            model_fold, filename, arch, k + 1, cv, epochs
                        )
                    )
                path_fold_model = "{}/{}/{}_crossens_{}fold{}out{}_ep{}.pt".format(
                    model_fold, filename, setting_string, arch, k + 1, cv, epochs
                )
                if os.path.exists(path_fold_model):
                    modelfold.to(device)
                    modelfold.load_state_dict(
                        torch.load(path_fold_model, map_location=device)
                    )
                else:
                    optimizer = make_optimizer(d_opt, modelfold)
                    start_time = time()
                    print("here we go to train")
                    modelfold = train(
                        modelfold,
                        device,
                        epochs,
                        optimizer,
                        "ce",
                        trainfold_dl,
                        td=d_loss["td"],
                        verbose=verb,
                        path_interm="{}/{}/{}_crossens_{}fold{}out{}".format(
                            model_fold, filename, setting_string, arch, k + 1, cv
                        ),
                    )
                    end_time = time()
                    times.append(end_time - start_time)
                    folds.append(k + 1)
                    torch.save(modelfold.state_dict(), path_fold_model)
                    res_time = pd.DataFrame(
                        zip(folds, times), columns=["folds", "time_to_fit"]
                    )
                    res_time.to_csv(
                        result_fold
                        + "/{}/TESTING_time_CROSS_{}_{}_{}baseline.csv".format(
                            filename, filename, epochs, setting_string
                        ),
                        index=False,
                    )
                if dict_tabular[filename] == False:
                    training_set = ImgFolder(
                        "{}/{}/train".format(data_fold, filename),
                        transform=transform_test,
                    )
                    hold_ = torch.utils.data.Subset(training_set, indices=test_idx)
                    holdfold_dl = torch.utils.data.DataLoader(
                        hold_, batch_size=int(dict_batch[filename] / 4), shuffle=False
                    )
                dict_folds[k] = copy.deepcopy(modelfold)
                if baseline == "aucross":
                    scores = predict_proba(
                        modelfold, device, holdfold_dl, meta="plugin"
                    )
                    conf = scores[:, 1].reshape(-1, 1)
                elif baseline in ["scross_ens"]:
                    conf = np.max(
                        predict_proba(modelfold, device, holdfold_dl, meta="plugin"),
                        axis=1,
                    ).reshape(-1, 1)
                z.append(conf)
                t_id = test_idx.reshape(-1, 1)
                ys.append(np.array(training_set.targets)[test_idx].reshape(-1, 1))
                idx.append(t_id)
            confs = np.vstack(z).flatten()
            y_hats = [
                predict_proba(dict_folds[k].to(device), device, testloader, meta)
                for k in range(cv)
            ]
            y_hat = np.mean(y_hats, axis=0)
            y_hats_cal = [
                predict_proba(dict_folds[k].to(device), device, calibloader, meta)
                for k in range(cv)
            ]
            y_hat_cal = np.mean(y_hats_cal, axis=0)
            preds = np.argmax(y_hat, axis=1)
            if baseline == "scross_ens":
                confs = np.max(y_hat, axis=1)
                confs_cal = np.max(y_hat_cal)
                thetas = []
                for i, cov in enumerate(coverages):
                    theta = np.quantile(confs_cal, 1 - cov)
                    thetas.append(theta)
                selected = np.digitize(confs, sorted(thetas), right=False)
            selected = selected.astype(int)
            results = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=y_test,
                selected=selected,
                y_scores=y_hat,
                y_preds=preds,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            if setting == "best":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, int(trial), filename
                )
            elif setting == "default":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, setting_string, filename
                )
            results["b_iter"] = 0
            if os.path.exists("{}/{}".format(result_fold, filename)) == False:
                os.mkdir("{}/{}".format(result_fold, filename))
            scorecols = ["classscore_{}".format(cl) for cl in range(y_hat.shape[1])]
            yy = pd.DataFrame(
                np.c_[y_test, preds, selected, y_hat],
                columns=["true", "preds", "bands"] + scorecols,
            )
            for b in range(1, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                tmp = get_metrics_test(
                    num_classes=n_classes,
                    coverages=coverages,
                    true=db["true"].values,
                    selected=db["bands"].values,
                    y_scores=db[scorecols].values,
                    y_preds=db["preds"].values,
                    meta=meta,
                    trial_num=trial,
                    dataset=filename,
                    arch=arch,
                    most_common_class=most_common_class,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
    elif baseline in ["ensemble", "ensemble_sr"]:
        seeds = [42, 73, 123, 456, 789, 999, 1111, 2222, 3333, 4444]
        dict_seeds = {s: None for s in seeds}
        st_time = time()
        times = []
        cov_to_train = []
        results = pd.DataFrame()
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, "BEST", filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        for i, seed_value in enumerate(seeds):
            set_seed(seed_value)
            g_seed = torch.Generator()
            g_seed.manual_seed(seed_value)
            # here we build the model for each coverage
            if dict_tabular[filename]:
                trainloader = torch.utils.data.DataLoader(
                    training_set,
                    batch_size=bsize,
                    shuffle=True,
                )
                calibloader = torch.utils.data.DataLoader(
                    calibration_set,
                    batch_size=bsize,
                    shuffle=False,
                )
                testloader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=bsize,
                    shuffle=False,
                )
                model = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
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
                testloader = torch.utils.data.DataLoader(
                    test_set, batch_size=int(bsize / 4), shuffle=False
                )
                calibloader = torch.utils.data.DataLoader(
                    calibration_set, batch_size=int(bsize / 4), shuffle=False
                )
                model = image_model(arch, n_classes, meta, d_arch)
            # here we train model for every coverage

            if setting == "best":
                if os.path.exists("{}/{}".format(model_fold, filename)) == False:
                    os.mkdir("{}/{}".format(model_fold, filename))
                if seed_value == 42:
                    path_seed_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                        model_fold, filename, arch, meta, int(trial), epochs
                    )
                else:
                    path_seed_model = "{}/{}/{}_{}_xx_tr{}_ep{}_seed{}.pt".format(
                        model_fold, filename, arch, meta, int(trial), epochs, seed_value
                    )
            elif setting == "default":
                raise NotImplementedError("Setting Not Yet Implemented")
            if os.path.exists(path_seed_model):
                model.to(device)
                model.load_state_dict(torch.load(path_seed_model, map_location=device))
                model.eval()
                dict_seeds[seed_value] = copy.deepcopy(model)
            else:
                optimizer = make_optimizer(d_opt, model)
                start_time = time()
                model = train(
                    model,
                    device,
                    epochs,
                    optimizer,
                    "ce",
                    trainloader,
                    td=d_loss["td"],
                    verbose=verb,
                    seed=seed_value,
                    path_interm="{}/{}/{}_{}_xx_tr{}_seed{}".format(
                        model_fold, filename, arch, meta, int(trial), epochs, seed_value
                    ),
                )
                model.eval()
                end_time = time()
                torch.save(model.state_dict(), path_seed_model)
                dict_seeds[seed_value] = copy.deepcopy(model)
                time_to_fit = end_time - start_time
                times.append(time_to_fit)
                cov_to_train.append(seed_value)
                res_time = pd.DataFrame(
                    zip(cov_to_train, times), columns=["coverage", "time_to_fit"]
                )
                res_time.to_csv(
                    result_fold
                    + "/{}/TESTING_time_{}_{}_{}_{}baseline.csv".format(
                        filename, meta, filename, epochs, setting_string
                    ),
                    index=False,
                )
        set_seed(42)
        y_hats = [
            predict_proba(dict_seeds[s].to(device), device, testloader, meta)
            for s in seeds
        ]
        y_hat = np.mean(y_hats, axis=0)
        y_hats_cal = [
            predict_proba(dict_seeds[s].to(device), device, calibloader, meta)
            for s in seeds
        ]
        y_hat_cal = np.mean(y_hats_cal, axis=0)
        preds = np.argmax(y_hat, axis=1)
        if baseline == "ensemble":
            confs = get_confidence_ensemble(y_hats, y_hat)
            confs_cal = get_confidence_ensemble(y_hats_cal, y_hat_cal)
            thetas = sorted(
                [np.quantile(confs_cal, q) for q in coverages], reverse=True
            )
            selected = np.zeros(confs.shape)
            for m, theta in enumerate(thetas):
                selected = np.where(confs >= theta, selected, m + 1)
        elif baseline == "ensemble_sr":
            confs = np.max(y_hat, axis=1)
            confs_cal = np.max(y_hat_cal, axis=1)
            thetas = []
            for i, cov in enumerate(coverages):
                theta = np.quantile(confs_cal, 1 - cov)
                thetas.append(theta)
            selected = np.digitize(confs, sorted(thetas), right=False)
        selected = selected.astype(int)
        results = get_metrics_test(
            num_classes=n_classes,
            coverages=coverages,
            true=y_test,
            selected=selected,
            y_scores=y_hat,
            y_preds=preds,
            meta=meta,
            trial_num=trial,
            dataset=filename,
            arch=arch,
            most_common_class=most_common_class,
        )
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, int(trial), filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        results["b_iter"] = 0
        if os.path.exists("{}/{}".format(result_fold, filename)) == False:
            os.mkdir("{}/{}".format(result_fold, filename))
        scorecols = ["classscore_{}".format(cl) for cl in range(y_hat.shape[1])]
        yy = pd.DataFrame(
            np.c_[y_test, preds, selected, y_hat],
            columns=["true", "preds", "bands"] + scorecols,
        )
        for b in range(1, boot_iter + 1):
            if b == 0:
                db = yy.copy()
            else:
                db = yy.sample(len(yy), random_state=b, replace=True)
            tmp = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=db["true"].values,
                selected=db["bands"].values,
                y_scores=db[scorecols].values,
                y_preds=db["preds"].values,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            tmp["b_iter"] = b
            results = pd.concat([results, tmp], axis=0)
            results.to_csv(results_filename, index=False)

    elif baseline in [
        "selnet",
        "selnet_sr",
        "selnet_em",
        "selnet_em_sr",
        "selnet_te",
        "selnet_te_sr",
    ]:
        dict_covs = {}
        scores = []
        preds = []
        st_time = time()
        times = []
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f)
        cov_to_train = []
        results = pd.DataFrame()
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, "BEST", filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        for i, cov in enumerate(coverages):
            set_seed(seed)
            g_seed = torch.Generator()
            g_seed.manual_seed(seed)
            # here we build the model for each coverage
            if dict_tabular[filename]:
                trainloader = torch.utils.data.DataLoader(
                    training_set,
                    batch_size=bsize,
                    shuffle=True,
                )
                calibloader = torch.utils.data.DataLoader(
                    calibration_set,
                    batch_size=bsize,
                    shuffle=False,
                )
                testloader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=bsize,
                    shuffle=False,
                )
                selnet = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
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
                testloader = torch.utils.data.DataLoader(
                    test_set, batch_size=int(bsize / 4), shuffle=False
                )
                calibloader = torch.utils.data.DataLoader(
                    calibration_set, batch_size=int(bsize / 4), shuffle=False
                )
                selnet = image_model(arch, n_classes, meta, d_arch)
            # here we train model for every coverage
            if setting == "best":
                path_cov_model = "{}/{}/{}_{}_{}_tr{}_ep{}.pt".format(
                    model_fold, filename, arch, meta, cov, int(trial), epochs
                )
            elif setting == "default":
                path_cov_model = "{}/{}/{}_{}_{}_tr{}_ep{}.pt".format(
                    model_fold, filename, arch, meta, cov, setting_string, epochs
                )
            else:
                raise NotImplementedError("Setting Not Yet Implemented")
            if os.path.exists(path_cov_model):
                selnet.to(device)
                selnet.load_state_dict(torch.load(path_cov_model, map_location=device))
                selnet.eval()
                dict_covs[cov] = copy.deepcopy(selnet)
            else:
                optimizer = make_optimizer(d_opt, selnet)
                start_time = time()
                if "_em" in meta:
                    print(cov)
                    selnet = train(
                        selnet,
                        device,
                        epochs,
                        optimizer,
                        meta,
                        trainloader,
                        td=d_loss["td"],
                        verbose=verb,
                        coverage=cov,
                        beta=d_loss["beta"],
                        alpha=d_loss["alpha"],
                        lamda=d_loss["lamda"],
                        path_interm="{}/{}/{}_{}_{}_tr{}".format(
                            model_fold, filename, arch, meta, cov, int(trial)
                        ),
                    )
                elif "_te" in meta:
                    print(cov)
                    selnet = train(
                        selnet,
                        device,
                        epochs,
                        optimizer,
                        meta,
                        trainloader,
                        td=d_loss["td"],
                        verbose=verb,
                        coverage=cov,
                        beta=d_loss["beta"],
                        alpha=d_loss["alpha"],
                        lamda=d_loss["lamda"],
                        path_interm="{}/{}/{}_{}_{}_tr{}".format(
                            model_fold, filename, arch, meta, cov, int(trial)
                        ),
                    )

                else:
                    print(cov)
                    selnet = train(
                        selnet,
                        device,
                        epochs,
                        optimizer,
                        meta,
                        trainloader,
                        td=d_loss["td"],
                        verbose=verb,
                        coverage=cov,
                        alpha=d_loss["alpha"],
                        lamda=d_loss["lamda"],
                        path_interm="{}/{}/{}_{}_{}_tr{}".format(
                            model_fold, filename, arch, meta, cov, int(trial)
                        ),
                    )

                selnet.eval()
                end_time = time()
                torch.save(selnet.state_dict(), path_cov_model)
                dict_covs[cov] = copy.deepcopy(selnet)
                time_to_fit = end_time - start_time
                times.append(time_to_fit)
                cov_to_train.append(cov)
                res_time = pd.DataFrame(
                    zip(cov_to_train, times), columns=["coverage", "time_to_fit"]
                )
                res_time.to_csv(
                    result_fold
                    + "/{}/TESTING_time_{}_{}_{}_{}baseline.csv".format(
                        filename, meta, filename, epochs, setting_string
                    ),
                    index=False,
                )
            selnet.eval()
            set_seed(seed)
            y_hat = predict_proba(selnet, device, testloader, meta)
            scores.append(y_hat)
            preds = np.argmax(y_hat, axis=1)
            confs = predict_conf(selnet, device, testloader, baseline)
            confs_cal = predict_conf(selnet, device, calibloader, baseline)
            theta = np.quantile(confs_cal, 1 - cov)
            selected = np.where(confs >= theta, 1, 0)
            if os.path.exists("{}/{}".format(result_fold, filename)) == False:
                os.mkdir("{}/{}".format(result_fold, filename))
            scorecols = ["classscore_{}".format(cl) for cl in range(y_hat.shape[1])]
            yy = pd.DataFrame(
                np.c_[y_test, preds, selected, y_hat],
                columns=["true", "preds", "bands"] + scorecols,
            )
            for b in range(0, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                tmp = get_metrics_test_selnet(
                    num_classes=n_classes,
                    coverages=[cov],
                    true=db["true"].values,
                    selected=db["bands"].values,
                    y_scores=db[scorecols].values,
                    y_preds=db["preds"].values,
                    meta=meta,
                    trial_num=trial,
                    dataset=filename,
                    arch=arch,
                    most_common_class=most_common_class,
                    true_cov=cov,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
    elif baseline in [
        "sat",
        "sat_te",
        "sat_te_sr",
        "sat_sr",
        "plugin",
        "pluginauc",
        "dg",
    ]:
        set_seed(seed)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        if dict_tabular[filename]:
            trainloader = torch.utils.data.DataLoader(
                training_set,
                batch_size=bsize,
                shuffle=True,
            )
            calibloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=bsize,
                shuffle=False,
            )
            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=bsize,
                shuffle=False,
            )
            model = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
        else:
            trainloader = torch.utils.data.DataLoader(
                training_set,
                batch_size=bsize,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
                worker_init_fn=seed_worker,
                generator=g_seed,
            )
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=int(bsize / 8), shuffle=False
            )
            calibloader = torch.utils.data.DataLoader(
                calibration_set, batch_size=int(bsize / 8), shuffle=False
            )
            model = image_model(arch, n_classes, meta, d_arch)
        if setting == "best":
            path_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                model_fold, filename, arch, meta, int(trial), epochs
            )
        elif setting == "default":
            path_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                model_fold, filename, arch, meta, setting_string, epochs
            )
            print(path_model)
        if os.path.exists(path_model):
            model.to(device)
            model.load_state_dict(torch.load(path_model, map_location=device))
            model.eval()
            set_seed(seed)
            scores = predict_proba(model, device, testloader, meta)
            preds = np.argmax(scores, axis=1)
            selected = qband(
                model, device, testloader, calibloader, baseline, coverages
            )
            results = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=y_test,
                selected=selected,
                y_scores=scores,
                y_preds=preds,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            if setting == "best":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, int(trial), filename
                )
            elif setting == "default":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, setting_string, filename
                )
            results["b_iter"] = 0
            if os.path.exists("{}/{}".format(result_fold, filename)) == False:
                os.mkdir("{}/{}".format(result_fold, filename))
            scorecols = ["classscore_{}".format(cl) for cl in range(scores.shape[1])]
            yy = pd.DataFrame(
                np.c_[y_test, preds, selected, scores],
                columns=["true", "preds", "bands"] + scorecols,
            )
            for b in range(1, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                tmp = get_metrics_test(
                    num_classes=n_classes,
                    coverages=coverages,
                    true=db["true"].values,
                    selected=db["bands"].values,
                    y_scores=db[scorecols].values,
                    y_preds=db["preds"].values,
                    meta=meta,
                    trial_num=trial,
                    dataset=filename,
                    arch=arch,
                    most_common_class=most_common_class,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
        else:
            print("Not Found the Model")
            optimizer = make_optimizer(d_opt, model)
            if meta == "plugin":
                model = train(
                    model,
                    device,
                    epochs,
                    optimizer,
                    "ce",
                    trainloader,
                    td=d_loss["td"],
                    verbose=verb,
                    path_interm="{}/{}/{}_{}_xx_tr{}".format(
                        model_fold, filename, arch, meta, int(trial)
                    ),
                )

            elif meta == "sat":
                model = train(
                    model,
                    device,
                    epochs,
                    optimizer,
                    meta,
                    trainloader,
                    td=d_loss["td"],
                    pretrain=d_loss["pretrain"],
                    momentum=d_loss["momentum"],
                    verbose=verb,
                    path_interm="{}/{}/{}_{}_xx_tr{}".format(
                        model_fold, filename, arch, meta, int(trial)
                    ),
                )
            elif meta == "sat_te":
                model = train(
                    model,
                    device,
                    epochs,
                    optimizer,
                    meta,
                    trainloader,
                    td=d_loss["td"],
                    pretrain=d_loss["pretrain"],
                    momentum=d_loss["momentum"],
                    beta=d_loss["beta"],
                    verbose=verb,
                    path_interm="{}/{}/{}_{}_xx_tr{}".format(
                        model_fold, filename, arch, meta, int(trial)
                    ),
                )
            elif meta == "dg":
                model = train(
                    model,
                    device,
                    epochs,
                    optimizer,
                    "ce",
                    trainloader,
                    td=d_loss["td"],
                    reward=d_loss["reward"],
                    verbose=verb,
                    path_interm="{}/{}/{}_{}_xx_tr{}".format(
                        model_fold, filename, arch, meta, int(trial)
                    ),
                )
            torch.save(model.state_dict(), path_model)
            set_seed(seed)
            model.eval()
            scores = predict_proba(model, device, testloader, meta)
            preds = np.argmax(scores, axis=1)
            selected = qband(
                model, device, testloader, calibloader, baseline, coverages
            )
            results = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=y_test,
                selected=selected,
                y_scores=scores,
                y_preds=preds,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            if setting == "best":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, int(trial), filename
                )
            elif setting == "default":
                results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                    result_fold, filename, baseline, arch, setting_string, filename
                )
            results["b_iter"] = 0
            if os.path.exists("{}/{}".format(result_fold, filename)) == False:
                os.mkdir("{}/{}".format(result_fold, filename))
            scorecols = ["classscore_{}".format(cl) for cl in range(scores.shape[1])]
            yy = pd.DataFrame(
                np.c_[y_test, preds, selected, scores],
                columns=["true", "preds", "bands"] + scorecols,
            )
            for b in range(1, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                tmp = get_metrics_test(
                    num_classes=n_classes,
                    coverages=coverages,
                    true=db["true"].values,
                    selected=db["bands"].values,
                    y_scores=db[scorecols].values,
                    y_preds=db["preds"].values,
                    meta=meta,
                    trial_num=trial,
                    dataset=filename,
                    arch=arch,
                    most_common_class=most_common_class,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
    elif baseline in ["sele", "cn", "reg"]:
        set_seed(seed)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        deactiv = True
        if meta in ["sele", "reg"]:
            deactiv = False
        if dict_tabular[filename]:
            calibloader = torch.utils.data.DataLoader(
                calibration_set,
                batch_size=bsize,
                shuffle=False,
            )
            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=bsize,
                shuffle=False,
            )
            model = tabular_model(arch, x_cont, cat_dim, n_classes, meta, d_arch)
        else:
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=int(bsize / 8), shuffle=False
            )
            calibloader = torch.utils.data.DataLoader(
                calibration_set, batch_size=int(bsize / 8), shuffle=False
            )
            model = image_model(arch, n_classes, meta, d_arch)
        path_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
            model_fold, filename, arch, meta, int(trial), epochs
        )
        path_model_unc = "{}/{}/unc_{}_{}_{}_tr{}_ep{}.pt".format(
            model_fold, filename, arch, meta, "xx", int(trial), epochs
        )
        if arch == "VGG":
            if len([el for el in model.named_modules()]) > 50:
                model_unc = buildConfidNet("VGG16bn", model, {}, deactivate=deactiv)
            else:
                model_unc = buildConfidNet("VGG16", model, {}, deactivate=deactiv)
        else:
            model_unc = buildConfidNet(arch, model, {}, deactivate=deactiv)
        if os.path.exists(path_model):
            model.load_state_dict(torch.load(path_model, map_location=device))
            model.to(device)
            model_unc.load_state_dict(torch.load(path_model_unc, map_location=device))
            model_unc.to(device)
        set_seed(seed)
        scores = predict_proba(model, device, testloader, meta)
        preds = np.argmax(scores, axis=1)
        selected = qband(model_unc, device, testloader, calibloader, meta, coverages)
        results = get_metrics_test(
            num_classes=n_classes,
            coverages=coverages,
            true=y_test,
            selected=selected,
            y_scores=scores,
            y_preds=preds,
            meta=meta,
            trial_num=trial,
            dataset=filename,
            arch=arch,
            most_common_class=most_common_class,
        )
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, int(trial), filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        results["b_iter"] = 0
        if os.path.exists("{}/{}".format(result_fold, filename)) == False:
            os.mkdir("{}/{}".format(result_fold, filename))
        scorecols = ["classscore_{}".format(cl) for cl in range(scores.shape[1])]
        yy = pd.DataFrame(
            np.c_[y_test, preds, selected, scores],
            columns=["true", "preds", "bands"] + scorecols,
        )
        for b in range(1, boot_iter + 1):
            if b == 0:
                db = yy.copy()
            else:
                db = yy.sample(len(yy), random_state=b, replace=True)
            tmp = get_metrics_test(
                num_classes=n_classes,
                coverages=coverages,
                true=db["true"].values,
                selected=db["bands"].values,
                y_scores=db[scorecols].values,
                y_preds=db["preds"].values,
                meta=meta,
                trial_num=trial,
                dataset=filename,
                arch=arch,
                most_common_class=most_common_class,
            )
            tmp["b_iter"] = b
            results = pd.concat([results, tmp], axis=0)
            results.to_csv(results_filename, index=False)


# @track_emissions(measure_power_secs=300)
def main(dataset, baseline, device, setup="best", verbose=False, model_fold="models"):
    testing_torch(
        dataset, baseline, device, setting=setup, verb=verbose, model_fold=model_fold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    use_cuda = torch.cuda.is_available()
    # Random seed
    set_seed(42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--setup", type=str, default="best")
    parser.add_argument("--verb", type=str, default="False")
    parser.add_argument("--mfold", type=str, default="models")

    # Parse the argument
    args = parser.parse_args()
    device = args.device
    filenames = args.data
    verb = eval(str(args.verb))
    modelfold = args.mfold
    for filename in filenames:
        if args.base == "all":
            if dict_binary[filename]:
                all_baselines = [
                    "plugin",
                    "pluginauc",
                    "cn",
                    "sele",
                    "reg",
                    "dg",
                    "sat",
                    "sat_te",
                    "sat_sr",
                    "sat_te_sr",
                    "scross",
                    "aucross",
                    "selnet",
                    "selnet_te",
                    "selnet_sr",
                    "selnet_te_sr",
                ]
            else:
                all_baselines = [
                    "plugin",
                    "cn",
                    "sele",
                    "reg",
                    "dg",
                    "sat",
                    "sat_te",
                    "sat_sr",
                    "sat_te_sr",
                    "scross",
                    "selnet",
                    "selnet_te",
                    "selnet_sr",
                    "selnet_te_sr",
                ]
            for base in all_baselines:
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        elif args.base == "notrain":
            baselines = [
                "plugin",
                "pluginauc",
                "dg",
                "sat",
                "sat_te",
                "sat_sr",
                "sat_te_sr",
            ]
            for base in baselines:
                print("Running {}".format(base))
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        elif args.base == "ensembles":
            baselines = ["ensemble", "ensemble_sr", "scross_ens"]
            for base in baselines:
                print("Running {}".format(base))
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        elif args.base == "unc":
            baselines = ["cn", "sele", "reg"]
            for base in baselines:
                print("Running {}".format(base))
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        elif args.base == "no_te":
            baselines = [
                "plugin",
                "pluginauc",
                "dg",
                "sat",
                "sat_sr",
                "scross",
                "aucross",
                "selnet",
                "selnet_sr",
            ]
            for base in baselines:
                print("Running {}".format(base))
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        elif args.base == "train":
            baselines = [
                "selnet",
                "selnet_te",
                "selnet_sr",
                "selnet_te_sr",
                "scross",
                "aucross",
            ]
            for base in baselines:
                print("Running {}".format(base))
                main(
                    filename,
                    base,
                    device,
                    setup=args.setup,
                    verbose=verb,
                    model_fold=modelfold,
                )
        else:
            main(
                filename,
                args.base,
                device,
                setup=args.setup,
                verbose=verb,
                model_fold=modelfold,
            )
