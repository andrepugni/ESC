import os.path
from classes.attributes import *
from classes.utils import *
import argparse
from sklearn.model_selection import StratifiedKFold


def make_optimizer(d_opt, network):
    """
    Make optimizer
    :param d_opt:
    :param network:
    :return:
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
    Calibrate AUCross
    :param ys:
    :param z:
    :param quantiles:
    :return:
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


def check_architecture(d_arch, filename, meta):
    """
    correct errors in architectures due to roundings
    :param d_arch:
    :param filename:
    :param meta:
    :return:
    """
    if (meta == "dg") & (filename == "higgs"):
        d_arch["ffn_d_hidden"] = 85
    return d_arch


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
    result_fold="test_ood_results",
    boot_iter=100,
    seed=42,
    verb=False,
):
    """
    Testing function
    :param filename:
    :param baseline:
    :param device:
    :param coverages:
    :param setting:
    :param tabular_dictionary:
    :param structure_dictionary:
    :param epochs_dictionary:
    :param baseline_dictionary:
    :param batch_dictionary:
    :param data_fold:
    :param model_fold:
    :param result_fold:
    :param boot_iter:
    :param seed:
    :param verb:
    :return:
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
        bm = get_best_trial(meta, filename, arch)
        d_arch, d_loss, d_opt, trial, arch = get_best_params(meta, filename, arch)
        d_arch = check_architecture(d_arch, filename, meta)
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
        print("ood testing only for image datasets")
    else:
        transform_train, transform_test, input_size = transformation(
            filename, also_inp_size=True
        )
        training_set = ImgFolder(
            "{}/{}/train".format(data_fold, filename),
            transform=transform_train,
        )
        orig_test_set = ImgFolder(
            "{}/{}/test".format(data_fold, filename),
            transform=transform_test,
        )
        calibration_set = ImgFolder(
            "{}/{}/cal".format(data_fold, filename),
            transform=transform_test,
        )
        test_set = FkeData(
            size=int(len(orig_test_set)),
            image_size=(3, input_size, input_size),
            num_classes=get_num_classes(training_set),
            transform=transform_test,
        )
        if architecture == "vgg":
            d_arch["input_size"] = input_size
        n_classes = len(training_set.classes)
    y_test = np.array(
        [test_set.__getitem__(el)[1] for el in range(0, len(orig_test_set))]
    )
    p_train = np.sum(training_set.targets) / len(training_set)
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
                # we load the model for each fold
                modelfold = image_model(arch, n_classes, meta, d_arch)
                path_fold_model = "{}/{}/best_scross_{}fold{}out{}_ep{}.pt".format(
                    model_fold, filename, arch, k + 1, cv, epochs
                )
                path_interm = "{}/{}/{}_cross_{}fold{}out{}".format(
                    model_fold, filename, setting_string, arch, k + 1, cv
                )
                if os.path.exists(path_fold_model):
                    modelfold.to(device)
                    modelfold.load_state_dict(
                        torch.load(path_fold_model, map_location=device)
                    )
                dict_folds[k] = copy.deepcopy(modelfold)
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
            path_final_model = "{}/{}/{}_scross_{}final_ep{}.pt".format(
                model_fold, filename, arch, setting_string, epochs
            )
            if os.path.exists(path_final_model):
                modelfinal.to(device)
                modelfinal.load_state_dict(
                    torch.load(path_final_model, map_location=device)
                )
            else:
                raise FileNotFoundError(
                    "Final model not found.\nRun the testing.py file first."
                )
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
            p_train = np.sum(training_set.targets) / len(training_set)
            results = get_metrics_test_ood(
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
                perc_train=p_train,
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
                tmp = get_metrics_test_ood(
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
                    perc_train=p_train,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)
    elif baseline == "scross_ens":
        # we use cross-fitting hence no need for calibration dataset (this is used as an additional part of training set)
        cv = 5
        torch.manual_seed(seed)
        set_seed(42)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        print("Dataset: {}, Image, n classes: {}".format(filename, n_classes))
        calibloader = torch.utils.data.DataLoader(
            calibration_set,
            batch_size=bsize,
            shuffle=False,
        )
        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=int(bsize / 4), shuffle=False
        )
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
                modelfold = image_model(arch, n_classes, meta, d_arch)
                path_fold_model = "{}/{}/{}_scrossens_{}fold{}out{}_ep{}.pt".format(
                    model_fold, filename, setting_string, arch, k + 1, cv, epochs
                )
                if os.path.exists(path_fold_model):
                    modelfold.to(device)
                    modelfold.load_state_dict(
                        torch.load(path_fold_model, map_location=device)
                    )
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
            scores = np.mean(y_hats, axis=0)
            y_hats_cal = [
                predict_proba(dict_folds[k].to(device), device, calibloader, meta)
                for k in range(cv)
            ]
            y_hat_cal = np.mean(y_hats_cal, axis=0)
            preds = np.argmax(scores, axis=1)
            if baseline == "scross_ens":
                confs = np.max(scores, axis=1)
                confs_cal = np.max(y_hat_cal)
                thetas = []
                for i, cov in enumerate(coverages):
                    theta = np.quantile(confs, 1 - cov)
                    thetas.append(theta)
                selected = np.digitize(confs, sorted(thetas), right=False)
            selected = selected.astype(int)
            p_train = np.sum(training_set.targets) / len(training_set)
            results = get_metrics_test_ood(
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
                perc_train=p_train,
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
                tmp = get_metrics_test_ood(
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
                    perc_train=p_train,
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
                raise FileNotFoundError(
                    "The main experiment has not been run yet.\n Please run testing.py and then re run this experiment."
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
        results = get_metrics_test_ood(
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
            perc_train=p_train,
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
            tmp = get_metrics_test_ood(
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
                perc_train=p_train,
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
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f)
        cov_to_train = []
        results = pd.DataFrame()
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, int(trial), filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        for i, cov in enumerate(coverages):
            set_seed(seed)
            g_seed = torch.Generator()
            g_seed.manual_seed(seed)
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
                selnet.eval()
                set_seed(seed)
                y_hat = predict_proba(selnet, device, testloader, meta)
                scores.append(y_hat)
                preds = np.argmax(y_hat, axis=1)
                confs = predict_conf(selnet, device, testloader, baseline)
                confs_cal = predict_conf(selnet, device, calibloader, baseline)
                theta = np.quantile(confs_cal, 1 - cov)
                selected = np.where(confs > theta, 1, 0)
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
                    tmp = get_metrics_test_ood_selnet(
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
                        perc_train=p_train,
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
            p_train = np.sum(training_set.targets) / len(training_set)
            results = get_metrics_test_ood(
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
                perc_train=p_train,
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
                tmp = get_metrics_test_ood(
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
                    perc_train=p_train,
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
            selected = qband(
                model_unc, device, testloader, calibloader, meta, coverages
            )
            p_train = np.sum(training_set.targets) / len(training_set)
            results = get_metrics_test_ood(
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
                perc_train=p_train,
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
                tmp = get_metrics_test_ood(
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
                    perc_train=p_train,
                )
                tmp["b_iter"] = b
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(results_filename, index=False)


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
                    "scross_ens",
                    "selnet",
                    "selnet_te",
                    "selnet_sr",
                    "selnet_te_sr",
                    "ensemble",
                    "ensemble_sr",
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
                    "scross_ens",
                    "selnet",
                    "selnet_te",
                    "selnet_sr",
                    "selnet_te_sr",
                    "ensemble",
                    "ensemble_sr",
                ]
            for base in all_baselines:
                try:
                    main(
                        filename,
                        base,
                        device,
                        setup=args.setup,
                        verbose=verb,
                        model_fold=modelfold,
                    )
                except:
                    continue
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
        elif args.base == "unc":
            baselines = ["sele", "reg", "cn"]
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
        elif args.base == "crossens":
            baselines = ["scross_ens"]
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
            if dict_binary[filename]:
                baselines = [
                    "ensemble",
                    "ensemble_sr",
                    "scross",
                    "aucross",
                    "scross_ens",
                ]
            else:
                baselines = ["ensemble", "ensemble_sr", "scross", "scross_ens"]
        else:
            main(
                filename,
                args.base,
                device,
                setup=args.setup,
                verbose=verb,
                model_fold=modelfold,
            )
