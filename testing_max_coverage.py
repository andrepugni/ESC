from classes.risk_control import *
from testing import *


def calibrate_aucross(ys, z, quantiles):
    """

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


def testing_torch_coverage(
    filename,
    baseline,
    device,
    coverages=[0.99, 0.85, 0.7],
    setting="best",
    tabular_dictionary=dict_tabular,
    structure_dictionary=dict_arch,
    epochs_dictionary=dict_epochs,
    baseline_dictionary=dict_baseline,
    batch_dictionary=dict_batch,
    data_fold="data/clean",
    model_fold="best_models",
    result_fold="test_results_coverage",
    seed=42,
    verb=False,
    boot_iter=100,
):
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
        # d_arch = check_architecture(d_arch, filename, meta)
        setting_string = ""
        # rstars = {}
        # for coverage in coverages:
        #     rstar = 1-bm[(bm['desired_coverage'] == coverage) & (bm['meta'] == meta)]["accuracy"].iloc[0]
        #     rstars[coverage] = rstar

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
    dict_error = {0: "e", 1: "e/2", 2: "e/5", 3: "e/10"}
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
        n_classes = len(training_set.classes)
        counts_mostcommon = np.max(
            np.unique(training_set.targets, return_counts=True)[1]
        )
        perc_train = counts_mostcommon / len(training_set)
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
        n_classes = len(training_set.classes)
        counts_mostcommon = np.max(
            np.unique(training_set.targets, return_counts=True)[1]
        )
        perc_train = counts_mostcommon / len(training_set)
    rstars = [
        1 - perc_train,
        (1 - perc_train) / 2,
        (1 - perc_train) / 5,
        (1 - perc_train) / 10,
    ]
    y_test = np.array(test_set.targets)
    y_cal = np.array(calibration_set.targets)
    y_train = np.array(training_set.targets)
    p_train = np.sum(training_set.targets) / len(training_set)
    set_seed(seed)
    results = pd.DataFrame()
    delta = 0.001
    if baseline in [
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
        if setting == "best":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, int(trial), filename
            )
        elif setting == "default":
            results_filename = "{}/{}/TESTING_results_{}_{}_{}_{}.csv".format(
                result_fold, filename, baseline, arch, setting_string, filename
            )
        for i, cov in enumerate([0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]):
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
                    )
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
            y_hat = predict_proba(selnet, device, testloader, meta)
            scores.append(y_hat)
            preds_test = np.argmax(y_hat, axis=1)
            confs_test = predict_conf(selnet, device, testloader, baseline)
            confs_cal = predict_conf(selnet, device, calibloader, baseline)
            preds_cal = predict(selnet, device, calibloader, baseline)
            residuals_cal = np.where(y_cal - preds_cal == 0, 0, 1)
            residuals_test = np.where(y_test - preds_test == 0, 0, 1)
            yy = pd.DataFrame(
                np.c_[residuals_test, confs_test], columns=["res_test", "conf_test"]
            )
            for b in range(0, boot_iter + 1):
                if b == 0:
                    db = yy.copy()
                else:
                    db = yy.sample(len(yy), random_state=b, replace=True)
                for j, rstar in enumerate(rstars):
                    rs = risk_control()
                    theta, bound, testrisk, testcov = rs.bound(
                        rstar,
                        delta,
                        confs_cal,
                        residuals_cal,
                        db["conf_test"],
                        db["res_test"],
                    )
                    tmp = pd.DataFrame()
                    tmp["theta"] = [theta]
                    tmp["bound"] = bound
                    tmp["error_rate"] = testrisk
                    tmp["test_coverage"] = testcov
                    tmp["rstar"] = rstar
                    tmp["baseline"] = baseline
                    tmp["b_iter"] = b
                    tmp["rstar_string"] = dict_error[j]
                    tmp["desired_coverage"] = cov
                    tmp["dataset"] = filename
                    results = pd.concat([results, tmp], axis=0)
                    results.to_csv(
                        "{}/{}/TESTING_SGR_alg_{}_{}_{}.csv".format(
                            result_fold, filename, baseline, setting, filename
                        ),
                        index=False,
                    )
    else:
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
                modelfinal = tabular_model(
                    arch, x_cont, cat_dim, n_classes, meta, d_arch
                )
            else:
                print("Dataset: {}, Image, n classes: {}".format(filename, n_classes))
                trainloader = torch.utils.data.DataLoader(
                    trainingcal_set,
                    batch_size=dict_batch[filename],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=g_seed,
                )
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
                preds = []
                # for each fold
                if dict_tabular[filename]:
                    path_final_model = "{}/{}/{}_cross_{}final_ep{}.pt".format(
                        model_fold, filename, arch, setting_string, epochs
                    )
                else:
                    path_final_model = "{}/{}/{}_scross_final_ep{}.pt".format(
                        model_fold, filename, arch, epochs
                    )
                print(path_final_model)
                if os.path.exists(path_final_model):
                    modelfinal.to(device)
                    modelfinal.load_state_dict(
                        torch.load(path_final_model, map_location=device)
                    )
                else:
                    raise FileNotFoundError(
                        "Not yet trained the final model. Run testing.py before running testing_max_coverage.py"
                    )
                for k, split in enumerate(splits):
                    train_idx, test_idx = split[0], split[1]
                    torch.manual_seed(seed)
                    set_seed(seed)
                    g_seed = torch.Generator()
                    g_seed.manual_seed(seed)
                    if dict_tabular[filename]:
                        training_set = TabularDataset(
                            filename,
                            atts=dict_atts[filename],
                            set="train",
                            device=device,
                        )
                        calibration_set = TabularDataset(
                            filename, atts=dict_atts[filename], set="cal", device=device
                        )
                        trainingcal_set = torch.utils.data.ConcatDataset(
                            [training_set, calibration_set]
                        )
                        hold_ = torch.utils.data.Subset(
                            trainingcal_set, indices=test_idx
                        )
                        holdfold_dl = torch.utils.data.DataLoader(
                            hold_,
                            batch_size=int(dict_batch[filename] / 4),
                            shuffle=False,
                        )
                        modelfold = tabular_model(
                            arch, x_cont, cat_dim, n_classes, meta, d_arch
                        )
                        path_fold_model = "{}/{}/{}_cross_{}fold{}out{}_ep{}.pt".format(
                            model_fold,
                            filename,
                            setting_string,
                            arch,
                            k + 1,
                            cv,
                            epochs,
                        )
                    else:
                        modelfold = image_model(arch, n_classes, meta, d_arch)
                        path_fold_model = (
                            "{}/{}/best_scross_{}fold{}out{}_ep{}.pt".format(
                                model_fold, filename, arch, k + 1, cv, epochs
                            )
                        )
                    print(path_fold_model)
                    if os.path.exists(path_fold_model):
                        modelfold.to(device)
                        modelfold.load_state_dict(
                            torch.load(path_fold_model, map_location=device)
                        )
                    else:
                        raise FileNotFoundError(
                            "Please run testing.py before this file."
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
                        hold_ = torch.utils.data.Subset(
                            trainingcal_set, indices=test_idx
                        )
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
                            predict_proba(
                                modelfold, device, holdfold_dl, meta="plugin"
                            ),
                            axis=1,
                        ).reshape(-1, 1)
                    pred = predict(modelfinal, device, holdfold_dl, meta="plugin")
                    preds.append(pred.reshape(-1, 1))
                    z.append(conf)
                    t_id = test_idx.reshape(-1, 1)
                    ys.append(y_traincal[test_idx].reshape(-1, 1))
                    idx.append(t_id)
                confs_cal = np.vstack(z).flatten()
                preds_cal = np.vstack(preds).flatten()
                ys = np.vstack(ys).flatten()
                if baseline == "scross":
                    sub_confs_1, sub_confs_2 = train_test_split(
                        confs_cal, test_size=0.5, random_state=42
                    )
                    tau = 1 / np.sqrt(2)
                    thetas = [
                        (
                            tau * np.quantile(confs_cal, q)
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
                            tau * np.quantile(confs_cal, q)
                            + (1 - tau)
                            * (
                                0.5 * np.quantile(sub_confs_1, q)
                                + 0.5 * np.quantile(sub_confs_2, q)
                            )
                        )
                        for q in quantiles
                    }
                    scores = predict_proba(modelfinal, device, testloader, meta)
                    preds_test = np.argmax(scores, axis=1)
                    confs_test = np.max(scores, axis=1)
                else:
                    m = len(quantiles)
                    thetas, dict_q = calibrate_aucross(ys, confs_cal, quantiles)
                    scores = predict_proba(modelfinal, device, testloader, meta)
                    preds_test = np.argmax(scores, axis=1)
                    selected = np.zeros(len(scores[:, 1])) + m
                    for i, t in enumerate(reversed(thetas)):
                        t1, t2 = t[0], t[1]
                        selected[((t1 <= scores[:, 1]) & (scores[:, 1] <= t2))] = (
                            m - i - 1
                        )
                residuals_cal = np.where(ys - preds_cal == 0, 0, 1)
                residuals_test = np.where(y_test - preds_test == 0, 0, 1)
        else:
            set_seed(seed)
            g_seed = torch.Generator()
            g_seed.manual_seed(seed)
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
            if setting == "best":
                path_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                    model_fold, filename, arch, meta, int(trial), epochs
                )
            elif setting == "default":
                path_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                    model_fold, filename, arch, meta, setting_string, epochs
                )
                print(path_model)
            if baseline in ["sele", "reg", "cn"]:
                path_model_unc = "{}/{}/unc_{}_{}_{}_tr{}_ep{}.pt".format(
                    model_fold, filename, arch, meta, "xx", int(trial), epochs
                )
                deactiv = False
                if baseline == "cn":
                    deactiv = True
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
                    model_unc = buildConfidNet(arch, model, {}, deactivate=deactiv)
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model, map_location=device))
                    model.to(device)
                    model_unc.load_state_dict(
                        torch.load(path_model_unc, map_location=device)
                    )
                    model_unc.to(device)
                scores = predict_proba(model, device, testloader, meta)
                preds_test = np.argmax(scores, axis=1)
                confs_test = predict_conf(model_unc, device, testloader, baseline)
                preds_cal = predict(model, device, calibloader, meta)
                confs_cal = predict_conf(model_unc, device, calibloader, baseline)
            elif baseline in ["ensemble", "ensemble_sr"]:
                seeds = [42, 73, 123, 456, 789, 999, 1111, 2222, 3333, 4444]
                dict_seeds = {s: None for s in seeds}
                y_hats = []
                y_hats_cal = []
                for i, seed_value in enumerate(seeds):
                    set_seed(seed_value)
                    g_seed = torch.Generator()
                    g_seed.manual_seed(seed_value)
                    # here we build the model for each coverage
                    if dict_tabular[filename]:
                        model = tabular_model(
                            arch, x_cont, cat_dim, n_classes, meta, d_arch
                        )
                    else:
                        model = image_model(arch, n_classes, meta, d_arch)
                    if setting == "best":
                        if (
                            os.path.exists("{}/{}".format(model_fold, filename))
                            == False
                        ):
                            os.mkdir("{}/{}".format(model_fold, filename))
                        if seed_value == 42:
                            path_seed_model = "{}/{}/{}_{}_xx_tr{}_ep{}.pt".format(
                                model_fold, filename, arch, meta, int(trial), epochs
                            )
                        else:
                            path_seed_model = (
                                "{}/{}/{}_{}_xx_tr{}_ep{}_seed{}.pt".format(
                                    model_fold,
                                    filename,
                                    arch,
                                    meta,
                                    int(trial),
                                    epochs,
                                    seed_value,
                                )
                            )
                    elif setting == "default":
                        raise NotImplementedError("Setting Not Yet Implemented")
                    if os.path.exists(path_seed_model):
                        print("here we go")
                        model.to(device)
                        model.load_state_dict(
                            torch.load(path_seed_model, map_location=device)
                        )
                        model.eval()

                        # dict_seeds[seed_value] = copy.deepcopy(model)
                    y_hats.append(predict_proba(model, device, testloader, meta))
                    y_hats_cal.append(predict_proba(model, device, calibloader, meta))
                set_seed(42)
                scores = np.mean(y_hats, axis=0)
                scores_cal = np.mean(y_hats_cal, axis=0)
                preds_test = np.argmax(scores, axis=1)
                preds_cal = np.argmax(scores_cal, axis=1)
                if baseline == "ensemble":
                    confs_test = -get_confidence_ensemble(y_hats, scores)
                    confs_cal = -get_confidence_ensemble(y_hats_cal, scores_cal)
                elif baseline == "ensemble_sr":
                    confs_test = np.max(scores, axis=1)
                    confs_cal = np.max(scores_cal, axis=1)
                    print(confs_test.max())
                    print(confs_cal.max())
                    print(confs_test.min())
                    print(confs_cal.min())
            else:
                if os.path.exists(path_model):
                    print("there we go")
                    model.to(device)
                    model.load_state_dict(torch.load(path_model, map_location=device))
                    scores = predict_proba(model, device, testloader, meta)
                    preds_test = np.argmax(scores, axis=1)
                    confs_test = predict_conf(model, device, testloader, baseline)
                    preds_cal = predict(model, device, calibloader, meta)
                    confs_cal = predict_conf(model, device, calibloader, baseline)
            if baseline in ["sat", "sat_te", "dg", "reg", "sele"]:
                confs_test = 1 - confs_test
                confs_cal = 1 - confs_cal
            residuals_cal = np.where(y_cal - preds_cal == 0, 0, 1)
            residuals_test = np.where(y_test - preds_test == 0, 0, 1)
        delta = 0.001
        results = pd.DataFrame()
        yy = pd.DataFrame(
            np.c_[residuals_test, confs_test], columns=["res_test", "conf_test"]
        )
        for b in range(0, boot_iter + 1):
            if b == 0:
                db = yy.copy()
            else:
                db = yy.sample(len(yy), random_state=b, replace=True)
            for i, rstar in enumerate(rstars):
                rs = risk_control()
                theta, bound, testrisk, testcov = rs.bound(
                    rstar,
                    delta,
                    confs_cal,
                    residuals_cal,
                    db["conf_test"],
                    db["res_test"],
                )
                tmp = pd.DataFrame()
                tmp["theta"] = [theta]
                tmp["bound"] = bound
                tmp["error_rate"] = testrisk
                tmp["test_coverage"] = testcov
                tmp["rstar"] = rstar
                tmp["baseline"] = baseline
                tmp["b_iter"] = b
                tmp["rstar_string"] = dict_error[i]
                tmp["dataset"] = filename
                results = pd.concat([results, tmp], axis=0)
                results.to_csv(
                    "{}/{}/TESTING_SGR_alg_{}_{}_{}.csv".format(
                        result_fold, filename, baseline, setting, filename
                    ),
                    index=False,
                )


def main(dataset, baseline, device, setup="best", verbose=False, model_fold="models"):
    testing_torch_coverage(
        dataset, baseline, device, setting=setup, verb=verbose, model_fold=model_fold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    use_cuda = torch.cuda.is_available()
    # Random seed
    set_seed(42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--base", type=str, default="all")
    parser.add_argument("--setup", type=str, default="best")
    parser.add_argument("--verb", type=str, default="False")
    parser.add_argument("--mfold", type=str, default="best_models")

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
                    "ensemble",
                    "ensemble_sr",
                    "cn",
                    "reg",
                    "sele",
                ]
            else:
                all_baselines = [
                    "plugin",
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
                    "ensemble",
                    "ensemble_sr",
                    "cn",
                    "reg",
                    "sele",
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
        elif args.base == "v2":
            baselines = ["ensemble", "ensemble_sr", "cn", "reg", "sele"]
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
        elif args.base == "ens":
            baselines = [
                "ensemble",
                "ensemble_sr",
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
            baselines = ["cn", "reg", "sele"]
            for base in baselines:
                print("Running {}".format(base))
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
        else:
            main(
                filename,
                args.base,
                device,
                setup=args.setup,
                verbose=verb,
                model_fold=modelfold,
            )
