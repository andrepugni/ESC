from classes.attributes import *
from classes.utils import *
import optuna
from optuna.samplers import TPESampler
from optuna.integration import BoTorchSampler
import argparse
from classes.parameters import *


def main(
    training_set,
    calibration_set,
    holdout_set,
    dataset,
    meta,
    device,
    architecture,
    sample_method,
    trials=20,
    iterations=200,
    tab_dictionary=dict_tabular,
    root_models="models",
    root_results="results",
    root_studies="studies",
    sub=False,
    batchsize=128,
    verb=False,
):
    """
    Main function to run the tuning
    :param training_set: TabularDataset or ImgFolder
        the training set to use
    :param calibration_set: TabularDataset or ImgFolder
        the calibration set to use
    :param holdout_set: TabularDataset or ImgFolder
        the holdout set to use
    :param dataset: str
        the name of the dataset
    :param meta: str
        the meta-learning algorithm to use
    :param device: str
        the device to use
    :param architecture: str
        the architecture to use
    :param sample_method: optuna sampler
        the sampler to use
    :param trials: int
        the number of trials to run
    :param iterations: int
        the number of epochs for training
    :param tab_dictionary: dict
        the dictionary containing the information about the datasets
    :param root_models: str
        the root path to store the models
    :param root_results: str
        the root path to store the results
    :param root_studies: str
        the root path to store the studies
    :param sub: bool
        whether to use the sub-sampling or not
    :param batchsize: int
        the batch size to use
    :param verb:
        whether to print the results or not
    :return:
    """
    tabular = tab_dictionary[dataset]
    seed = 42
    set_seed(seed)
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
    name_study = "study_{}_{}_{}_{}_{}".format(meta, dataset, arch, trials, iterations)
    if os.path.exists("{}/{}".format(root_models, dataset)) == False:
        os.mkdir("{}/{}".format(root_models, dataset))
    if os.path.exists("{}/{}".format(root_studies, dataset)) == False:
        try:
            os.mkdir("{}/{}".format(root_studies, dataset))
        except FileExistsError:
            print("Folder already created")
    study = optuna.create_study(
        study_name=name_study,
        storage="sqlite:///{}/{}/trials_{}_{}_sample{}.db".format(
            root_studies, dataset, dataset, meta, sub
        ),
        directions=[
            "minimize",
            "minimize",
            "minimize",
            "minimize",
            "minimize",
            "minimize",
        ],
        sampler=sample_method,
        load_if_exists=True,
    )
    if tabular:
        print("Using tabular data")
        study.optimize(
            lambda trial: objective(
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
                bsize=batchsize,
                verb=verb,
            ),
            n_trials=trials,
        )
    else:
        print("Using image data")
        study.enqueue_trial(dict_default_trials[dataset][meta]["params"])
        study.optimize(
            lambda trial: objective(
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
                bsize=batchsize,
                verb=verb,
            ),
            n_trials=trials,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    use_cuda = torch.cuda.is_available()
    # Random seed
    set_seed(42)
    sampler = BoTorchSampler(
        n_startup_trials=10, independent_sampler=TPESampler(seed=42), seed=42
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=20)
    # parser.add_argument("--iter", type=int, default=150)
    parser.add_argument("--mach", type=str, default="fastcep")
    parser.add_argument("--sub", default=False)
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--arch", type=str, default="DNN")
    # Parse the argument
    args = parser.parse_args()
    device = args.device
    parallel = False
    parallel_selnet = False
    verbose = eval(str(args.verbose))
    args.sub = eval(args.sub)
    if args.mach == "fastcep":
        root_path_data = "data/clean"
        root_path_models = "models"
        root_path_results = "results"
        root_path_studies = "studies"
    elif args.mach == "cudos01":
        root_path_data = "/cw/dtaijupiter/NoCsBack/dtai/andrea/esc/data/clean"
        root_path_models = "/cw/dtaijupiter/NoCsBack/dtai/andrea/esc/models"
        root_path_results = "/cw/dtaijupiter/NoCsBack/dtai/andrea/esc/results"
        root_path_studies = "/cw/dtaijupiter/NoCsBack/dtai/andrea/esc/studies"
    elif args.mach == "clusterben":
        root_path_data = "/home/users/pugnana/projects/esc/data/clean"
        root_path_models = "/home/users/pugnana/projects/esc/models"
        root_path_results = "/home/users/pugnana/projects/esc/results"
        root_path_studies = "/home/users/pugnana/projects/esc/studies"
        if "selnet" in args.meta:
            parallel_selnet = False
        elif "selnet_meta" in args.meta:
            parallel_selnet = False
        else:
            parallel = False
    if args.data[0] == "images":
        filenames = [f for f in os.listdir(root_path_data) if f in image_data]
    else:
        filenames = args.data
        for filename in filenames:
            if filename in image_data:
                tab = False
            else:
                tab = True
            epochs = dict_epochs[filename]
            #### CHANGE PATH DEPENDING ON WHERE YOU STORE YOUR DATA AND WHICH SERVER IS CURRENTLY BEING USED
            if tab:
                training_set = TabularDataset(
                    filename,
                    root=root_path_data,
                    atts=dict_atts[filename],
                    set="train",
                    device=device,
                )
                holdout_set = TabularDataset(
                    filename,
                    root=root_path_data,
                    atts=dict_atts[filename],
                    set="hold",
                    device=device,
                )
                calibration_set = TabularDataset(
                    filename,
                    root=root_path_data,
                    atts=dict_atts[filename],
                    set="cal",
                    device=device,
                )
            else:
                transform_train, transform_test, input_size = transformation(
                    filename, also_inp_size=True
                )
                training_set = ImgFolder(
                    "{}/{}/train".format(root_path_data, filename),
                    transform=transform_train,
                )
                holdout_set = ImgFolder(
                    "{}/{}/hold".format(root_path_data, filename),
                    transform=transform_test,
                )
                calibration_set = ImgFolder(
                    "{}/{}/cal".format(root_path_data, filename),
                    transform=transform_test,
                )
            if args.arch == "lgbm":
                main(
                    training_set,
                    calibration_set,
                    holdout_set,
                    filename,
                    args.meta[0],
                    args.device,
                    "lgbm",
                    sampler,
                    args.trials,
                    iterations=epochs,
                    root_models=root_path_models,
                    root_studies=root_path_studies,
                    root_results=root_path_results,
                    sub=args.sub,
                    batchsize=dict_batch[filename],
                    verb=verbose,
                )
            else:
                main(
                    training_set,
                    calibration_set,
                    holdout_set,
                    filename,
                    args.meta[0],
                    args.device,
                    dict_arch[filename],
                    sampler,
                    args.trials,
                    iterations=epochs,
                    root_models=root_path_models,
                    root_studies=root_path_studies,
                    root_results=root_path_results,
                    sub=args.sub,
                    batchsize=dict_batch[filename],
                    verb=verbose,
                )
