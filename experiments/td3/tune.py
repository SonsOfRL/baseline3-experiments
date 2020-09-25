import yaml
import optuna
import argparse
from multiprocessing import Process

from run import run
from optuna.samplers import TPESampler


def read_yaml(name):

    with open(name) as file_obj:
        tunes = yaml.load(file_obj, Loader=yaml.FullLoader)

    return tunes


def name_check(name):
    return name.replace("-", "_")


def get_defaults(args_path):
    default_args = {}
    argparse_dict = read_yaml(args_path)
    for arg_name, kwargs in argparse_dict.items():
        default_args[name_check(arg_name)] = kwargs["default"]
    return default_args


def get_hypers(trial, args_path, tune_path):
    hyper_params = get_defaults(args_path)
    for arg_name, suggest in read_yaml(tune_path).items():
        hyper_params[name_check(arg_name)] = getattr(
            trial, suggest["type"])(name_check(arg_name), **suggest["kwargs"])
    return hyper_params


def objective(trial, args_path, tune_path, n_same_runs):
    hypers = get_hypers(trial, args_path, tune_path)
    logs_dir = "logs/trial_{:3}".format(trial.number)
    hypers["tensorboard_log"] = logs_dir
    return sum(run(**hypers) for i in range(n_same_runs))


def run_study(n_trials, storage, args_path, tune_path, n_same_runs):
    study = optuna.create_study(
        storage=storage, study_name="td3", load_if_exists=True,
        direction="maximize", sampler=TPESampler())
    study.optimize(lambda trial: objective(
        trial, args_path, tune_path, n_same_runs), n_trials=n_trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", help="Number of trials for tuning",
                        default=10, type=int, required=False)
    parser.add_argument("--args_path", help="Default argmuents path (also used for argparse)",
                        default="args.yaml", type=str, required=False)
    parser.add_argument("--tune_path", help="Tuning arguments yaml file path",
                        default="tune.yaml", type=str, required=False)
    parser.add_argument("--storage", help="Storage string for optuna",
                        default="sqlite:///td3.db", type=str, required=False)
    parser.add_argument("--n_same_runs", help="Number of calls with the same set of paremters",
                        default=1, type=int, required=False)
    run_study(**vars(parser.parse_args()))
