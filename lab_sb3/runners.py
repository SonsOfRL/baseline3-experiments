import argparse
import yaml
import os
from abc import ABC, abstractmethod
from multiprocessing import Process
import warnings
import subprocess

import stable_baselines3 as sb3
import optuna
from optuna.samplers import (TPESampler,
                             GridSampler,
                             CmaEsSampler)

class BaseExperimentRunner(ABC):
    def str_to_fn(self, kwargs):
        for key, value in kwargs.items():
            if value == "int":
                kwargs[key] = int
            if value == "float":
                kwargs[key] = float
            if value == "str":
                kwargs[key] = str
            if value == "bool":
                kwargs[key] = bool

        return kwargs

    def read_yaml(self, path):
        with open(path) as file_obj:
            yaml_dict = yaml.load(file_obj, Loader=yaml.FullLoader)

        return yaml_dict

    def name_check(self, name):
        return name.replace("-", "_")

    def additional_args(self):
        raise NotImplementedError

    @staticmethod
    def get_sb3_commit_number():
        path = "/" + os.path.join(*sb3.__file__.split("/")[:-2])
        return subprocess.check_output(
            ["git", "describe", "--always"], cwd=path).strip().decode()


class ExperimentRunner(BaseExperimentRunner):

    args_name = "args.yaml"
    tune_name = "tune.yaml"
    log_name = "logs"
    sampler_map = {
        "TPE": TPESampler,
        "GRID": GridSampler,
        "CMA": CmaEsSampler,
    }

    def __init__(self, study_name="optuna"):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.study_name = study_name
        self.setup()

    def setup(self):
        self.default_argparse()
        self.additional_args()
        self.cl_args = vars(self.parser.parse_args())
        run_dir = self.cl_args["run_dir"]
        log_dir = os.path.join(run_dir, self.log_name)

        if not os.path.isdir(run_dir):
            raise NotADirectoryError(
                "Experiment run directory: {} is not valid!".format(run_dir))

        storage = "sqlite:///{}".format(
            os.path.join(run_dir, self.study_name + ".db"))
        sampler_class = self.sampler_map[self.cl_args["sampler"]]

        if not os.path.exists(storage):
            warnings.warn("Storage: {} already exist".format(storage))
        if not os.path.exists(log_dir):
            warnings.warn("Logging directory at: {} exists".format(log_dir))
        os.makedirs(log_dir, exist_ok=True)

        self.study = optuna.create_study(
            storage=storage, study_name=self.study_name, load_if_exists=True,
            direction="maximize", sampler=sampler_class())

    def start(self, run_fn):
        self.study.optimize(
            lambda trial: self.objective(run_fn, trial),
            n_trials=self.cl_args["n_trials"])

    def default_argparse(self):
        if not os.path.isfile(self.args_name):
            raise FileNotFoundError(
                "args.yaml could not found at {}".format(self.args_name))

        parse_args = self.read_yaml(self.args_name)
        for arg_name, args in parse_args.items():
            self.parser.add_argument("--" + arg_name, **self.str_to_fn(args))

    def _get_defaults(self):
        default_args = {}
        argparse_dict = self.read_yaml(self.args_name)
        for arg_name in argparse_dict.keys():
            arg_name = self.name_check(arg_name)
            default_args[arg_name] = self.cl_args[arg_name]
        return default_args

    def additional_args(self):
        self.parser.add_argument("--run_dir", help="Experiment path to log and read tune file",
                                 type=str, required=True)
        self.parser.add_argument("--verbose", help="print out the logs",
                                 action="store_false", required=False)
        self.parser.add_argument("--device", help="Pytorch device",
                                 default="cuda", action="store", type=str, required=False)
        self.parser.add_argument("--n_trials", help="Number of trials for tuning",
                                 default=100, type=int, required=False)
        self.parser.add_argument("--n_same_runs", help="Number of calls with the same set of paremters",
                                 default=3, type=int, required=False)
        self.parser.add_argument("--sampler", help="Optuna sampler method",
                                 default="TPE", type=str, choices=["TPE", "GRID", "CMA"], required=False)

    def _get_hypers(self, trial, tune_path):
        hyper_params = self._get_defaults()
        for arg_name, suggest in self.read_yaml(tune_path).items():
            hyper_params[self.name_check(arg_name)] = getattr(
                trial, suggest["type"])(self.name_check(arg_name), **suggest["kwargs"])
        return hyper_params

    def objective(self, run_fn, trial):
        n_same_runs = self.cl_args["n_same_runs"]
        run_dir = self.cl_args["run_dir"]

        tune_path = os.path.join(run_dir, self.tune_name)
        hypers = self._get_hypers(trial, tune_path)
        logs_dir = os.path.join(
            run_dir,
            "{}/trial_{:3}".format(self.log_name, trial.number).replace(" ", "0")
        )
        hypers["tensorboard_log"] = logs_dir
        hypers["device"] = self.cl_args["device"]
        hypers["verbose"] = self.cl_args["verbose"]
        return sum(run_fn(**hypers) for i in range(n_same_runs)) / n_same_runs
