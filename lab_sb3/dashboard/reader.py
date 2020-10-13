import os
import argparse
import warnings
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

import plotly.graph_objects as go


def read_logs(logs_dir):
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError("Argument <logs_dir> is not a valid directory")

    experiment_data = {}
    for trial_name in tqdm(os.listdir(logs_dir), ncols=80, desc="Trials "):
        trial_path = os.path.join(logs_dir, trial_name)

        trial_data = {"data": []}
        for run_name in os.listdir(trial_path):
            run_path = os.path.join(trial_path, run_name)
            if "progress.json" in os.listdir(run_path):
                with open(os.path.join(run_path, "progress.json")) as jsonf:
                    lines = jsonf.readlines()
                    data = list(map(json.loads, lines))

                    if "hyperparameters" in trial_data.keys():
                        if trial_data["hyperparameters"] != data[0]["arguments"]:
                            warnings.warn("Run {} has a different set of hyperparameters at"
                                          "trial {} and it is PASSED!".format(
                                              run_name, trial_name))
                            continue

                    trial_data["hyperparameters"] = data[0].pop("arguments")
                    trial_data["commit"] = data[0].pop("git")
                    trial_data["log_path"] = data[0].pop("log_path")

                    dataframe = pd.DataFrame(data)
                    trial_data["data"].append(dataframe)

            else:
                warnings.warn("Missing progress.json file for trial: {} and run: {}".format(
                    trial_name, run_name))
        if len(experiment_data) > 5:
            break
        experiment_data[trial_name] = trial_data
    return experiment_data


def save_experiment_data(data):
    with open("exp_data.b", "wb") as binf:
        pickle.dump(data, binf)

def load_experiment_data():
    if not os.path.exists("exp_data.b"):
        raise FileNotFoundError("No exp data binary file")
    
    with open("exp_data.b", "rb") as binf:
        return pickle.load(binf)

# def get_abstract_data(dataframes):
#     for df in dataframes:
#         if "rollout/ep_rew_mean" in df.columns:



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dashboard")
    parser.add_argument(
        "--logs-dir", help="Directory of logs", type=str, action="store", required=True)

    cl_args = parser.parse_args()
    read_logs(cl_args.logs_dir)
