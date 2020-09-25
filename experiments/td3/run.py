import gym
import numpy as np
import argparse
import yaml
import os
import subprocess

import stable_baselines3 as sb3
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise,
                                            VectorizedActionNoise)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from lab_sb3.callbacks import LoggerCallback


def run(train_freq,
        gradient_steps,
        batch_size,
        envname,
        n_envs,
        log_interval,
        learning_rate,
        buffer_size,
        tau,
        gamma,
        target_policy_noise,
        target_noise_clip,
        learning_starts,
        total_timesteps,
        policy_kwargs,
        action_noise_mean,
        action_noise_sigma,
        noise_type,
        eval_freq,
        n_eval_episodes,
        verbose=True,
        tensorboard_log="logs/"):

    # Normalize with multi environments
    eval_freq = max(eval_freq // n_envs, 1)
    buffer_size = max(buffer_size //n_envs, 1)

    all_args = locals()

    path = "/" + os.path.join(*sb3.__file__.split("/")[:-2])
    commit_num = subprocess.check_output(
        ["git", "describe", "--always"], cwd=path).strip().decode()

    env = gym.make(envname)
    vecenv = make_vec_env(envname, vec_env_cls=SubprocVecEnv, n_envs=n_envs)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    if noise_type == "OU":
        base_noise_class = OrnsteinUhlenbeckActionNoise
    elif noise_type == "Normal":
        base_noise_class = NormalActionNoise
    base_noise = base_noise_class(
        mean=np.ones(n_actions) * action_noise_mean,
        sigma=action_noise_sigma * np.ones(n_actions))
    action_noise = VectorizedActionNoise(base_noise, vecenv.num_envs)

    # Callbacks
    loggercallback = LoggerCallback(
        "json",
        [("arguments", all_args),
         ("git", commit_num)])
    evalcallback = EvalCallback(
        make_vec_env(envname, vec_env_cls=SubprocVecEnv),
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq)

    # Initiate the model and start learning
    model = TD3("MlpPolicy",
                vecenv,
                action_noise=action_noise,
                batch_size=batch_size,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                learning_starts=learning_starts,
                n_episodes_rollout=-1,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                tau=tau,
                gamma=gamma,
                create_eval_env=True,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
                verbose=verbose,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                device="cuda")
    model.learn(total_timesteps=total_timesteps,
                log_interval=log_interval,
                callback=[loggercallback, evalcallback],
                tb_log_name=envname,)
    model.env.close()
    evalcallback.eval_env.close()

    return evalcallback.best_mean_reward


def get_argparse_values(kwargs):
    for key, value in kwargs.items():
        if value == "int":
            kwargs[key] = int
        if value == "float":
            kwargs[key] = float
        if value == "str":
            kwargs[key] = str

    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    with open("args.yaml") as file_obj:
        parse_args = yaml.load(file_obj, Loader=yaml.FullLoader)
    for arg_name, args in parse_args.items():
        parser.add_argument("--" + arg_name, **get_argparse_values(args))

    parser.add_argument("--verbose", help="print out the logs",
                        action="store_false", required=False)
    parser.add_argument("--tensorboard_log", help="Logging directory for tensorboard objects",
                        default="logs/", type=str, required=False)

    kwargs = vars(parser.parse_args())
    run(**kwargs)
