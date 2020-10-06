import gym
import numpy as np
import argparse
import yaml
import os
import subprocess

import stable_baselines3 as sb3
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from lab_sb3.callbacks import LoggerCallback


def run(policy,
        envname,
        learning_rate,
        n_steps,
        batch_size,
        epochs,
        gamma,
        gae_lambda,
        ent_coef,
        vf_coef,
        max_grad_norm,
        normalize_advantage,
        policy_kwargs,
        n_eval_episodes,
        eval_freq,
        n_envs,
        n_stack,
        total_timesteps,
        log_interval,
        device="cuda",
        verbose=True,
        tensorboard_log="logs/"):

    # Normalize with multi environments
    seed = np.random.randint(1, 2**16)
    log_interval = log_interval // (n_steps * n_envs)
    all_args = locals()

    path = "/" + os.path.join(*sb3.__file__.split("/")[:-2])
    commit_num = subprocess.check_output(
        ["git", "describe", "--always"], cwd=path).strip().decode()

    env = make_atari_env(envname, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=n_stack)

    eval_env = make_atari_env(envname, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    eval_env = VecTransposeImage(eval_env)

    # Callbacks
    loggercallback = LoggerCallback(
        "json",
        [("arguments", all_args),
         ("git", commit_num)])

    # No seed as the evaluation has no effect on training or pruning
    

    # Initiate the model and start learning
    model = A2C(policy,
                env,
                learning_rate,
                n_steps,
                batch_size,
                epochs,
                gamma,
                gae_lambda,
                ent_coef,
                vf_coef,
                max_grad_norm,
                normalize_advantage,
                policy_kwargs,
                verbose=verbose,
                tensorboard_log=tensorboard_log,
                seed=seed,
                create_eval_env=False,
                device="cuda")

    evalcallback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq)

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
        if value == "bool":
            kwargs[key] = bool

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
