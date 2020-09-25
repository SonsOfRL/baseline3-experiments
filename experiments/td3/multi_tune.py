import os
from multiprocessing import Process
import argparse


def script(ix):
    os.system("python tune.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tunes", help="Number of parallel processes for tuning",
                        default=1, type=int, required=False)
    cl_arg = parser.parse_args()
    processes = []
    for ix in range(cl_arg.n_tunes):
        proc = Process(target=script, args=(ix,))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
