import datetime
import logging
import os
import subprocess
import _io

import yaml

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def save_git_info(git_commit_file: str, git_diff_file: str,
                  branch: str = "HEAD") -> None:
    repo_dir = os.path.dirname(os.path.realpath(__file__))

    with open(git_commit_file, "wb") as file:
        subprocess.run(["git", "log", "-1", "--format=%H", branch],
                       cwd=repo_dir, stdout=file, check=False)

    with open(git_diff_file, "wb") as file:
        subprocess.run(
            ["git", "--no-pager", "diff", "--color=always", branch],
            cwd=repo_dir, stdout=file, check=False)


def experiment_logging(root: str, experiment_id: str, args) -> str:
    experiment_dir = os.path.join(root, experiment_id)
    os.mkdir(experiment_dir)

    save_git_info(
        os.path.join(experiment_dir, "git_commit"),
        os.path.join(experiment_dir, "git_diff"))

    args_dict = {}
    for key, val in args.__dict__.items():
        if isinstance(val, _io.TextIOWrapper):
            args_dict[key] = os.path.realpath(val.name)
        else:
            args_dict[key] = val

    with open(os.path.join(experiment_dir, "args"), "w") as file:
        print(yaml.dump(args_dict), file=file)

    log_handler = logging.FileHandler(
        os.path.join(experiment_dir, "train.log"))
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(log_handler)

    return experiment_dir


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
