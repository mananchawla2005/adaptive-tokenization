"""Minimal experiment tracker using Weights & Biases."""
import os
import time
import random
import hashlib
import subprocess

import numpy as np
import torch
import wandb


def _get_git_commit():
    commit_file = os.path.join(os.path.dirname(__file__), "COMMIT.txt")
    if os.path.exists(commit_file):
        try:
            with open(commit_file) as f:
                return f.read().strip()
        except Exception:
            pass
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_experiment_id():
    commit = _get_git_commit()
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"exp-{ts}-g{commit}"


class Tracker:
    def __init__(
        self,
        name,
        project="adaptive-tokenization",
        config=None,
        group=None,
        job_type=None,
        tags=None,
    ):
        self.name = name
        self.project = project
        self.run = None
        self._config = config or {}
        self._group = group
        self._job_type = job_type
        self._tags = tags or []

    def init(self):
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self._config,
            group=self._group,
            job_type=self._job_type,
            tags=self._tags,
            reinit=True,
        )
        return self.run

    def log(self, data, step=None):
        if self.run:
            self.run.log(data, step=step)

    def summary(self, data):
        if self.run:
            for k, v in data.items():
                self.run.summary[k] = v

    def finish(self, exit_code=0):
        if self.run:
            self.run.finish(exit_code=exit_code)
