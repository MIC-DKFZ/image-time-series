import os
import re
import subprocess
import subprocess as subp
import sys
import argparse
from typing import Any, Dict, Optional, Union
from collections import defaultdict
import tempfile
import shutil
import zipfile
import torch

from visdom import Visdom

import pytorch_lightning as pl
import mlflow
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class SourcePacker(object):
    """
    Inspired by https://github.com/IDSIA/sacred
    """

    @staticmethod
    def join_paths(*parts):
        """Join different parts together to a valid dotted path."""
        return ".".join(str(p).strip(".") for p in parts if p)

    @staticmethod
    def iter_prefixes(path):
        """
        Iterate through all (non-empty) prefixes of a dotted path.
        Example:
        >>> list(iter_prefixes('foo.bar.baz'))
        ['foo', 'foo.bar', 'foo.bar.baz']
        """
        split_path = path.split(".")
        for i in range(1, len(split_path) + 1):
            yield SourcePacker.join_paths(*split_path[:i])

    @staticmethod
    def create_source_or_dep(mod, sources):
        filename = ""
        if mod is not None and hasattr(mod, "__file__"):
            filename = os.path.abspath(mod.__file__)

        ### To source or dependency
        if filename and filename not in sources and SourcePacker.is_source(filename):
            sources.add(filename)

    @staticmethod
    def is_source(filename):
        if (
            ".virtualenvs" in filename
            or "site-packages" in filename
            or re.search("python[0-9]\.[0-9]", filename) is not None
        ):
            return False
        else:
            return True

    @staticmethod
    def git_info(file_):

        old_dir = os.getcwd()
        file_path = os.path.abspath(file_)
        os.chdir(os.path.dirname(file_path))

        try:
            commit = subp.check_output(["git", "rev-parse", "HEAD"]).decode("ascii")[
                :-1
            ]
            branch = subp.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode("ascii")[:-1]
            repo = subp.check_output(["git", "remote", "-vv"]).decode("ascii")
            repo = re.findall(
                "(?<=origin[\s\t])(http.+|ssh.+|git.+)(?=[\s\t]\(fetch)", repo
            )[0]
            result = (repo, branch, commit)
        except Exception as e:
            print("Could not find git info for {}".format(file_path))
            print(e)
            result = (None, None, None)

        os.chdir(old_dir)
        return result

    @staticmethod
    def gather_sources_and_dependencies(globs):
        py_str = "python {}".format(sys.version)
        dependencies = (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            .decode("utf-8")
            .split("\n")
        )

        filename = globs.get("__file__")

        if filename is None:
            sources = set()
        else:
            sources = set()
            sources.add(filename)
        for glob in globs.values():
            if isinstance(glob, type(sys)):
                mod_path = glob.__name__
            elif hasattr(glob, "__module__"):
                mod_path = glob.__module__
            else:
                continue

            if not mod_path:
                continue

            for modname in SourcePacker.iter_prefixes(mod_path):
                mod = sys.modules.get(modname)
                SourcePacker.create_source_or_dep(mod, sources)

        return py_str, sources, dependencies

    @staticmethod
    def zip_sources(globs, filename):

        py_str, sources, dependencies = SourcePacker.gather_sources_and_dependencies(
            globs=globs
        )
        repo, branch, commit = SourcePacker.git_info(globs.get("__file__"))
        cmd = " ".join(sys.argv)

        with zipfile.ZipFile(filename, mode="w") as zf:
            for source in sources:
                zf.write(source)

            zf.writestr("python_version.txt", py_str)
            dep_str = "\n".join(dependencies)
            zf.writestr("modules.txt", dep_str)
            git_str = "Repository: {}\nBranch: {}\nCommit: {}".format(
                repo, branch, commit
            )
            zf.writestr("git_info.txt", git_str)
            zf.writestr("command.txt", cmd)


vis_ = None


def get_visdom(
    name: Optional[str] = "main",
    server: Optional[str] = "http://localhost",
    endpoint: Optional[str] = "events",
    port: Optional[int] = 8097,
    base_url: Optional[str] = "/",
    ipv6: Optional[bool] = True,
    http_proxy_host: Optional[str] = None,
    http_proxy_port: Optional[int] = None,
    send: Optional[bool] = True,
    raise_exceptions: Optional[bool] = None,
    use_incoming_socket: Optional[bool] = True,
    log_to_filename: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    proxies: Optional[str] = None,
    offline: Optional[bool] = False,
    use_polling: Optional[bool] = False,
):

    global vis_

    if vis_ is None:
        vis_ = Visdom(
            server=server,
            endpoint=endpoint,
            port=port,
            base_url=base_url,
            ipv6=ipv6,
            http_proxy_host=http_proxy_host,
            http_proxy_port=http_proxy_port,
            env=name,
            send=send,
            raise_exceptions=raise_exceptions,
            use_incoming_socket=use_incoming_socket,
            log_to_filename=log_to_filename,
            username=username,
            password=password,
            proxies=proxies,
            offline=offline,
            use_polling=use_polling,
        )

    return vis_


class VisdomLogger(LightningLoggerBase):
    def __init__(
        self,
        name: Optional[str] = "main",
        server: Optional[str] = "http://localhost",
        endpoint: Optional[str] = "events",
        port: Optional[int] = 8097,
        base_url: Optional[str] = "/",
        ipv6: Optional[bool] = True,
        http_proxy_host: Optional[str] = None,
        http_proxy_port: Optional[int] = None,
        send: Optional[bool] = True,
        raise_exceptions: Optional[bool] = None,
        use_incoming_socket: Optional[bool] = True,
        log_to_filename: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxies: Optional[str] = None,
        offline: Optional[bool] = False,
        use_polling: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()
        self._name = name or ""

        self._experiment = get_visdom(
            server=server,
            endpoint=endpoint,
            port=port,
            base_url=base_url,
            ipv6=ipv6,
            http_proxy_host=http_proxy_host,
            http_proxy_port=http_proxy_port,
            name=self._name,
            send=send,
            raise_exceptions=raise_exceptions,
            use_incoming_socket=use_incoming_socket,
            log_to_filename=log_to_filename,
            username=username,
            password=password,
            proxies=proxies,
            offline=offline,
            use_polling=use_polling,
        )
        self.hparams = {}
        self._kwargs = kwargs

        self._step_cntr = defaultdict(int)

    @property
    @rank_zero_experiment
    def experiment(self) -> Visdom:
        if self._experiment is None:
            return get_visdom()
        else:
            return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[Dict[str, Any], argparse.Namespace],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        print("Visdom log_hyperparams not implemented yet...")

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # metrics = self._add_prefix(metrics)

        _default_title_lookup = {"loss": "Loss"}

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                print("Not implemented yet...")
                # self.experiment.add_scalars(k, v, step)
            else:
                try:
                    if k in self._step_cntr:
                        update_str = "append"
                    else:
                        update_str = "replace"
                    if step is None:
                        step = self._step_cntr[k] + 1
                    win = k
                    for key, val in _default_title_lookup.items():
                        if key in k:
                            win = val
                            break
                    opts = {"title": win, "showlegend": True}
                    get_visdom().line(
                        Y=[v], X=[step], win=win, update=update_str, name=k, opts=opts
                    )
                    self._step_cntr[k] = int(step)
                except Exception as e:
                    m = f"\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor."
                    type(e)(e.message + m)

    @rank_zero_only
    def log_graph(self, model: LightningModule, input_array=None):
        print("Visdom log_graph not implemented yet...")

    @rank_zero_only
    def save(self) -> None:
        # super().save()
        get_visdom().save([self.name])

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state


def make_default_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--visdom", type=str, default=None)
    parser.add_argument("--mlflow", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("--test", action="store_true", help="Run test.")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training (you probably want to set --test as well).",
    )

    return parser


def log_model_summary(experiment):

    summary = str(pl.core.memory.ModelSummary(experiment, mode="full"))
    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary)
        mlflow.log_artifact(summary_file)
    finally:
        shutil.rmtree(tempdir)


def log_sources(globs):

    tempdir = tempfile.mkdtemp()
    try:
        f = os.path.join(tempdir, "sources.zip")
        SourcePacker.zip_sources(globs, f)
        mlflow.log_artifact(f)
    finally:
        shutil.rmtree(tempdir)


def run_experiment(
    ExperimentModule,
    DataModule,
    args,
    name,
    globs=None,
    callback_monitor="val_loss_total",
    **callback_kwargs,
):

    # Logging
    mlflow.set_tracking_uri(args.mlflow)
    mlflow.set_experiment(name)
    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=name,
            tracking_uri=args.mlflow,
            tags=mlflow.tracking.context.registry.resolve_tags(),
        )
    ]
    if args.visdom:
        loggers.append(VisdomLogger(name=args.name, port=args.visdom))
    mlflow.start_run(run_id=loggers[0].run_id)
    if args.name != name:
        # illegal :)
        mlflow.set_tag("mlflow.runName", args.name)
    args.default_root_dir = mlflow.active_run().info.artifact_uri[7:]

    # seeding first
    pl.seed_everything(args.seed)

    # checkpointing
    default_callback_kwargs = dict(
        dirpath=args.default_root_dir,
        filename="checkpoint_{epoch:03d}_{val_loss_total:.3f}",
        save_top_k=3,
        save_last=True,
        mode="min",
    )
    default_callback_kwargs.update(callback_kwargs)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=callback_monitor, **default_callback_kwargs
    )

    # construct data, trainer
    if args.resume_from_checkpoint:
        dm = DataModule.load_from_checkpoint(args.resume_from_checkpoint)
        experiment = ExperimentModule.load_from_checkpoint(args.resume_from_checkpoint)
    else:
        dm = DataModule(**vars(args))
        experiment = ExperimentModule(**vars(args))
    trainer = pl.Trainer().from_argparse_args(
        args, logger=loggers, deterministic=True, callbacks=[checkpoint_callback]
    )

    # log all hparams and model summary
    # MLFlow can only log 100 hparams at once...
    hparam_dicts = []
    for i, (key, val) in enumerate(experiment.hparams.items()):
        if i % 100 == 0:
            hparam_dicts.append({})
        hparam_dicts[-1][key] = val
    for hp in hparam_dicts:
        mlflow.log_params(hp)
    mlflow.log_param("command", " ".join(sys.argv))
    log_model_summary(experiment)

    # log source files
    if globs is not None:
        log_sources(globs)

    # run training
    if not args.no_train:
        trainer.fit(experiment, datamodule=dm)

    # run testing
    if args.test:
        if args.resume_from_checkpoint:
            trainer.test(model=experiment, datamodule=dm)
        else:
            try:
                trainer.test(ckpt_path="best", datamodule=dm)
            except pl.utilities.exceptions.MisconfigurationException:
                trainer.test(ckpt_path=None, datamodule=dm)