import os
import re
import subprocess
import subprocess as subp
import sys
import argparse
import json
from typing import Any, Dict, Optional, Union, Tuple, List
from types import ModuleType
from collections import defaultdict
import tempfile
import shutil
import zipfile
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from visdom import Visdom
import pytorch_lightning as pl
import mlflow
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class SourcePacker(object):
    """
    Methods taken from https://github.com/IDSIA/sacred,
    will collect all custom source code.
    """

    @staticmethod
    def join_paths(*parts: str) -> str:
        """Join different parts together to a valid dotted path.

        Args:
            parts: Elements of a path.

        Return:
            Path elements joined by dots.

        """
        return ".".join(str(p).strip(".") for p in parts if p)

    @staticmethod
    def iter_prefixes(path: str) -> str:
        """Iterate through all (non-empty) prefixes of a dotted path.

        Example:
        >>> list(iter_prefixes('foo.bar.baz'))
        ['foo', 'foo.bar', 'foo.bar.baz']

        Args:
            path: Path with elements separated by dots.

        Returns:
            All possible prefixes.

        """
        split_path = path.split(".")
        for i in range(1, len(split_path) + 1):
            yield SourcePacker.join_paths(*split_path[:i])

    @staticmethod
    def create_source_or_dep(mod: ModuleType, sources: list):
        """Add module file to source list.

        Args:
            mod: Module to provide files.
            sources: List to collect file names.

        """

        filename = ""
        if mod is not None and hasattr(mod, "__file__"):
            filename = os.path.abspath(mod.__file__)

        if filename and filename not in sources and SourcePacker.is_source(filename):
            sources.add(filename)

    @staticmethod
    def is_source(filename: str) -> bool:
        """Check if a file is a proper source file.

        Args:
            filename: File to check.

        Returns:
            True or False

        """

        if (
            ".virtualenvs" in filename
            or "site-packages" in filename
            or re.search("python[0-9]\.[0-9]", filename) is not None
        ):
            return False
        else:
            return True

    @staticmethod
    def git_info(filename: str) -> Tuple[str, str, str]:
        """Get git information (repo, branch, commit) for a file.

        Args:
            filename: The file for which we want to find the git information.

        Returns:
            Tuple of repo, branch and commit (or None, None, None)

        """

        old_dir = os.getcwd()
        file_path = os.path.abspath(filename)
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
    def gather_sources_and_dependencies(
        globs: Dict,
    ) -> Tuple[str, List[str], List[str]]:
        """Get source information from globals() or another dict.

        Args:
            globs: globals()

        Returns:
            Tuple of python version, list of source files, list of dependencies.

        """

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
    def zip_sources(globs: Dict, filename: str):
        """Create zip file with all custom source files and meta information.

        Args:
            globs: globals().
            filename: Output file.

        """

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
    server: str = "http://localhost",
    endpoint: str = "events",
    port: int = 8097,
    base_url: str = "/",
    ipv6: bool = True,
    http_proxy_host: Optional[str] = None,
    http_proxy_port: Optional[int] = None,
    send: bool = True,
    raise_exceptions: Optional[bool] = None,
    use_incoming_socket: bool = True,
    log_to_filename: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    offline: bool = False,
    use_polling: bool = False,
) -> Visdom:
    """Helper function to construct or get a global Visdom server object.

    Args:
        name: Name of the Visdom environment.
        server: Hostname for the server.
        endpoint: Endpoint to post events.
        port: Run the server on this port.
        base_url: Server base URL.
        ipv6: Use IPv6.
        http_proxy_host: Deprecated, use proxies.
        http_proxy_port: Deprecated, use proxies.
        send: Not entirely sure what this does that couldn't be achieved with
            use_incoming_socket...
        raise_exceptions: Raise exceptions instead of printing.
        use_incoming_socket: Activate this to use callbacks.
        log_to_filename: Log all events to this file.
        username: Username for authentication.
        password: Password for authentication (will be SHA256 hashed).
        proxies: Dictionary of proxy mappings, e.g. {'http': 'foo.bar:3128'}.
        offline: Offline mode, logs to file instead of server.
        use_polling: Use polling instead of websocket?

    Returns:
        A global Visdom instance

    """

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
    """A Lightning logger class for Visdom.

    Args:
        name: Name of the Visdom environment.
        server: Hostname for the server.
        endpoint: Endpoint to post events.
        port: Run the server on this port.
        base_url: Server base URL.
        ipv6: Use IPv6.
        http_proxy_host: Deprecated, use proxies.
        http_proxy_port: Deprecated, use proxies.
        send: Not entirely sure what this does that couldn't be achieved with
            use_incoming_socket...
        raise_exceptions: Raise exceptions instead of printing.
        use_incoming_socket: Activate this to use callbacks.
        log_to_filename: Log all events to this file.
        username: Username for authentication.
        password: Password for authentication (will be SHA256 hashed).
        proxies: Dictionary of proxy mappings, e.g. {'http': 'foo.bar:3128'}.
        offline: Offline mode, logs to file instead of server.
        use_polling: Use polling instead of websocket?

    """

    def __init__(
        self,
        name: Optional[str] = "main",
        server: str = "http://localhost",
        endpoint: str = "events",
        port: int = 8097,
        base_url: str = "/",
        ipv6: bool = True,
        http_proxy_host: Optional[str] = None,
        http_proxy_port: Optional[int] = None,
        send: bool = True,
        raise_exceptions: Optional[bool] = None,
        use_incoming_socket: bool = True,
        log_to_filename: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxies: Optional[str] = None,
        offline: bool = False,
        use_polling: bool = False,
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
        """Return the underlying Visdom object."""

        if self._experiment is None:
            return get_visdom()
        else:
            return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[Dict[str, Any], argparse.Namespace],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Not implemented yet, only here for API completeness."""
        print("Visdom log_hyperparams not implemented yet...")

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar valued metrics.

        Args:
            metrics: A {name: value} dictionary.
            step: Value for the x axis.

        """

        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # metrics = self._add_prefix(metrics)

        _default_title_lookup = {"loss": "Loss"}

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                print("Dict logging not implemented yet...")
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
                    m = (
                        "\n you tried to log {}, ".format(v)
                        + "which is not currently supported."
                        + "Try a dict or a scalar/tensor."
                    )
                    type(e)(e.message + m)

    @rank_zero_only
    def log_graph(self, model: LightningModule, input_array: torch.Tensor = None):
        """Not implemented yet, only here for API completeness."""
        print("Visdom log_graph not implemented yet...")

    @rank_zero_only
    def save(self):
        """Save the Visdom environment."""
        self.experiment.save([self.name])

    @rank_zero_only
    def finalize(self, status: str):
        """Save environment at the end."""
        self.save()

    @property
    def name(self) -> str:
        """Return the Visdom environment name."""
        return self._name

    @property
    def version(self) -> int:
        """Don't use this information."""
        return 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state


def make_default_parser() -> argparse.ArgumentParser:
    """Construct a parser with a few commonly used options."""

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


def str2bool(v: Union[bool, str]) -> bool:
    """Use this as the type for an ArgumentParser argument.

    Allows you to do --my_flag false and similar, so the flag name can be positive,
    regardless of the default value.

    Args:
        v: The parsed value.

    Returns:
        v interpreted as a boolean.

    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DictArgument(argparse.Action):
    """Custom action to allow dictionary CLI inputs.

    Use with `action=DictArgument` and `nargs="+"` to allow user inputs like::

        --my_dict a=b c=0.1

    """

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                kv = json.loads("{" + '"{}":{}'.format(k, v) + "}")
            except json.decoder.JSONDecodeError:
                try:
                    kv = json.loads("{" + '"{}":"{}"'.format(k, v) + "}")
                except Exception as e:
                    raise e
            except Exception as e:
                raise e
            my_dict.update(kv)
        setattr(namespace, self.dest, my_dict)


def log_model_summary(experiment: LightningModule):
    """Log a summary of a modules model in MLFlow."""

    summary = str(pl.core.memory.ModelSummary(experiment, max_depth=-1))
    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary)
        mlflow.log_artifact(summary_file)
    finally:
        shutil.rmtree(tempdir)


def log_sources(globs: Dict):
    """Log source files for experiment in MLFlow.

    Args:
        globs: globals().

    """

    tempdir = tempfile.mkdtemp()
    try:
        f = os.path.join(tempdir, "sources.zip")
        SourcePacker.zip_sources(globs, f)
        mlflow.log_artifact(f)
    finally:
        shutil.rmtree(tempdir)


def run_experiment(
    ExperimentModule: LightningModule,
    DataModule: LightningDataModule,
    args: argparse.Namespace,
    name: str,
    globs: Optional[Dict] = None,
    callback_monitor: Optional[str] = "val_loss_total",
    **callback_kwargs,
):
    """Run a LightningModule with a LightningDataModule.

    This sets up logging, seeds, checkpointing and some other stuff.

    Args:
        ExperimentModule: The LightningModule
        DataModule: The LightningDataModule
        args: Parsed arguments from an ArgumentParser. Should have a few attributes,
            including .mlflow, .visdom, .name, .seed, .no_train, .test, and anything
            else that you would normally pass directly to the modules.
        name:
            A name for the experiment. MLFlow will use this as the experiment name and
            args.name as the run name.
        globs: globals(). Will log source files if this is provided.
        callback_monitor: Monitor this metric for model checkpointing.
        callback_kwargs: ModelCheckpoint will be initialized with this.

    """

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
        loggers.append(VisdomLogger(name=name, port=args.visdom))
    mlflow.start_run(run_id=loggers[0].run_id)
    if args.name != name:
        # illegal :)
        mlflow.set_tag("mlflow.runName", args.name)
    args.default_root_dir = mlflow.active_run().info.artifact_uri[7:]

    # seeding first
    pl.seed_everything(args.seed)

    # checkpointing
    if callback_monitor is not None:
        default_callback_kwargs = dict(
            dirpath=args.default_root_dir,
            filename="checkpoint_{epoch:03d}_{val_loss_total:.3f}",
            save_top_k=3,
            save_last=True,
            mode="min",
            # every_n_train_steps=None,
            # every_n_val_epochs=1,
            every_n_epochs=1,
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
    # hparam_dicts = []
    # for i, (key, val) in enumerate(experiment.hparams.items()):
    #     if i % 100 == 0:
    #         hparam_dicts.append({})
    #     hparam_dicts[-1][key] = val
    # for hp in hparam_dicts:
    #     mlflow.log_params(hp)
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