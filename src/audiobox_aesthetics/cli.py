# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import itertools
from pathlib import Path

import submitit
from .infer import load_dataset, main_predict


def parse_args():
    parser = argparse.ArgumentParser("CLI for audiobox-aesthetics inference")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--remote", action="store_true", default=False, help="Set true to run via SLURM"
    )

    # remote == True
    parser.add_argument(
        "--job-dir", default="/tmp", type=str, help="Slurm job directory"
    )
    parser.add_argument(
        "--partition", default="learn", type=str, help="Slurm partition"
    )
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="", type=str, help="Slurm account")
    parser.add_argument("--comment", default="", type=str, help="Slurm job comment")
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint eg.: ampere80gb For using A100s or volta32gb for using V100s.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Exclude certain nodes from the slurm job.",
    )
    parser.add_argument(
        "--array", default=100, type=int, help="Slurm max array parallelism"
    )
    parser.add_argument(
        "--chunk", default=1000, type=int, help="chunk size per instance"
    )
    return parser.parse_args()


def app():
    args = parse_args()

    metadata = load_dataset(args.input_file, 0, 2**64)
    fn_wrapped = partial(main_predict, batch_size=args.batch_size, ckpt=args.ckpt)

    if args.remote:
        # chunk metadata
        chunksize = args.chunk
        chunked = [
            metadata[ii : ii + chunksize] for ii in range(0, len(metadata), chunksize)
        ]

        job_dir = Path(args.job_dir)
        job_dir.mkdir(exist_ok=True)

        executor = submitit.AutoExecutor(folder=f"{job_dir}/%A/")

        kwargs = {}
        if len(args.constraint):
            kwargs["slurm_constraint"] = args.constraint
        if args.comment:
            kwargs["slurm_comment"] = args.comment
        if args.qos:
            kwargs["slurm_qos"] = args.qos
        if args.account:
            kwargs["slurm_account"] = args.account

        # Set the parameters for the Slurm job
        executor.update_parameters(
            slurm_nodes=1,
            slurm_gpus_per_node=1,
            slurm_tasks_per_node=1,
            slurm_cpus_per_task=10,
            timeout_min=60 * 20,  # max is 20 hours
            slurm_array_parallelism=min(
                len(chunked), args.array
            ),  # number of tasks in the array job
            slurm_partition=args.partition,
            slurm_exclude=args.exclude,
            **kwargs,
        )

        jobs = executor.map_array(fn_wrapped, chunked)
        outputs = [job.result() for job in jobs]

        outputs = itertools.chain(*outputs)
    else:
        outputs = fn_wrapped(metadata)
    print("\n".join(str(x) for x in outputs))


if __name__ == "__main__":
    """
    Example usage:
    python cli.py input.jsonl --batch-size 100 --ckpt /path/to/ckpt > output.jsonl
    """
    app()
