import ray
# import datetime
import os.path
import csv

import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import numpy as np
from ray.experimental.dask import ray_dask_get

import time


def load_dataset(nbytes, npartitions, sort):
    num_bytes_per_partition = nbytes // npartitions
    filenames = []
    for i in range(npartitions):
        filename = ("gs://dl-clark-dev/dask_on_ray_benchmarks/source_data/"
                    "df-{}-{}-{}.parq".format("sort" if sort else "groupby",
                                              num_bytes_per_partition, i))
        print("Partition file", filename)
        if sort:
            nrows = num_bytes_per_partition // 8
            print("Allocating dataset with {} rows".format(nrows))
            dataset = pd.DataFrame(
                np.random.randint(
                    0,
                    np.iinfo(np.uint64).max,
                    size=(nrows, 1),
                    dtype=np.uint64),
                columns=["a"])
        else:
            nrows = num_bytes_per_partition // (8 * 2)
            print("Allocating dataset with {} rows".format(nrows))
            dataset = pd.DataFrame(
                np.random.randint(
                    0,
                    np.iinfo(np.uint64).max,
                    size=(nrows, 2),
                    dtype=np.uint64),
                columns=["a", "b"])
        print("Done allocating")
        dataset.to_parquet(filename, compression="gzip")
        print("Done writing to disk")
        filenames.append(filename)

    df = dd.read_parquet(filenames)
    return df


def trial(nbytes, n_partitions, sort, use_ray):
    df = load_dataset(nbytes, n_partitions, sort)
    times = []
    start = time.time()
    for i in range(10):
        print("Trial {} start".format(i))
        trial_start = time.time()

        if sort:
            df.set_index("a").compute(
                scheduler=ray_dask_get if use_ray else "distributed")
        else:
            df.groupby("b").a.mean().compute(
                scheduler=ray_dask_get if use_ray else "distributed")

        trial_end = time.time()
        duration = trial_end - trial_start
        times.append(duration)
        print("Trial {} done after {}".format(i, duration))

        if time.time() - start > 60:
            break
    return times


def setup(is_dask=False, is_ray=False, create_local=False):
    if not is_dask and not is_ray:
        raise ValueError("Need to specify at least one of --dask or --ray")

    if is_dask:
        # We assume that there's a dask-scheduler and n dask-workers on the
        # current node.
        if create_local:
            client = Client()
        else:
            client = Client("127.0.0.1:8786")
            client.restart()
    if is_ray:
        if create_local:
            ray.init(_lru_evict=True)
        else:
            ray.init(address="auto")


def run_benchmark(
        nbytes,
        npartitions,
        max_partition_size=10e9,
        num_nodes=1,
        sort=False,
        timeline=False,
        is_dask=False,
        is_ray=False,
        output_filename="outputs/shuffle.csv",
):
    if is_dask:
        print("dask", trial(1000, 10, sort, use_ray=False))
    if is_ray:
        print("ray", trial(1000, 10, sort, use_ray=True))
    print("WARMUP DONE")

    npartitions = npartitions
    if nbytes // npartitions > max_partition_size:
        npartitions = nbytes // max_partition_size

    if is_dask:
        dask_output = trial(nbytes, npartitions, sort, use_ray=False)
        print("dask mean over {} trials: {} +- {}".format(
            len(dask_output), np.mean(dask_output), np.std(dask_output)))
    else:
        dask_output = []
    if is_ray:
        ray_output = trial(nbytes, npartitions, sort, use_ray=True)
        print("ray mean over {} trials: {} +- {}".format(
            len(ray_output), np.mean(ray_output), np.std(ray_output)))
    else:
        ray_output = []

    write_header = not os.path.exists(output_filename) or os.path.getsize(
        output_filename) == 0
    # filename = "shuffle"
    # if sort:
    #     filename += "_sort"
    # filename += f"_max_part_{max_partition_size:.2e}"
    # filename += f"_parts_{npartitions:.2e}"
    # filename += f"_bytes_{nbytes:.2e}"
    # if dask and ray:
    #     filename += f"_dask_ray"
    # elif dask:
    #     filename += f"_dask"
    # else:
    #     filename += f"_ray"
    # now = datetime.datetime.now().isoformat()
    # filename += f"_{now}"

    # with open(f"outputs/{filename}.csv", "a+") as csvfile:

    with open(output_filename, "a+") as csvfile:
        fieldnames = [
            "system",
            "operation",
            "num_nodes",
            "nbytes",
            "npartitions",
            "duration",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "operation": "sort" if sort else "groupby",
            "num_nodes": num_nodes,
            "nbytes": nbytes,
            "npartitions": npartitions,
        }
        for output in dask_output:
            row["system"] = "dask"
            row["duration"] = output
            writer.writerow(row)
        for output in ray_output:
            row["system"] = "ray"
            row["duration"] = output
            writer.writerow(row)

    if timeline:
        time.sleep(1)
        ray.timeline(filename="dask.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--nbytes", type=int, default=1_000_000)
    parser.add_argument("--npartitions", type=int, default=100, required=False)
    # Max partition size is 1GB.
    parser.add_argument(
        "--max-partition-size", type=int, default=1000_000_000, required=False)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--dask", action="store_true")
    parser.add_argument("--ray", action="store_true")
    args = parser.parse_args()

    setup(is_dask=args.dask, is_ray=args.ray)

    run_benchmark(
        args.nbytes,
        args.npartitions,
        max_partition_size=args.max_partition_size,
        num_nodes=args.num_nodes,
        sort=args.sort,
        timeline=args.timeline,
        is_dask=args.dask,
        is_ray=args.ray,
    )
