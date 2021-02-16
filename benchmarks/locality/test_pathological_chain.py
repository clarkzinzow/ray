import argparse
import timeit
import numpy as np
import ray

from ray.cluster_utils import Cluster
from ray.ray_perf import check_optimized_build


@ray.remote
def f(x):
    return x + 1


def run_trials(ntrials, nbytes, chain_length, timeout):
    print("Running pathological chain benchmark")
    arr = np.zeros(nbytes // 8, dtype=np.int64)

    times = []
    start = timeit.default_timer()

    for i in range(ntrials):
        trial_start = timeit.default_timer()
        a = f.remote(arr)
        for n in range(chain_length - 1):
            if n % 2 == 0:
                a = f.options(resources={"pin": 1}).remote(a)
            else:
                a = f.remote(a)
        ray.get(a)
        trial_end = timeit.default_timer()
        duration = trial_end - trial_start
        times.append(duration)
        print(f"Trial {i} done after {duration}")
        if timeit.default_timer() - start > timeout:
            break

    return times


if __name__ == "__main__":
    check_optimized_build()

    parser = argparse.ArgumentParser(
        description="Run data locality benchmarks")
    parser.add_argument(
        "--ntrials", required=False, type=int, default=5,
        help="Number of trials")
    parser.add_argument(
        "--nbytes", required=False, type=int, default=800 * 1024 * 1024,
        help="Size of data")
    parser.add_argument(
        "--chain-length", required=False, type=int, default=20,
        help="Length of task chain")
    parser.add_argument(
        "--trials-timeout", required=False, type=int, default=60,
        help="Timeout for all trials")
    parser.add_argument(
        "--cluster", action="store_true",
        help="Whether to run on multi-node cluster")
    parser.add_argument(
        "--locality-aware-scheduling", action="store_true",
        help="Whether locality-aware scheduling is enabled")
    args = parser.parse_args()

    print("Testing data locality")

    if not args.cluster:
        print("Starting local Ray cluster")
        system_config = {
            "locality_aware_leasing_enabled": args.locality_aware_scheduling,
        }
        cluster = Cluster(
            initialize_head=True,
            connect=True,
            head_node_args={
                "object_store_memory": 10 * 1024 * 1024 * 1024,
                "num_cpus": 2,
            },
            _system_config=system_config)
        cluster.add_node(
            object_store_memory=10 * 1024 * 1024 * 1024,
            num_cpus=2,
            resources={"pin": 1})
    else:
        print("Connecting to existing Ray cluster")
        ray.init(address="auto")

    print("Running warmup...")
    times = run_trials(
        ntrials=5, nbytes=80 * 1024 * 1024, chain_length=10, timeout=20)
    print(f"Warmup pathological chain mean over {len(times)} trials: "
          f"{np.mean(times)} +- {np.std(times)}")

    print("Starting real trials...")
    times = run_trials(
        args.ntrials, args.nbytes, args.chain_length, args.trials_timeout)
    print(f"Pathological chain mean over {len(times)} trials: "
          f"{np.mean(times)} +- {np.std(times)}")
