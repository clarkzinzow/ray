import time
import numpy as np
import ray

from ray.cluster_utils import Cluster
from ray.ray_perf import check_optimized_build


def test_data_locality():
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={
            "object_store_memory": 10 * 1024 * 1024 * 1024,
            "num_cpus": 2,
            "_system_config": {
                # Disable worker caching so worker leases are not reused, and
                # disable inlining of return objects so return objects are
                # always put into Plasma.
                "worker_lease_timeout_milliseconds": 0,
                "max_direct_call_object_size": 0,
                # "put_small_object_in_memory_store": True,
            },
        })
    cluster.add_node(
        object_store_memory=10 * 1024 * 1024 * 1024,
        num_cpus=2,
        resources={"pin": 1})

    print("Running pathological chain benchmark")

    @ray.remote
    def f(x):
        return x + 1

    arr = np.zeros(100 * 1024 * 1024, dtype=np.int64)  # 800 MiB

    print("Running warmup...")

    # warmup
    a = f.remote(arr)
    for n in range(10):
        if n % 2 == 0:
            a = f.options(resources={"pin": 1}).remote(a)
        else:
            a = f.remote(a)
    ray.get(a)
    del a

    print("Starting real trials...")

    # real trials
    chain_length = 20
    ntrials = 5
    times = []
    timeout = 60
    start = time.time()

    for i in range(ntrials):
        trial_start = time.time()
        a = f.remote(arr)
        for n in range(chain_length - 1):
            if n % 2 == 0:
                a = f.options(resources={"pin": 1}).remote(a)
            else:
                a = f.remote(a)
        ray.get(a)
        trial_end = time.time()
        duration = trial_end - trial_start
        times.append(duration)
        print(f"Trial {i} done after {duration}")
        if time.time() - start > timeout:
            break

    print(f"Pathological chain mean over {len(times)} trials: "
          f"{np.mean(times)} +- {np.std(times)}")


def main():
    check_optimized_build()

    print("Testing data locality")
    test_data_locality()


if __name__ == "__main__":
    main()
