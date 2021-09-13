import argparse
import logging
# import os
import pdb
import time

import ray
# from ray.util.placement_group import (
#     placement_group,
#     placement_group_table,
#     remove_placement_group
# )


def create_parser():
    parser = argparse.ArgumentParser(description="Eric Example")
    parser.add_argument("--address")
    parser.add_argument("--num-rows", type=int, default=1 * (10**9))
    parser.add_argument("--num-files", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--read-cache", action="store_true", default=False)
    parser.add_argument(
        "--data-dir", type=str,
        default="s3://shuffling-data-loader-benchmarks/data/")
    return parser


parser = create_parser()
args = parser.parse_args()

# read_bundle = {"CPU": 0.3}
# pg = placement_group([read_bundle] * num, strategy="SPREAD")
# ray.wait([pg.ready()], timeout=0)

files = [
    "s3://shuffling-data-loader-benchmarks/data/r1_000_000_000-f200"
    f"/input_data_{i}.parquet.snappy" for i in range(args.num_files)]
# data_dir = os.path.join(
#     args.data_dir, f"r{args.num_rows:_}-f{args.num_files}/")

logging.debug("start")
print(f"reading {len(files)} files from s3")

print("start read_parquet()")
start = time.time()
# ds = ray.data.read_parquet(
#     files, ray_remote_args={"num_cpus": 0.3}, placement_group=pg)
ds = ray.data.read_parquet(files, ray_remote_args={"num_cpus": 0.3})
print("creating blocks")
blocks = ds.get_blocks()
print("start waiting for blocks")
ray.wait(blocks, num_returns=len(blocks), fetch_local=False)
# ray.wait(ds._blocks)
# ds.map_batches(lambda x:None)
# .take(1)   # first block
print(f"read_parquet took {time.time() - start}")
# remove_placement_group(pg)

# import pdb;pdb.set_trace()

total_size = ds.count()
print(f"ds size: {ds.size_bytes() / 2**30} gb")
#
# print("start repart()")
# start = time.time()
# ds = ds.repartition(num_blocks=16)  # second block
# print(f"repart took {time.time() - start}")

# pg = placement_group([read_bundle] * num, strategy="SPREAD")
# ray.wait([pg.ready()], timeout=0)
print("start shuffle()")
start = time.time()
# ds = ds.random_shuffle(placement_group=pg)  # second block
ds = ds.random_shuffle()  # second block
print(f"shuffle took {time.time() - start}")
pdb.set_trace()

# start = time.time()
# print("start iterbatches()")
# if args.num_workers == 1:
#     for i in ds.iter_batches(batch_size=args.batch_size):
#         pass
#     print(f"iter batches took {time.time() - start}")
# else:
#     splits = ds.split(args.num_workers)
#
#     @ray.remote(num_gpus=1, num_cpus=0)
#     def consume(split, rank=None, batch_size=None):
#         import time
#         start = time.time()
#         total_batches = 0
#         for i, x in enumerate(split.iter_batches(batch_size=batch_size)):
#             total_batches += len(x)
#         total = time.time() - start
#         print(f"Took {total} seconds. Got {total_batches} batches")
#         return
#
#     ray.get([consume.remote(s, batch_size=args.batch_size) for s in splits])
