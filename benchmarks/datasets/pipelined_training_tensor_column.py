import argparse
import glob
import json
import math
import os
import ray
import time
import timeit

import numpy as np
import torch
import horovod.torch as hvd
from horovod.ray import RayExecutor

from data_generation import generate_data, DATA_SPEC_TENSOR_COLS, TENSOR_COLS
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.impl.torch_iterable_dataset import TorchIterableDataset

DEFAULT_DATA_DIR = "s3://shuffling-data-loader-benchmarks/tensor_col_data"

# Training settings
parser = argparse.ArgumentParser(
    description="Datasets ML ingest with tensor columns")
parser.add_argument(
    "--num-rows",
    type=int,
    default=20_000_000,
    help="Number of rows in the generated dataset (default: 2M)")
parser.add_argument(
    "--num-files",
    type=int,
    default=400,
    help="Number of files in the generated dataset (default: 400)")
parser.add_argument(
    "--num-row-groups-per-file",
    type=int,
    default=1,
    help="Number of row groups per file in the generated dataset (default: 1)")
parser.add_argument(
    "--batch-size",
    type=int,
    default=250000,
    metavar="N",
    help="input batch size for training (default: 64)")
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)")
parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
parser.add_argument("--clear-old-data", action="store_true")
parser.add_argument("--use-old-data", action="store_true")
parser.add_argument(
    "--debug", action="store_true", default=False, help="disables hvd")
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed (default: 42)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help=("how many batches to wait before logging training "
          "status"))
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--mock-train-step-time", type=float, default=1.0)
parser.add_argument("--num-windows", type=int, default=1)


def train_main(args, splits, num_bytes):
    # Horovod: initialize library.
    hvd.init()
    num_epochs = args.epochs
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)
    rank = hvd.rank()

    samples_seen = 0

    start = timeit.default_timer()

    def _train(epoch, train_dataset):
        nonlocal samples_seen
        start_epoch = timeit.default_timer()
        last_batch_time = start_epoch
        batch_wait_times = []
        to_gpu_times = []
        for batch_idx, (data, target) in enumerate(train_dataset):
            start_batch = timeit.default_timer()
            batch_wait_times.append(start_batch - last_batch_time)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                to_gpu_times.append(timeit.default_timer() - start_batch)

            if batch_idx % args.log_interval == 0:
                print(
                    f"Processing batch {batch_idx} in epoch {epoch} on worker "
                    f"{rank}.")
            # time.sleep(args.mock_train_step_time)
            samples_seen += len(data)

            last_batch_time = timeit.default_timer()
        epoch_duration = timeit.default_timer() - start_epoch
        avg_batch_wait_time = np.mean(batch_wait_times)
        std_batch_wait_time = np.std(batch_wait_times)
        max_batch_wait_time = np.max(batch_wait_times)
        min_batch_wait_time = np.min(batch_wait_times)
        print(f"\nEpoch {epoch}, worker {rank} stats over "
              f"{len(batch_wait_times)} steps: {epoch_duration:.3f}")
        print(f"Mean batch wait time: {avg_batch_wait_time:.3f}s +- "
              f"{std_batch_wait_time}")
        print(f"Max batch wait time: {max_batch_wait_time:.3f}s")
        print(f"Min batch wait time: {min_batch_wait_time:.3f}s")
        return batch_wait_times, to_gpu_times

    print(f"Starting training on worker {rank}.")
    batch_wait_times = []
    to_gpu_times = []
    for epoch, split_ds in enumerate(splits[rank].iter_datasets()):
        train_dataset = create_torch_iterator(split_ds, args.batch_size, rank)
        new_batch_times, new_to_gpu_times = _train(epoch, train_dataset)
        new_batch_times.pop(0)
        batch_wait_times.extend(new_batch_times)
        to_gpu_times.extend(new_to_gpu_times)
    end = timeit.default_timer()
    duration = end - start
    print(f"Done training on worker {rank}:\n"
          f"{samples_seen} samples processed in {duration} seconds.")
    avg_batch_wait_time = np.mean(batch_wait_times)
    std_batch_wait_time = np.std(batch_wait_times)
    max_batch_wait_time = np.max(batch_wait_times)
    min_batch_wait_time = np.min(batch_wait_times)
    print(f"\nWorker {rank} training stats over {args.epochs} epochs:")
    print(f"Mean batch wait time: {avg_batch_wait_time:.3f}s +- "
          f"{std_batch_wait_time}")
    print(f"Max batch wait time: {max_batch_wait_time:.3f}s")
    print(f"Min batch wait time: {min_batch_wait_time:.3f}s")
    throughput_rows = samples_seen / time
    duration_list = hvd.allgather_object(duration, name="duration")
    throughput_rows_list = hvd.allgather_object(
        throughput_rows, name="throughput_rows")
    to_gpu_time_list = hvd.allgather_object(to_gpu_times, name="to_gpu_times")
    if rank == 0:
        agg_throughput_rows = sum(throughput_rows_list)
        avg_throughput_rows = agg_throughput_rows / hvd.size()
        avg_throughput_bytes = (
            num_bytes * num_epochs / (sum(duration_list) / 1024.0**3))
        agg_throughput_bytes = avg_throughput_bytes * hvd.size()
        print(f"\n\nAvg training row throughput:"
              f"\n\t{avg_throughput_rows} rows/sec")
        print(f"Aggregate training row throughput:"
              f"\n\t{agg_throughput_rows} rows/sec")
        print(f"Average training bytes throughput:"
              f"\n\t{avg_throughput_bytes} GiB/sec")
        print(f"Aggregate training bytes throughput:"
              f"\n\t{agg_throughput_bytes} GiB/sec")
        avg_to_gpu_time = np.mean(to_gpu_time_list)
        std_to_gpu_time = np.std(to_gpu_time_list)
        print(f"Mean to GPU time: {avg_to_gpu_time:.3f}s +- "
              f"{std_to_gpu_time}")


######################################################

numpy_to_torch_dtype = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}


def create_torch_iterator(split, batch_size, rank=None):
    print(f"Creating Torch shuffling dataset for worker {rank} with "
          f"{batch_size} batch size.")
    columns, column_dtypes = zip(*DATA_SPEC_TENSOR_COLS.items())
    feature_columns = columns[:-1]
    feature_column_dtypes = column_dtypes[:-1]
    label_column = columns[-1]
    label_column_dtype = column_dtypes[-1]

    def make_generator():
        for epoch_ds in split.iter_epochs():
            for batch in epoch_ds.iter_batches(
                    batch_size=batch_size, batch_format="pandas"):
                label_vals = batch.pop(label_column).values
                label_tensor = torch.as_tensor(
                    label_vals, dtype=label_column_dtype)
                feature_tensors = []
                if feature_columns:
                    batch = batch[feature_columns]
                if feature_column_dtypes:
                    dtypes = feature_column_dtypes
                else:
                    dtypes = [None] * len(batch.columns)

                for col, dtype in zip(batch.columns, dtypes):
                    col_vals = batch[col].values
                    if col in TENSOR_COLS:
                        col_vals = np.asarray(col_vals)
                    t = torch.as_tensor(col_vals, dtype=dtype)
                    feature_tensors.append(t)

                yield feature_tensors, label_tensor

    return TorchIterableDataset(make_generator)


def create_dataset(files, num_workers=4, epochs=50, num_windows=1):
    if num_windows > 1:
        # num_rows = ray.data.read_parquet(
        #     files, _spread_resource_prefix="node:").count(
        #     )  # This should only read Parquet metadata.
        file_splits = np.array_split(files, num_windows)

        class Windower:
            def __init__(self):
                self.i = 0
                self.iterations = epochs * num_windows

            def __iter__(self):
                return self

            def __next__(self):
                if self.i >= self.iterations:
                    raise StopIteration()
                split = file_splits[self.i % num_windows]
                self.i += 1
                return lambda: ray.data.read_parquet(
                    list(split), _spread_resource_prefix="node:")

        pipe = DatasetPipeline.from_iterable(Windower())
        # split_indices = [
        #     i * num_rows // num_windows // num_workers
        #     for i in range(1, num_workers)
        # ]
        pipe = pipe.random_shuffle_each_window(_spread_resource_prefix="node:")
        # pipe_shards = pipe.split_at_indices(split_indices)
        pipe_shards = pipe.split(num_workers, equal=True)
    else:
        ds = ray.data.read_parquet(files, _spread_resource_prefix="node:")
        pipe = ds.repeat(epochs)
        pipe = pipe.random_shuffle_each_window(_spread_resource_prefix="node:")
        pipe_shards = pipe.split(num_workers, equal=True)
    return pipe_shards


@ray.remote
def consume(split, rank=None, batch_size=None):
    torch_iterator = create_torch_iterator(
        split, batch_size=batch_size, rank=rank)
    for i, (x, y) in enumerate(torch_iterator):
        if i % 10 == 0:
            print(i)
    return


UNITS = ["", "K", "M", "B", "T", "Q"]


def human_readable_big_num(num):
    idx = int(math.log10(num) // 3)
    unit = UNITS[idx]
    new_num = num / 10**(3 * idx)
    if new_num % 1 == 0:
        return f"{int(new_num)}{unit}"
    else:
        return f"{new_num:.1f}{unit}"


def human_readable_size(num, precision=1, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0 or unit == "Zi":
            break
        num /= 1024.0
    return f"{num:.{precision}f}{unit}{suffix}"


if __name__ == "__main__":
    args = parser.parse_args()
    import ray
    print("Connecting to Ray cluster...")
    ray.init(address="auto")
    num_rows = args.num_rows
    num_row_groups_per_file = args.num_row_groups_per_file
    num_files = args.num_files
    max_row_group_skew = 0

    if args.clear_old_data and args.use_old_data:
        raise ValueError(
            "Only one of --clear-old-data and --use-old-data should be "
            "specified.")

    data_dir = args.data_dir
    if args.clear_old_data:
        print(f"Clearing old data from {data_dir}.")
        files = glob.glob(os.path.join(data_dir, "*.parquet.snappy"))
        # Assuming that these are local files.
        for f in files:
            os.remove(f)

    if not args.use_old_data:
        print(f"Generating {num_rows} rows over {num_files} files, with "
              f"{num_row_groups_per_file} row groups per file and at most "
              f"{100 * max_row_group_skew:.1f}% row group skew.")
        filenames, num_bytes = generate_data(num_rows, num_files,
                                             num_row_groups_per_file,
                                             max_row_group_skew, data_dir)
        print(f"Generated {len(filenames)} files containing {num_rows} rows "
              f"with {num_row_groups_per_file} row groups per file, totalling "
              f"{human_readable_size(num_bytes)}.")
    else:
        filenames = [
            os.path.join(data_dir, f"input_data_{file_index}.parquet.snappy")
            for file_index in range(num_files)
        ]
        print("Not generating input data, using existing data instead.")

    start = time.time()

    splits = create_dataset(
        filenames,
        num_workers=args.num_workers,
        epochs=args.epochs,
        num_windows=args.num_windows)

    if args.debug:
        tasks = [
            consume.options(num_gpus=1).remote(
                split, rank=idx, batch_size=args.batch_size)
            for idx, split in enumerate(splits)
        ]
        ray.get(tasks)
    else:
        print("Create Ray executor")
        settings = RayExecutor.create_settings(timeout_s=30)
        executor = RayExecutor(
            settings, num_workers=args.num_workers, use_gpu=True)
        executor.start()
        executor.run(train_main, args=[args, splits, num_bytes])
        executor.shutdown()

    delta = time.time() - start
    print(f"success! total time {delta}")
    with open(os.environ["TEST_OUTPUT_JSON"], "w") as f:
        f.write(json.dumps({"ingest_time": delta, "success": 1}))
