from typing import List, Callable, Iterable, Generic, TypeVar

import ray
from ray.experimental.data.dataset import Dataset

T = TypeVar("T")
U = TypeVar("U")


@ray.remote
def wait(ds_fn: Callable[[], [Dataset]], num_async: int = float("inf"),
         submitted: List[ray.ObjectRef] = [], *args, **kwargs) -> Dataset:
    if len(submitted) >= num_async:
        ray.wait(submitted, num_returns=len(submitted) - num_async + 1,
                 fetch_to_local=False)
    ds = ds_fn(*args, **kwargs)
    ds.wait()
    return ds


class DatasetPipeline(Generic[T]):
    def __init__(self, dataset_futures: List[ray.ObjectRef[Dataset]],
                 num_async: int = None):
        self._datasets = dataset_futures
        self._num_async = (
            num_async if num_async is not None else float("inf"))

    @classmethod
    def from_datasets(cls, datasets: Iterable[Dataset], num_async: int = None):
        return cls([wait.remote(lambda: ds) for ds in datasets], num_async)

    @classmethod
    def from_iterable(
            cls, dataset_producers: Iterable[Callable[[], [Dataset]]],
            num_async: int = None):
        return cls([
            wait.remote(ds_fn) for ds_fn in dataset_producers], num_async)

    def _apply_op(self, fn_attr: str, *args, **kwargs) -> "DatasetPipeline":
        num_async = kwargs.pop("num_async", None) or self._num_async
        new_dses = []
        for ds in self._datasets:
            ds_fn = getattr(ds, fn_attr)
            new_ds = (
                ray.put(ds_fn(*args, **kwargs))
                if len(new_dses) < self._num_async
                else wait.remote(
                    ds_fn, num_async, new_dses, *args, **kwargs))
            new_dses.append(new_ds)
        return DatasetPipeline(new_dses)

    def map(self, *args, num_async=None, **kwargs) -> "DatasetPipeline":
        return self._apply_op("map", *args, num_async=num_async, **kwargs)

    def map_batches(
            self, *args, num_async: int = None, **kwargs) -> "DatasetPipeline":
        return self._apply_op(
            "map_batches", *args, num_async=num_async, **kwargs)

    def flat_map(
            self, *args, num_async: int = None, **kwargs) -> "DatasetPipeline":
        return self._apply_op(
            "flat_map", *args, num_async=num_async, **kwargs)

    def filter(self, *args, num_async=None, **kwargs) -> "DatasetPipeline":
        return self._apply_op("filter", *args, num_async=num_async, **kwargs)
