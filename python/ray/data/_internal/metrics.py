from abc import ABC
from dataclasses import dataclass, fields
import operator

from typing import Dict, Callable


@dataclass
class Metrics(ABC):
    def merge_with(self, other: "Metrics"):
        if not isinstance(other, type(self)):
            raise ValueError(f"Expected {type(self)}, got: {type(other)}")

        merged_metrics = {}
        for metric_field in fields(other):
            metric_name = metric_field.name
            self_metric = getattr(self, metric_name)
            other_metric = getattr(other, metric_name)
            if isinstance(self_metric, Metrics):
                assert isinstance(other_metric, Metrics)
                new_metric = self_metric.merge_with(other_metric)
            else:
                metric_merger = self.get_metric_merger(metric_name)
                new_metric = metric_merger(self_metric, other_metric)
            merged_metrics[metric_name] = new_metric
        return type(other)(**merged_metrics)

    def get_metric_merger(self, metric: str) -> Callable[[int, int], int]:
        return operator.add

    def to_metrics_dict(self) -> Dict[str, int]:
        metrics = {}
        for metric in fields(self):
            metric_value = getattr(self, metric.name)
            if isinstance(metric_value, Metrics):
                metric_value = metric_value.to_metrics_dict()
            metrics[metric.name] = metric_value
        return metrics


@dataclass
class DataMungingMetrics(Metrics):
    num_copies: int = 0
    num_rows_copied: int = 0
    num_format_conversions: int = 0
    num_slices: int = 0
    num_concatenations: int = 0


class MetricsCollector:
    def __init__(self):
        self._metrics = DataMungingMetrics()

    def record_metrics(self, metrics: DataMungingMetrics):
        self._metrics = self._metrics.merge_with(metrics)

    def get_metrics(self) -> DataMungingMetrics:
        return self._metrics


@dataclass
class ObjectStoreMetrics(Metrics):
    """Metrics for object store memory allocations."""

    alloc: int
    freed: int
    cur: int
    peak: int

    def get_metric_merger(self, metric: str) -> Callable[[int, int], int]:
        if metric == "peak":
            return max
        else:
            return operator.add

    def to_metrics_dict(self) -> Dict[str, int]:
        return {
            "obj_store_mem_alloc": self.alloc,
            "obj_store_mem_freed": self.freed,
            "obj_store_mem_peak": self.peak,
        }
