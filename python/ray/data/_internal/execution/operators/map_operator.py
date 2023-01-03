from typing import List, Iterator, Any, Dict, Callable, Optional

import ray
from ray.data.block import Block, BlockMetadata
from ray.data._internal.stats import StatsDict
from ray.data._internal.compute import (
    ComputeStrategy,
    TaskPoolStrategy,
    ActorPoolStrategy,
)
from ray.data._internal.execution.interfaces import (
    RefBundle,
    PhysicalOperator,
)
from ray.data._internal.execution.operators.map_operator_tasks_impl import (
    MapOperatorTasksImpl,
)
from ray.data._internal.execution.operators.map_operator_actors_impl import (
    MapOperatorActorsImpl,
)


class MapOperator(PhysicalOperator):
    """A streaming operator that maps input bundles 1:1 to output bundles.

    This operator implements the distributed map operation, supporting both task
    and actor compute strategies.
    """

    def __init__(
        self,
        transform_fn: Callable[[Iterator[Block]], Iterator[Block]],
        input_op: PhysicalOperator,
        name: str = "Map",
        compute_strategy: Optional[ComputeStrategy] = None,
        min_rows_per_batch: Optional[int] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ):
        """Create a MapOperator.

        Args:
            transform_fn: The function to apply to each ref bundle input.
            input_op: Operator generating input data for this op.
            name: The name of this operator.
            compute_strategy: Customize the compute strategy for this op.
            min_rows_per_batch: The number of rows to gather per batch passed to the
                transform_fn, or None to use the block size. Setting the batch size is
                important for the performance of GPU-accelerated transform functions.
                The actual rows passed may be less if the dataset is small.
            ray_remote_args: Customize the ray remote args for this op's tasks.
        """
        self._transform_fn = transform_fn
        self._strategy = compute_strategy or TaskPoolStrategy()
        self._remote_args = (ray_remote_args or {}).copy()
        self._output_metadata: List[BlockMetadata] = []
        self._min_rows_per_batch = min_rows_per_batch
        if isinstance(self._strategy, TaskPoolStrategy):
            self._execution_state = MapOperatorTasksImpl(self)
        elif isinstance(self._strategy, ActorPoolStrategy):
            self._execution_state = MapOperatorActorsImpl(self)
        else:
            raise ValueError(f"Unsupported execution strategy {self._strategy}")
        super().__init__(name, [input_op])

    def get_transform_fn(
        self,
    ) -> Callable[[Iterator[Block]], Iterator[Block]]:
        """Return the block transformation to run on a worker process.

        This callable must be serializable as it will be sent to remote processes.

        Returns:
            A callable taking an iterator over input blocks of a RefBundle. Typically,
            this will yield only a single block, unless the transformation has
            multiple inputs. It is an iterator for memory efficiency.
        """
        return self._transform_fn

    # TODO(ekl): slim down ComputeStrategy to only specify the compute
    # config and not contain implementation code.
    def compute_strategy(self) -> ComputeStrategy:
        """Return the compute strategy to use for executing these tasks.

        Supported strategies: {TaskPoolStrategy, ActorPoolStrategy}.
        """
        return self._strategy

    def ray_remote_args(self) -> Dict[str, Any]:
        """Return extra ray remote args to use for execution."""
        return self._remote_args

    def get_metrics(self) -> Dict[str, int]:
        return {
            "obj_store_mem_alloc": self._execution_state._obj_store_mem_alloc,
            "obj_store_mem_freed": self._execution_state._obj_store_mem_freed,
            "obj_store_mem_peak": self._execution_state._obj_store_mem_peak,
        }

    def add_input(self, refs: RefBundle, input_index: int) -> None:
        assert input_index == 0, input_index
        self._execution_state.add_input(refs)

    def inputs_done(self, input_index: int) -> None:
        self._execution_state.inputs_done(input_index)

    def has_next(self) -> bool:
        return self._execution_state.has_next()

    def get_next(self) -> RefBundle:
        bundle = self._execution_state.get_next()
        for _, meta in bundle.blocks:
            self._output_metadata.append(meta)
        return bundle

    def get_work_refs(self) -> List[ray.ObjectRef]:
        return self._execution_state.get_work_refs()

    def notify_work_completed(self, task: ray.ObjectRef) -> None:
        self._execution_state.work_completed(task)

    def get_stats(self) -> StatsDict:
        return {self._name: self._output_metadata}

    def shutdown(self) -> None:
        self._execution_state.shutdown()
