from typing import List, Iterator, Any, Dict, Callable, Optional

import ray
from ray.data.block import Block
from ray.data._internal.compute import (
    ComputeStrategy,
    TaskPoolStrategy,
    ActorPoolStrategy,
)
from ray.data._internal.execution.interfaces import (
    RefBundle,
    PhysicalOperator,
)
from ray.data._internal.execution.operators.map_operator_task_impl import (
    MapOperatorTaskImpl,
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
        transform_fn: Callable[[Iterator[Block], Dict], Iterator[Block]],
        input_op: PhysicalOperator,
        name: str = "Map",
        compute_strategy: Optional[ComputeStrategy] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ):
        """Create a MapOperator.

        Args:
            transform_fn: The function to apply to each ref bundle input.
            input_op: Operator generating input data for this op.
            name: The name of this operator.
            compute_strategy: Customize the compute strategy for this op.
            ray_remote_args: Customize the ray remote args for this op's tasks.
        """
        self._transform_fn = transform_fn
        self._strategy = compute_strategy or TaskPoolStrategy()
        self._remote_args = (ray_remote_args or {}).copy()
        if isinstance(self._strategy, TaskPoolStrategy):
            self._execution_state = MapOperatorTaskImpl(self)
        elif isinstance(self._strategy, ActorPoolStrategy):
            self._execution_state = MapOperatorActorsImpl(self)
        else:
            raise NotImplementedError
        super().__init__(name, [input_op])

    def get_transform_fn(
        self,
    ) -> Callable[[Iterator[Block], Dict[str, Any]], Iterator[Block]]:
        """Return the block transformation to run on a worker process.

        This callable must be serializable as it will be sent to remote processes.

        Returns:
            A callable taking the following inputs:
                block_bundle: Iterator over input blocks of a RefBundle. Typically,
                    this will yield only a single block, unless the transformation has
                    multiple inputs, e.g., in the SortReduce or ZipBlocks cases. It is
                    an iterator instead of a list for memory efficiency.
                input_metadata: Extra metadata provided from the upstream operator.
        """
        return self._transform_fn

    def compute_strategy(self) -> ComputeStrategy:
        """Return the compute strategy to use for executing these tasks.

        Supported strategies: {TaskPoolStrategy, ActorPoolStrategy}.
        """
        return self._strategy

    def ray_remote_args(self) -> Dict[str, Any]:
        """Return extra ray remote args to use for execution."""
        return self._remote_args

    def add_input(self, refs: RefBundle, input_index: int) -> None:
        assert input_index == 0, input_index
        self._execution_state.add_input(refs)

    def has_next(self) -> bool:
        return self._execution_state.has_next()

    def get_next(self) -> RefBundle:
        return self._execution_state.get_next()

    def get_tasks(self) -> List[ray.ObjectRef]:
        return self._execution_state.get_tasks()

    def notify_task_completed(self, task: ray.ObjectRef) -> None:
        self._execution_state.task_completed(task)
