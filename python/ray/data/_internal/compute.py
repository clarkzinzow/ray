import collections
import logging
import math
from typing import (
    Any, Set, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
)

import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
    BatchUDF,
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockMetadata,
    BlockPartition,
    CallableClass,
    RowUDF,
)
from ray.data.context import DEFAULT_SCHEDULING_STRATEGY, DatasetContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

# Block transform function applied by task and actor pools.
BlockTransform = Union[
    # TODO(Clark): Once Ray only supports Python 3.8+, use protocol to constrain block
    # transform type.
    # Callable[[Block, ...], Iterable[Block]]
    # Callable[[Block, BatchUDF, ...], Iterable[Block]],
    Callable[[Iterable[Block]], Iterable[Block]],
    Callable[[Iterable[Block], Union[BatchUDF, RowUDF]], Iterable[Block]],
    Callable[..., Iterable[Block]],
]

# UDF on a batch or row.
UDF = Union[BatchUDF, RowUDF]


@DeveloperAPI
class ComputeStrategy:
    def _apply(
        self,
        block_fn: BlockTransform,
        remote_args: dict,
        blocks: BlockList,
        clear_input_blocks: bool,
    ) -> BlockList:
        raise NotImplementedError


@DeveloperAPI
class TaskPoolStrategy(ComputeStrategy):
    def _apply(
        self,
        block_fn: BlockTransform,
        remote_args: dict,
        block_list: BlockList,
        clear_input_blocks: bool,
        name: Optional[str] = None,
        target_block_size: Optional[int] = None,
        fn: Optional[UDF] = None,
        fn_args: Optional[Iterable[Any]] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
        fn_constructor_args: Optional[Iterable[Any]] = None,
        fn_constructor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BlockList:
        assert fn_constructor_args is None and fn_constructor_kwargs is None
        if fn_args is None:
            fn_args = tuple()
        if fn_kwargs is None:
            fn_kwargs = {}

        context = DatasetContext.get_current()

        # Handle empty datasets.
        if block_list.initial_num_blocks() == 0:
            return block_list

        if name is None:
            name = "map"
        blocks = block_list.get_blocks_with_metadata()
        # Bin blocks by target block size.
        if target_block_size is not None:
            _check_batch_size(blocks, target_block_size, name)
            block_bundles = _bundle_blocks_up_to_size(blocks, target_block_size, name)
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks]
        del blocks
        name = name.title()
        map_bar = ProgressBar(name, total=len(block_bundles))

        if context.block_splitting_enabled:
            map_block = cached_remote_fn(_map_block_split).options(
                num_returns="dynamic", **remote_args
            )
            refs = [
                map_block.remote(
                    block_fn,
                    [f for m in ms for f in m.input_files],
                    fn,
                    len(bs),
                    *(bs + fn_args),
                    **fn_kwargs,
                )
                for bs, ms in block_bundles
            ]
        else:
            map_block = cached_remote_fn(_map_block_nosplit).options(
                **dict(remote_args, num_returns=2)
            )
            all_refs = [
                map_block.remote(
                    block_fn,
                    [f for m in ms for f in m.input_files],
                    fn,
                    len(bs),
                    *(bs + fn_args),
                    **fn_kwargs,
                )
                for bs, ms in block_bundles
            ]
            data_refs, refs = map(list, zip(*all_refs))

        in_block_owned_by_consumer = block_list._owned_by_consumer
        # Release input block references.
        if clear_input_blocks:
            del block_bundles
            block_list.clear()

        # Common wait for non-data refs.
        try:
            results = map_bar.fetch_until_complete(refs)
        except (ray.exceptions.RayTaskError, KeyboardInterrupt) as e:
            # One or more mapper tasks failed, or we received a SIGINT signal
            # while waiting; either way, we cancel all map tasks.
            for ref in refs:
                ray.cancel(ref)
            # Wait until all tasks have failed or been cancelled.
            for ref in refs:
                try:
                    ray.get(ref)
                except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError):
                    pass
            # Reraise the original task failure exception.
            raise e from None

        new_blocks, new_metadata = [], []
        if context.block_splitting_enabled:
            for ref_generator in results:
                refs = list(ref_generator)
                metadata = ray.get(refs.pop(-1))
                assert len(metadata) == len(refs)
                new_blocks += refs
                new_metadata += metadata
        else:
            for block, metadata in zip(data_refs, results):
                new_blocks.append(block)
                new_metadata.append(metadata)
        return BlockList(
            list(new_blocks),
            list(new_metadata),
            owned_by_consumer=in_block_owned_by_consumer,
        )


@PublicAPI
class ActorPoolStrategy(ComputeStrategy):
    """Specify the compute strategy for a Dataset transform.

    ActorPoolStrategy specifies that an autoscaling pool of actors should be used
    for a given Dataset transform. This is useful for stateful setup of callable
    classes.

    To autoscale from ``m`` to ``n`` actors, specify
    ``compute=ActorPoolStrategy(m, n)``.
    For a fixed-sized pool of size ``n``, specify ``compute=ActorPoolStrategy(n, n)``.

    To increase opportunities for pipelining task dependency prefetching with
    computation and avoiding actor startup delays, set max_tasks_in_flight_per_actor
    to 2 or greater; to try to decrease the delay due to queueing of tasks on the worker
    actors, set max_tasks_in_flight_per_actor to 1.
    """

    def __init__(
        self,
        min_size: int = 1,
        max_size: Optional[int] = None,
        max_tasks_in_flight_per_actor: Optional[int] = 2,
    ):
        """Construct ActorPoolStrategy for a Dataset transform.

        Args:
            min_size: The minimum size of the actor pool. The actor pool size is also
                bounded from above by the number of blocks being transformed, which
                takes precedence over min_size if smaller.
            max_size: The maximum size of the actor pool. The actor pool size is also
                bounded from above by the number of blocks being transformed, which
                takes precedence over max_size if smaller.
            max_tasks_in_flight_per_actor: The maximum number of tasks to concurrently
                send to a single actor worker. Increasing this will increase
                opportunities for pipelining task dependency prefetching with
                computation and avoiding actor startup delays, but will also increase
                queueing delay.
        """
        if min_size < 1:
            raise ValueError("min_size must be > 1", min_size)
        if max_size is not None and min_size > max_size:
            raise ValueError("min_size must be <= max_size", min_size, max_size)
        if max_tasks_in_flight_per_actor < 1:
            raise ValueError(
                "max_tasks_in_flight_per_actor must be >= 1, got: ",
                max_tasks_in_flight_per_actor,
            )
        self.min_size = min_size
        self.max_size = max_size or float("inf")
        self.max_tasks_in_flight_per_actor = max_tasks_in_flight_per_actor
        # High water mark for the pool.
        self.max_num_workers = 0
        self.ready_to_total_workers_ratio = 0.8

    def _apply(
        self,
        block_fn: BlockTransform,
        remote_args: dict,
        block_list: BlockList,
        clear_input_blocks: bool,
        name: Optional[str] = None,
        target_block_size: Optional[int] = None,
        fn: Optional[UDF] = None,
        fn_args: Optional[Iterable[Any]] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
        fn_constructor_args: Optional[Iterable[Any]] = None,
        fn_constructor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BlockList:
        """Note: this is not part of the Dataset public API."""
        if fn_args is None:
            fn_args = tuple()
        if fn_kwargs is None:
            fn_kwargs = {}
        if fn_constructor_args is None:
            fn_constructor_args = tuple()
        if fn_constructor_kwargs is None:
            fn_constructor_kwargs = {}

        if name is None:
            name = "map"
        blocks_in = block_list.get_blocks_with_metadata()
        # Bin blocks by target block size.
        if target_block_size is not None:
            _check_batch_size(blocks_in, target_block_size, name)
            block_bundles = _bundle_blocks_up_to_size(
                blocks_in, target_block_size, name
            )
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks_in]
        del blocks_in
        owned_by_consumer = block_list._owned_by_consumer

        # Early release block references.
        if clear_input_blocks:
            block_list.clear()

        orig_num_blocks = len(block_bundles)
        # Bound actor pool size by maximum parallel map task submission, which is
        # equivalent to the number of input blocks adjusted for the max concurrently
        # submitted in-flight tasks per worker.
        pool_upper_bound = max(orig_num_blocks // self.max_tasks_in_flight_per_actor, 1)
        max_size = min(pool_upper_bound, self.max_size)
        min_size = min(pool_upper_bound, self.min_size)
        results = []
        name = name.title()
        map_bar = ProgressBar(name, total=orig_num_blocks)

        class BlockWorker:
            def __init__(
                self,
                *fn_constructor_args: Any,
                **fn_constructor_kwargs: Any,
            ):
                if not isinstance(fn, CallableClass):
                    if fn_constructor_args or fn_constructor_kwargs:
                        raise ValueError(
                            "fn_constructor_{kw}args only valid for CallableClass "
                            f"UDFs, but got: {fn}"
                        )
                    self.fn = fn
                else:
                    self.fn = fn(*fn_constructor_args, **fn_constructor_kwargs)

            def ready(self):
                return "ok"

            def map_block_split(
                self,
                input_files: List[str],
                num_blocks: int,
                *blocks_and_fn_args,
                **fn_kwargs,
            ) -> BlockPartition:
                return _map_block_split(
                    block_fn,
                    input_files,
                    self.fn,
                    num_blocks,
                    *blocks_and_fn_args,
                    **fn_kwargs,
                )

            @ray.method(num_returns=2)
            def map_block_nosplit(
                self,
                input_files: List[str],
                num_blocks: int,
                *blocks_and_fn_args,
                **fn_kwargs,
            ) -> Tuple[Block, BlockMetadata]:
                return _map_block_nosplit(
                    block_fn,
                    input_files,
                    self.fn,
                    num_blocks,
                    *blocks_and_fn_args,
                    **fn_kwargs,
                )

        if "num_cpus" not in remote_args:
            remote_args["num_cpus"] = 1

        if "scheduling_strategy" not in remote_args:
            ctx = DatasetContext.get_current()
            if ctx.scheduling_strategy == DEFAULT_SCHEDULING_STRATEGY:
                remote_args["scheduling_strategy"] = "SPREAD"
            else:
                remote_args["scheduling_strategy"] = ctx.scheduling_strategy

        BlockWorker = ray.remote(BlockWorker).options(**remote_args)
        workers = {
            BlockWorker.remote(*fn_constructor_args, **fn_constructor_kwargs)
            for _ in range(min_size)
        }
        self.max_num_workers = len(workers)
        tasks = {w.ready.remote(): w for w in workers}
        tasks_in_flight = collections.defaultdict(int)
        metadata_mapping = {}
        block_indices = {}
        ready_workers = set()

        try:
            while len(results) < orig_num_blocks:
                ready, _ = ray.wait(
                    list(tasks.keys()), timeout=0.01, num_returns=1, fetch_local=False
                )
                if not ready:
                    # Only create a new worker if:
                    if (
                        # 1. The actor pool size does not exceed the configured max
                        # size.
                        len(workers) < max_size
                        # 2. At least 80% of the workers in the pool have already
                        # started. This will ensure that workers will be launched in
                        # parallel while bounding the worker pool to requesting 125% of
                        # the cluster's available resources. The excess workers will be
                        # scaled down by the policy given at the end of this autoscaling
                        # loop.
                        and len(ready_workers) / len(workers)
                        > self.ready_to_total_workers_ratio
                        # 3. There will be work for the new worker to do. Specifically,
                        # whether there are not enough (still-initializing) workers
                        # to transform the remaining blocks in the queue in parallel,
                        # adjusted for the number of tasks we try to concurrently send
                        # to each worker.
                        # NOTE: This bound assumes that the existing workers won't
                        # process any of the remaining blocks, and therefore may cause
                        # us to over-allocate workers, but we'd rather spin up a few
                        # unnecessary workers than risk blocking the work queue.
                        and (
                            len(workers) - len(ready_workers)
                            < len(block_bundles) // self.max_tasks_in_flight_per_actor
                        )
                    ):
                        w = BlockWorker.remote(
                            *fn_constructor_args, **fn_constructor_kwargs
                        )
                        tasks[w.ready.remote()] = w
                        workers.add(w)
                        # Update pool's high water mark.
                        self.max_num_workers = max(len(workers), self.max_num_workers)
                        map_bar.set_description(
                            "Map Progress ({} actors {} pending)".format(
                                len(ready_workers), len(workers) - len(ready_workers)
                            )
                        )
                    continue

                [obj_id] = ready
                worker = tasks.pop(obj_id)

                # Process task result.
                if worker in ready_workers:
                    results.append(obj_id)
                    tasks_in_flight[worker] -= 1
                    map_bar.update(1)
                else:
                    ready_workers.add(worker)
                    map_bar.set_description(
                        "Map Progress ({} actors {} pending)".format(
                            len(ready_workers), len(workers) - len(ready_workers)
                        )
                    )

                # Schedule a new task.
                while (
                    block_bundles
                    and tasks_in_flight[worker] < self.max_tasks_in_flight_per_actor
                ):
                    blocks, metas = block_bundles.pop()
                    # TODO(swang): Support block splitting for compute="actors".
                    ref, meta_ref = worker.map_block_nosplit.remote(
                        [f for meta in metas for f in meta.input_files],
                        len(blocks),
                        *(blocks + fn_args),
                        **fn_kwargs,
                    )
                    metadata_mapping[ref] = meta_ref
                    tasks[ref] = worker
                    block_indices[ref] = len(block_bundles)
                    tasks_in_flight[worker] += 1

                # Try to scale down if there's no more work to do and pending/idle
                # workers.
                if not block_bundles:
                    should_update = False
                    if tasks_in_flight[worker] == 0:
                        should_update = True
                        # No more work to do and worker is idle, so we terminate the
                        # worker.
                        ready_workers.remove(worker)
                        workers.remove(worker)
                        # Explicitly terminate the actor to expedite cleanup.
                        ray.kill(worker)
                    # Terminate all pending workers, since we now know that they will
                    # have no work to do. We try to eagerly terminate these pending
                    # workers in order to:
                    #  1. Free up resources they would otherwise acquire.
                    #  2. Decrease Ray's scheduling load.
                    #  3. Account for resource-bounded case where the actors will never
                    #  be scheduled, in which case the actor creation will always be
                    #  pending util the pool is destroyed.
                    workers_to_terminate = workers - ready_workers
                    if workers_to_terminate:
                        # NOTE: This will run at most once in a pool's lifetime.
                        should_update = True
                        workers_to_ready_tasks = {}
                        for task, worker in tasks.items():
                            if worker in workers_to_terminate:
                                # Worker is pending and should therefore only have a
                                # single "ready" task in the task map.
                                assert worker not in workers_to_ready_tasks
                                workers_to_ready_tasks[worker] = task
                        for worker_ in workers_to_terminate:
                            workers.remove(worker_)
                            del tasks[workers_to_ready_tasks[worker_]]
                            # Explicitly terminate the actor to expedite cleanup.
                            ray.kill(worker_)
                    if should_update:
                        map_bar.set_description(
                            "Map Progress ({} actors {} pending)".format(
                                len(ready_workers), len(workers) - len(ready_workers)
                            )
                        )

            map_bar.close()
            new_blocks, new_metadata = [], []
            # Put blocks in input order.
            results.sort(key=block_indices.get)
            # TODO(swang): Support block splitting for compute="actors".
            for block in results:
                new_blocks.append(block)
                new_metadata.append(metadata_mapping[block])
            new_metadata = ray.get(new_metadata)
            return BlockList(
                new_blocks, new_metadata, owned_by_consumer=owned_by_consumer
            )

        except Exception as e:
            try:
                for worker in workers:
                    ray.kill(worker)
            except Exception as err:
                logger.exception(f"Error killing workers: {err}")
            finally:
                raise e


# Autoscaling actor pool abstractions:
#  - autoscaling config: configuration for policy
#  - autoscaling policy: how many workers to prestart, whether we should scasle up when
#  no workers are available after wait timeout, whether we should scale down after
#  the work queue decreases in size.
#  - worker pool manager: managing pool of workers, i.e. adding a worker, leasing a worker for
#  execution, returning a worker to the pool, killing a worker
#  - worker: handles worker creation and destruction, work submission
#  - worker pool executor: runs autoscaling + work submission loop


class ActorPoolAutoscalingConfig:
    min_workers: int
    max_workers: int
    ready_to_total_workers_ratio: int
    wait_timeout: float


class ActorPoolAutoscalingPolicy:
    def __init__(self, config: ActorPoolAutoscalingConfig):
        self.config = config

    def num_workers_to_prestart(self):
        pass

    def should_scale_up(
        self, num_pending_workers, num_running_workers, work_queue_size
    ):
        pass

    def should_scale_down_worker(self, worker, work_queue_size):
        return work_queue_size == 0 and worker.tasks_in_flight == 0

    def should_scale_down_pending_workers(self, work_queue_size):
        return work_queue_size == 0

    def scale_down_if_needed(self, actor_pool, work_queue_size, worker):
        if work_queue_size == 0:
            if worker.tasks_in_flight == 0:
                actor_pool.kill_worker(worker)
            actor_pool.kill_pending_workers()

    def get_wait_timeout(self):
        return self.config.wait_timeout


class TaskResult:
    @abstractmethod
    def get_result_future(self):
        raise NotImplementedError()


@dataclass
class TaskResult:
    result_future: ray.ObjectRef
    metadata: BlockMetadata
    order_index: int


class Worker(ABC):
    @abstractmethod
    def submit(self, work: Work) -> Result:
        pass

    @abstractmethod
    def task_done(self):
        pass

    @abstractmethod
    def kill(self):
        pass

    @property
    @abstractmethod
    def ready_future(self) -> ray.ObjectRef:
        pass


class BlockWorker:
    def __init__(self, fn, fn_constructor_args, fn_constructor_kwargs):
        self._worker = _BlockWorker.remote(
            fn, *fn_constructor_args, **fn_constructor_kwargs
        )
        self._ready_future = self._worker.ready.remote()

    @property
    def ready_future(self) -> ray.ObjectRef:
        return self._ready_future

    def submit(
        self,
        block_bundle: Tuple[List[ray.ObjectRef], List[BlockMetadata]],
        order_index: int
        fn_args,
        fn_kwargs,
    ) -> TaskResult:
        blocks, metas = block_bundle
        block, meta = self._worker.map_block_nosplit(
            [f for meta in metas for f in meta.input_files],
            len(blocks),
            *(blocks + fn_args),
            **fn_kwargs,
        )
        return TaskResult(ref=block, metadata=meta, order_index=order_index)

    def kill(self):
        ray.kill(self._worker)


class ActorPool:
    def __init__(self, config: ActorPoolConfig):
        self._config = config
        self._pending_workers: Dict[ray.ObjectRef, Worker] = {}
        self._tasks_to_workers: Dict[ray.ObjectRef, Worker] = {}
        self._tasks_to_results: Dict[ray.ObjectRef, TaskResult] = {}
        self._workers_to_tasks: Dict[
            Worker, Set[ray.ObjectRef]]
        ] = collections.defaultdict(set)

    def create_worker(self, *worker_constructor_args):
        worker = Worker(*worker_constructor_args)
        self._pending_workers[worker.ready_future] = worker

    def wait_for_result_or_worker(
        self, timeout: float
    ) -> Tuple[Optional[TaskResult], Optional[Worker]]:
        ready, _ = ray.wait(
            list(self._tasks_to_workers.keys()) + list(self._pending_workers.keys()),
            timeout=timeout,
            num_returns=1,
            fetch_local=False,
        )
        if not ready:
            return None, None
        task = ready[0]
        if task in self._pending_workers:
            worker = self._pending_workers.pop(task)
            task_result = None
        else:
            worker = self._tasks_to_workers.pop(task)
            task_result = self._tasks_to_results.pop(task)
            self._workers_to_tasks[worker].discard(task)
        return task_result, worker

    def submit_tasks_up_to_capacity(self, block_bundles, worker, *task_args):
        while (
            block_bundles
            and (
                len(self._workers_to_tasks[worker])
                < self.config.max_tasks_in_flight_per_worker
            )
        ):
            block_bundle = block_bundles.pop()
            task_result = worker.submit_task(
                block_bundle, len(block_bundles), *task_args,
            )
            future = task_result.result_future
            self._tasks_to_workers[future] = worker
            self._tasks_to_results[future] = task_result
            self._workers_to_tasks[worker].add(future)

    def kill_pending_workers(self):
        for worker in self._pending_workers.values():
            worker.kill()

    def kill_running_worker(self, worker):
        for task in self._workers_to_tasks.pop(worker):
            del self._tasks_to_workers
            del self._tasks_to_results
        worker.kill()

    def kill_all_workers(self):
        self._kill_pending_workers()
        for worker in self._workers_to_tasks.keys():
            self.kill_running_worker(worker)

    @property
    def num_pending_workers(self):
        return len(self._pending_workers)

    @property
    def num_running_workers(self):
        return len(self._workers_to_tasks)


class ActorPoolExecutor:
    def __init__(self, name: str, actor_pool: Optional[ActorPool] = None):
        if actor_pool is None:
            actor_pool = ActorPool()
        else:
            # TODO(Clark): Support sharing actor pools with pending workers.
            assert actor_pool.num_pending_workers == 0
        self._actor_pool: ActorPool = actor_pool
        self._tasks: Dict[ray.ObjectRef, Tuple[TaskResult, ray.actor.ActorHandle]] = {}

    def execute(blocks: BlockList) -> BlockList:
        owned_by_consumer = blocks._owned_by_consumer
        block_bundles = self._bundle_blocks(blocks)
        del blocks
        orig_num_blocks = len(block_bundles)
        name = name.title()
        map_bar = ProgressBar(name, total=orig_num_blocks)
        for _ in range(self._autoscaling_policy.num_workers_to_prestart()):
            self._actor_pool.add_worker()
        map_bar.set_description(
            "Map Progress ({} actors {} pending)".format(
                (
                    self._actor_pool.num_pending_workers
                    + self._actor_pool.num_running_workers
                ),
                self._actor_pool.num_pending_workers,
            )
        )
        results = []
        while len(results) < orig_num_blocks:
            # Wait for task to be done or a new worker to be ready: if the former, we
            # have a task result to process; in either case, we have a worker available
            # for running a new task.
            result_or_worker = self._actor_pool.wait_for_result_or_worker(
                self._autoscaling_policy.get_wait_timeout()
            )
            if result_or_worker is None:
                # If no task done or worker ready before timeout, try to scale up the
                # actor pool.
                if self._autoscaling_policy.should_scale_up(
                    self._actor_pool.num_pending_workers,
                    self._actor_pool.num_running_workers,
                    len(block_bundles),
                ):
                    self._actor_pool.add_worker()
                continue

            task_result, worker = result_or_worker
            if task_result is not None:
                # Task is done, add to results and update progress bar.
                results.append(task_result)
                map_bar.update(1)
            else:
                # New worker is ready, update the progress bar.
                map_bar.set_description(
                    "Map Progress ({} actors {} pending)".format(
                        (
                            self._actor_pool.num_pending_workers
                            + self._actor_pool.num_running_workers
                        ),
                        self._actor_pool.num_pending_workers,
                    )
                )

            # Submit as many tasks as possible to the worker, up to the worker's
            # capacity.
            block_bundles = self._actor_pool.submit_tasks_up_to_capacity(
                block_bundles, worker,
            )
            # Try to scale down current worker.
            if self._autoscaling_policy.should_scale_down_worker(
                worker, len(block_bundles)
            ):
                self._actor_pool.kill_running_worker(worker)
            # Try to scale down pending workers.
            if self._autoscaling_policy.should_scale_down_pending_workers(
                len(block_bundles)
            ):
                self._actor_pool.kill_pending_workers()
        # Package results into block list.
        return self._get_results(results, owned_by_consumer)

    def _submit_task(
        self,
        block_bundle: Tuple[List[ray.ObjectRef], List[BlockMetadata]],
        worker: ray.actor.ActorHandle
    ):
        blocks, metas = block_bundles.pop()
        # TODO(swang): Support block splitting for compute="actors".
        ref, meta_ref = worker.map_block_nosplit.remote(
            [f for meta in metas for f in meta.input_files],
            len(blocks),
            *(blocks + self._fn_args),
            **self._fn_kwargs,
        )
        self._tasks[ref] = (TaskResult(ref, meta_ref, len(block_bundles)), worker)
        self._tasks_in_flight[worker] += 1

    def _wait_for_result_or_worker(self):
        ready, _ = ray.wait(
            list(self._tasks.keys()) + self._actor_pool.pending_worker_futures,
            timeout=0.01,
            num_returns=1,
            fetch_local=False,
        )
        return ready

    def _prestart_workers(self):
        for _ in range(self._autoscaling_policy.num_workers_to_prestart()):
            self._actor_pool.create_worker()

    def _scale_up_if_needed(self):
        if self._autoscaling_policy.should_scale_up():
            self._actor_pool.create_worker()

    def _scale_down_if_needed(self, worker):
        if self._autoscaling_policy.should_scale_down():
            self._actor_pool.kill_pending_workers()

    def _bundle_blocks(
        self, blocks: BlockList, target_block_size: Optional[int], name: str,
    ) -> List[Tuple[List[ray.ObjectRef], List[BlockMetadata]]]:
        blocks = blocks.get_blocks_with_metadata()
        # Bin blocks by target block size.
        if target_block_size is not None:
            _check_batch_size(blocks, target_block_size, name)
            block_bundles = _bundle_blocks_up_to_size(
                blocks, target_block_size, name
            )
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks]
        return block_bundles

    def _get_results(
        self, results: List[ray.ObjectRef], owned_by_consumer: bool
    ) -> BlockList:
        new_blocks, new_metadata = [], []
        # Put blocks in input order.
        results.sort(key=self._block_indices.get)
        # TODO(swang): Support block splitting for compute="actors".
        for block in results:
            new_blocks.append(block)
            new_metadata.append(self._metadata_mapping[block])
        new_metadata = ray.get(new_metadata)
        return BlockList(
            new_blocks, new_metadata, owned_by_consumer=owned_by_consumer
        )


class ActorPool:
    def __init__(self):
        self._pending_workers: Dict[ray.ObjectRef, ray.actor.ActorHandle] = {}
        self._idle_workers: Set[ray.actor.ActorHandle]
        self._leased_workers: Set[ray.actor.ActorHandle]

    def create_pending_worker(self):
        worker = self._worker_provider()
        ready_future = worker.ready.remote()
        self._pending_workers[ready_future] = ready_future

    def lease_pending_worker(self, future: ray.ObjectRef) -> ray.actor.ActorHandle:
        assert worker in self._pending_workers
        worker = self._pending_workers.pop(future)
        self.lease_worker(worker)
        return worker

    def return_worker_to_pool(self, worker: ray.actor.ActorHandle):
        assert worker in self._leased_workers
        assert worker not in self._pending_workers
        assert worker not in self._idle_workers
        self._leased_workers.discard(worker)
        self._idle_workers.add(worker)

    def lease_worker(self, worker: ray.actor.ActorHandle):
        assert worker not in self._lease_worker
        assert worker in self._idle_workers
        self._idle_workers.discard(worker)
        self._leased_workers.add(worker)

    def is_future_for_pending_worker(self, future: ray.ObjectRef) -> bool:
        return future in self._pending_workers

    @property
    def pending_worker_futures(self) -> List[ray.ObjectRef]:
        return list(self._pending_workers.values())

    @property
    def num_idle_workers(self):
        return len(self._idle_workers)

    @property
    def num_leased_workers(self):
        return len(self._leased_workers)

    @property
    def num_pending_workers(self):
        return len(self._pending_workers)

    @property
    def num_workers(self):
        return (
            self.num_pending_workers + self.num_idle_workers + self.num_leased_workers
        )

    def kill_worker(self, worker: ray.actor.ActorHandle):
        self._pending_workers.discard(worker)
        self._idle_workers.discard(worker)
        self._leased_workers.discard(worker)
        ray.kill(worker)

    def kill_pending_workers(self):
        for worker in self._pending_workers:
            self.kill_worker(worker)

    def shutdown(self):
        for worker_set in [
            self._pending_workers, self._idle_workers, self._leased_workers
        ]:
            for worker in worker_set:
                self.kill_worker(worker)


class ActorPoolExecutor:
    def __init__(self, name: str, actor_pool: Optional[ActorPool] = None):
        if actor_pool is None:
            actor_pool = ActorPool()
        else:
            # TODO(Clark): Support sharing actor pools with pending workers.
            assert actor_pool.num_pending_workers == 0
        self._actor_pool: ActorPool = actor_pool
        self._tasks: Dict[ray.ObjectRef, Tuple[TaskResult, ray.actor.ActorHandle]] = {}

    def execute(blocks: BlockList) -> BlockList:
        owned_by_consumer = blocks._owned_by_consumer
        block_bundles = self._bundle_blocks(blocks)
        del blocks
        orig_num_blocks = len(block_bundles)
        name = name.title()
        map_bar = ProgressBar(name, total=orig_num_blocks)
        self._prestart_workers()
        results = []
        while len(results) < orig_num_blocks:
            # Wait for task to be done or a new worker to be ready: if the former, we
            # have a task result to process; in either case, we have a worker available
            # for running a new task.
            future = self._wait_for_result_or_worker()
            if future is None:
                # If no task done or worker ready before timeout, try to scale up the
                # actor pool.
                self._scale_up_if_needed()
                continue
            if self._actor_pool.is_future_for_pending_worker(future):
                worker = self._actor_pool.lease_pending_worker(future)
                map_bar.set_description(
                    "Map Progress ({} actors {} pending)".format(
                        len(ready_workers), len(workers) - len(ready_workers)
                    )
                )
            else:
                task_result, worker = self._tasks.pop(future)
                results.append(task_result)
                self._tasks_in_flight[worker] -= 1
                self._actor_pool.lease_worker(worker)
                map_bar.update(1)
            self._submit_tasks(block_bundles, worker)
            # Schedule new tasks.
            while (
                block_bundles
                and self._tasks_in_flight[worker] < self.max_tasks_in_flight_per_actor
            ):
                self._submit_task(block_bundles.pop(), worker)
            self._actor_pool.return_worker_to_pool(worker)
            if not block_bundles:
                self._scale_down_if_needed(worker)
        return self._get_results(results, owned_by_consumer)

    def _submit_task(
        self,
        block_bundle: Tuple[List[ray.ObjectRef], List[BlockMetadata]],
        worker: ray.actor.ActorHandle
    ):
        blocks, metas = block_bundles.pop()
        # TODO(swang): Support block splitting for compute="actors".
        ref, meta_ref = worker.map_block_nosplit.remote(
            [f for meta in metas for f in meta.input_files],
            len(blocks),
            *(blocks + self._fn_args),
            **self._fn_kwargs,
        )
        self._tasks[ref] = (TaskResult(ref, meta_ref, len(block_bundles)), worker)
        self._tasks_in_flight[worker] += 1

    def _wait_for_result_or_worker(self):
        task_futures = [tr.ref for tr in self._tasks.keys()]
        ready, _ = ray.wait(
            task_futures + self._actor_pool.pending_worker_futures,
            timeout=0.01,
            num_returns=1,
            fetch_local=False,
        )
        return ready

    def _prestart_workers(self):
        for _ in range(self._autoscaling_policy.num_workers_to_prestart()):
            self._actor_pool.create_worker()

    def _scale_up_if_needed(self):
        if self._autoscaling_policy.should_scale_up():
            self._actor_pool.create_worker()

    def _scale_down_if_needed(self, worker):
        if self._autoscaling_policy.should_scale_down():
            self._actor_pool.kill_pending_workers()

    def _bundle_blocks(
        self, blocks: BlockList, target_block_size: Optional[int], name: str,
    ) -> List[Tuple[List[ray.ObjectRef], List[BlockMetadata]]]:
        blocks = blocks.get_blocks_with_metadata()
        # Bin blocks by target block size.
        if target_block_size is not None:
            _check_batch_size(blocks, target_block_size, name)
            block_bundles = _bundle_blocks_up_to_size(
                blocks, target_block_size, name
            )
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks]
        return block_bundles

    def _get_results(
        self, results: List[ray.ObjectRef], owned_by_consumer: bool
    ) -> BlockList:
        new_blocks, new_metadata = [], []
        # Put blocks in input order.
        results.sort(key=self._block_indices.get)
        # TODO(swang): Support block splitting for compute="actors".
        for block in results:
            new_blocks.append(block)
            new_metadata.append(self._metadata_mapping[block])
        new_metadata = ray.get(new_metadata)
        return BlockList(
            new_blocks, new_metadata, owned_by_consumer=owned_by_consumer
        )


def get_compute(compute_spec: Union[str, ComputeStrategy]) -> ComputeStrategy:
    if not compute_spec or compute_spec == "tasks":
        return TaskPoolStrategy()
    elif compute_spec == "actors":
        return ActorPoolStrategy()
    elif isinstance(compute_spec, ComputeStrategy):
        return compute_spec
    else:
        raise ValueError("compute must be one of [`tasks`, `actors`, ComputeStrategy]")


def is_task_compute(compute_spec: Union[str, ComputeStrategy]) -> bool:
    return (
        not compute_spec
        or compute_spec == "tasks"
        or isinstance(compute_spec, TaskPoolStrategy)
    )


def _map_block_split(
    block_fn: BlockTransform,
    input_files: List[str],
    fn: Optional[UDF],
    num_blocks: int,
    *blocks_and_fn_args: Union[Block, Any],
    **fn_kwargs,
) -> BlockPartition:
    stats = BlockExecStats.builder()
    blocks, fn_args = blocks_and_fn_args[:num_blocks], blocks_and_fn_args[num_blocks:]
    if fn is not None:
        fn_args = (fn,) + fn_args
    new_metas = []
    for new_block in block_fn(blocks, *fn_args, **fn_kwargs):
        accessor = BlockAccessor.for_block(new_block)
        new_meta = BlockMetadata(
            num_rows=accessor.num_rows(),
            size_bytes=accessor.size_bytes(),
            schema=accessor.schema(),
            input_files=input_files,
            exec_stats=stats.build(),
        )
        yield new_block
        new_metas.append(new_meta)
        stats = BlockExecStats.builder()
    yield new_metas


def _map_block_nosplit(
    block_fn: BlockTransform,
    input_files: List[str],
    fn: Optional[UDF],
    num_blocks: int,
    *blocks_and_fn_args: Union[Block, Any],
    **fn_kwargs,
) -> Tuple[Block, BlockMetadata]:
    stats = BlockExecStats.builder()
    builder = DelegatingBlockBuilder()
    blocks, fn_args = blocks_and_fn_args[:num_blocks], blocks_and_fn_args[num_blocks:]
    if fn is not None:
        fn_args = (fn,) + fn_args
    for new_block in block_fn(blocks, *fn_args, **fn_kwargs):
        builder.add_block(new_block)
    new_block = builder.build()
    accessor = BlockAccessor.for_block(new_block)
    return new_block, accessor.get_metadata(
        input_files=input_files, exec_stats=stats.build()
    )


def _bundle_blocks_up_to_size(
    blocks: List[Tuple[ObjectRef[Block], BlockMetadata]],
    target_size: int,
    name: str,
) -> List[Tuple[List[ObjectRef[Block]], List[BlockMetadata]]]:
    """Group blocks into bundles that are up to (but not exceeding) the provided target
    size.
    """
    block_bundles = []
    curr_bundle = []
    curr_bundle_size = 0
    for b, m in blocks:
        num_rows = m.num_rows
        if num_rows is None:
            num_rows = float("inf")
        if curr_bundle_size > 0 and curr_bundle_size + num_rows > target_size:
            block_bundles.append(curr_bundle)
            curr_bundle = []
            curr_bundle_size = 0
        curr_bundle.append((b, m))
        curr_bundle_size += num_rows
    if curr_bundle:
        block_bundles.append(curr_bundle)
    if len(blocks) / len(block_bundles) >= 10:
        logger.warning(
            f"`batch_size` is set to {target_size}, which reduces parallelism from "
            f"{len(blocks)} to {len(block_bundles)}. If the performance is worse than "
            "expected, this may indicate that the batch size is too large or the "
            "input block size is too small. To reduce batch size, consider decreasing "
            "`batch_size` or use the default in `map_batches`. To increase input "
            "block size, consider decreasing `parallelism` in read."
        )
    return [tuple(zip(*block_bundle)) for block_bundle in block_bundles]


def _check_batch_size(
    blocks_and_meta: List[Tuple[ObjectRef[Block], BlockMetadata]],
    batch_size: int,
    name: str,
):
    """Log a warning if the provided batch size exceeds the configured target max block
    size.
    """
    batch_size_bytes = None
    for _, meta in blocks_and_meta:
        if meta.num_rows and meta.size_bytes:
            batch_size_bytes = math.ceil(batch_size * (meta.size_bytes / meta.num_rows))
            break
    context = DatasetContext.get_current()
    if (
        batch_size_bytes is not None
        and batch_size_bytes > context.target_max_block_size
    ):
        logger.warning(
            f"Requested batch size {batch_size} results in batches of "
            f"{batch_size_bytes} bytes for {name} tasks, which is larger than the "
            f"configured target max block size {context.target_max_block_size}. This "
            "may result in out-of-memory errors for certain workloads, and you may "
            "want to decrease your batch size or increase the configured target max "
            "block size, e.g.: "
            "from ray.data.context import DatasetContext; "
            "DatasetContext.get_current().target_max_block_size = 4_000_000_000"
        )
