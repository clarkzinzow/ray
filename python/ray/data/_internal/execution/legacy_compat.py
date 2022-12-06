"""This file contains temporary helper functions for legacy plan/executor interaction.

It should be deleted once we fully move to the new executor backend.
"""

import ray.cloudpickle as cloudpickle
from typing import Iterator

import ray
from ray.data.block import Block, BlockMetadata
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import get_compute
from ray.data._internal.plan import ExecutionPlan, OneToOneStage, Stage
from ray.data._internal.execution.operators import MapOperator, InputDataBuffer
from ray.data._internal.execution.interfaces import (
    Executor,
    PhysicalOperator,
    RefBundle,
)


def execute_to_legacy_block_list(executor: Executor, plan: ExecutionPlan) -> BlockList:
    """Execute a plan with the new executor and translate it into a legacy block list.

    Args:
        executor: The executor to use.
        plan: The legacy plan to execute.

    Returns:
        The output as a legacy block list.
    """
    dag = _to_operator_dag(plan)
    blocks, metadata = [], []
    for ref_bundle in executor.execute(dag):
        for block, meta in ref_bundle.blocks:
            blocks.append(block)
            metadata.append(meta)
    return BlockList(blocks, metadata, owned_by_consumer=True)


def _to_operator_dag(plan: ExecutionPlan) -> PhysicalOperator:
    """Translate a plan into an operator DAG for the new execution backend."""

    blocks, _, stages = plan._optimize()
    operator = _blocks_to_input_buffer(blocks)
    for stage in stages:
        operator = _stage_to_operator(stage, operator)
    return operator


def _blocks_to_input_buffer(blocks: BlockList) -> PhysicalOperator:
    """Translate a block list into an InputBuffer operator.

    Args:
        blocks: The block list to translate

    Returns:
        The physical operator representing the input block list.
    """

    if hasattr(blocks, "_tasks"):
        read_tasks = blocks._tasks
        inputs = InputDataBuffer(
            [
                RefBundle(
                    [
                        (
                            ray.put(read_task),
                            # TODO(ekl) Use BlockAccessor.get_metadata in the future
                            # once we get rid of the read task as block legacy hack.
                            BlockMetadata(
                                num_rows=1,
                                size_bytes=len(cloudpickle.dumps(read_task)),
                                schema=None,
                                input_files=[],
                                exec_stats=None,
                            ),
                        )
                    ],
                    owns_blocks=True,
                )
                for read_task in read_tasks
            ]
        )

        def do_read(blocks: Iterator[Block], _) -> Iterator[Block]:
            for read_task in blocks:
                for output_block in read_task():
                    yield output_block

        return MapOperator(do_read, inputs, name="DoRead")
    else:
        output = []
        for block, meta in blocks.iter_blocks_with_metadata():
            output.append(
                RefBundle(
                    [
                        (
                            block,
                            meta,
                        )
                    ],
                    owns_blocks=False,  # TODO
                )
            )
        return InputDataBuffer(output)


def _stage_to_operator(stage: Stage, input_op: PhysicalOperator) -> PhysicalOperator:
    """Translate a stage into a PhysicalOperator.

    Args:
        stage: The stage to translate.
        input_op: The upstream operator (already translated).

    Returns:
        The translated operator that depends on the input data.
    """

    if isinstance(stage, OneToOneStage):
        if stage.fn_constructor_args or stage.fn_constructor_kwargs:
            raise NotImplementedError
        if stage.compute != "tasks":
            raise NotImplementedError

        block_fn = stage.block_fn
        fn = stage.fn
        fn_args = stage.fn_args
        fn_kwargs = stage.fn_kwargs

        def do_map(blocks: Iterator[Block], _) -> Iterator[Block]:
            for output_block in block_fn(blocks, fn, *fn_args, **fn_kwargs):
                yield output_block

        return MapOperator(
            do_map,
            input_op,
            name=stage.name,
            compute_strategy=get_compute(stage.compute),
            ray_remote_args=stage.ray_remote_args,
        )
    else:
        raise NotImplementedError
