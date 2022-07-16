import collections
import enum
import logging
from typing import Iterable, Tuple, List, Optional

import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockMetadata,
)
from ray.types import ObjectRef

logger = logging.getLogger(__name__)


def _calculate_blocks_size(
    blocks_with_metadata: Iterable[Tuple[ObjectRef[Block], BlockMetadata]],
) -> List[int]:
    get_num_rows = cached_remote_fn(_get_num_rows)
    block_sizes = []
    for block, metadata in blocks_with_metadata:
        if metadata.num_rows is None:
            # Need to fetch number of rows.
            num_rows = ray.get(get_num_rows.remote(block))
        else:
            num_rows = metadata.num_rows
        block_sizes.append(num_rows)
    return block_sizes


def _generate_split_plan(
    num_rows_per_block: List[int],
    split_indices: List[int],
) -> List[List[int]]:
    split_plan = []
    current_input_block_id = 0
    current_block_split_index = []
    offset = 0
    current_index_id = 0

    while current_index_id < len(split_indices):
        split_index = split_indices[current_index_id]
        current_block_size = num_rows_per_block[current_input_block_id]
        if split_index - offset <= current_block_size:
            current_block_split_index.append(split_index - offset)
            current_index_id += 1
            continue
        split_plan.append(current_block_split_index)
        current_block_split_index = []
        offset += num_rows_per_block[current_input_block_id]
        current_block_id += 1

    while len(split_plan) < len(num_rows_per_block):
        split_plan.append(current_block_split_index)
        current_block_split_index = []
    return split_plan


def _split_single_block(
    block: ObjectRef[Block],
    meta: BlockMetadata,
    block_size: int,
    split_indices: List[int],
) -> List[Tuple[ObjectRef[Block], BlockMetadata]]:
    """Split the provided block at the given row index."""
    if len(split_indices) == 0:
        return [(block, meta)]
    split_indices.append(block_size)
    split_result = []
    stats = BlockExecStats.builder()
    block = BlockAccessor.for_block(block)
    prev_index = 0
    for index in split_indices:
        logger.debug(f"slicing block {prev_index}:{index}")
        split_block = block.slice(prev_index, index, copy=True)
        prev_index = index
        accessor = BlockAccessor.for_block(split_block)
        split_meta = BlockMetadata(
            num_rows=accessor.num_rows(),
            size_bytes=accessor.size_bytes(),
            schema=meta.schema,
            input_files=meta.input_files,
            exec_stats=stats.build(),
        )
        split_result.append((ray.put(split_block), split_meta))
    return split_result


def _split_at_indices(
    blocks_with_metadata: Iterable[Tuple[ObjectRef[Block], BlockMetadata]],
    indices: List[int],
) -> Tuple[List[List[ObjectRef[Block]]], List[List[BlockMetadata]]]:
    """Split blocks at the provided indices.

    Args:
        blocks_with_metadata: Block futures to split, including the associated metadata.
        indices: The (global) indices at which to split the blocks.
    Returns:
        The block split futures and their metadata. If an index split is empty, the
        corresponding block split future will be None.
    """
    split_single_block = cached_remote_fn(_split_single_block)

    block_sizes = _calculate_blocks_size(blocks_with_metadata)
    split_plan = _generate_split_plan(block_sizes, indices)
    split_results = ray.get(
        [
            split_single_block.remote(
                block_with_metadata[0],
                block_with_metadata[1],
                block_sizes[i],
                split_plan[i],
            )
            for i, block_with_metadata in enumerate(blocks_with_metadata)
        ]
    )


def _get_num_rows(block: Block) -> int:
    """Get the number of rows contained in the provided block."""
    return BlockAccessor.for_block(block).num_rows()
