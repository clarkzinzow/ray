import collections
import itertools
import math
from typing import List, Any, Optional, Dict, Union

import ray
from ray.types import ObjectRef
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.impl.block_list import BlockList
from ray.data.impl.remote_fn import cached_remote_fn
from ray.data.impl.arrow_block import DelegatingArrowBlockBuilder


DEFAULT_ROWS_PER_BLOCK = 1000000


_SPLIT_BLOCK_AT_INDICES = None


def _cached_split_block_at_indices():
    global _SPLIT_BLOCK_AT_INDICES

    if _SPLIT_BLOCK_AT_INDICES:
        _SPLIT_BLOCK_AT_INDICES = cached_remote_fn(_split_block_at_indices)
    return _SPLIT_BLOCK_AT_INDICES


_MERGE_BLOCKS = None


def _cached_merge_blocks():
    global _MERGE_BLOCKS

    if _MERGE_BLOCKS:
        _MERGE_BLOCKS = cached_remote_fn(_merge_blocks)
    return _MERGE_BLOCKS


# TODO(Clark): Expose rows_per_block instead of blocks_per_split? The
# former is probably more intuitive.
def split(
        blocks: BlockList,
        n: int,
        *,
        equal: bool,
        blocks_per_split: Optional[int],
        locality_hints: Optional[List[Any]]) -> List[BlockList]:
    """Split the dataset into ``n`` disjoint pieces.

    This returns a list of sub-datasets that can be passed to Ray tasks
    and actors and used to read the dataset records in parallel.

    Examples:
        >>> # Split up a dataset to process over `n` worker actors.
        >>> shards = ds.split(len(workers), locality_hints=workers)
        >>> for shard, worker in zip(shards, workers):
        ...     worker.consume.remote(shard)

    Time complexity: O(1)

    Args:
        n: Number of child datasets to return.
        equal: Whether to guarantee each split has an equal
            number of records. This will drop at most ds.count() % n rows.
        blocks_per_split: The desired number of blocks per split.
            By default, this will be one block per 1,000,000 rows.
        locality_hints: A list of Ray actor handles of size ``n``. The
            system will try to co-locate the blocks of the ith dataset
            with the ith actor to maximize data locality.

    Returns:
        A list of ``n`` disjoint dataset splits.
    """
    if n <= 0:
        raise ValueError(f"The number of splits {n} is not positive.")

    if locality_hints is None:
        return _split_simple(n, blocks_per_split)

    if locality_hints and len(locality_hints) != n:
        raise ValueError(
            f"The length of locality_hints {len(locality_hints)} "
            "doesn't equal the number of splits {n}.")

    splitter = LocalityAwareFairSplitter.build(
        blocks, equal, locality_hints, blocks_per_split)
    return splitter.split()


class LocalityAwareFairSplitter:
    # If locality hints are given, we use a five-round greedy algorithm
    # for creating a fair redistribution of blocks across the provided set
    # of actors such that:
    #
    #  1. All actors are allocated the same number of blocks each with the
    #     same number of rows (guarantees no dataset or block skew).
    #  2. Block locality is maximized (minimizes network transfer).
    #  3. Block splitting is minimized (minimizes in-memory copying).
    #
    # This algorithm has the following rounds:
    #
    # Round 1 - primary location allocation:
    #  - Allocate each block to the actors on the block's primary
    #    location, up to each actor's row allocation limit.
    #
    # Round 2 - secondary location allocation:
    #  - Allocate each block to the actors on the block's secondary
    #    locations, up to each actor's row allocation limit.
    #
    # Round 3 - locality-aware splitting:
    #  - Split the remaining blocks, attempting to allocate the splits to
    #    actors that are located on nodes that already have the block.
    #
    # Round 4 - locality-agnostic residual splitting:
    #  - Split the remaining blocks, allocating the splits to arbitrary
    #    nodes.
    #
    # Round 5 - rebalance blocks:
    #  - Split blocks larger than the target block size into target-sized
    #    blocks; merge blocks smaller than the target block size into
    #    target-sized blocks.

    DEFAULT_ROWS_PER_BLOCK = 1000000

    def __init__(
            self,
            locality_hints: List[Any],
            equal: bool,
            metadata: Dict[ObjectRef[Block], BlockMetadata],
            node_ids_by_block: Dict[ObjectRef[Block], str],
            actors_by_node_id: Dict[str, Any],
            addresses_by_node_id: Dict[str, str],
            blocks_per_split: Optional[int] = None):
        self.locality_hints = locality_hints
        self.equal = equal
        self.blocks_per_split = blocks_per_split

        # Useful tables.
        self.node_ids_by_block = node_ids_by_block
        self.actors_by_node_id = actors_by_node_id
        self.addresses_by_node_id = addresses_by_node_id

        # Block metadata.
        self.metadata_mapping = metadata
        self.num_rows_by_block = {b: m.num_rows for b, m in metadata.items()}
        assert all(
            num_rows is not None
            for num_rows in self.num_rows_by_block.values())

        # Expected number of rows.
        self.total_rows = sum(self.num_rows_by_block.values())
        self.target_num_rows_per_actor = (
            self.total_rows // len(locality_hints))

        # Block allocation maps.
        self.block_allocation_per_actor = collections.defaultdict(list)
        self.num_rows_allocated_per_actor = collections.Counter(
            {actor: 0 for actor in locality_hints})

    @classmethod
    def build_splitter(
            cls, blocks: BlockList, equal: bool, locality_hints: List[Any],
            blocks_per_split: Optional[int] = None):
        # Block metadata.
        metadata = _get_metadata_for_blocks(blocks)

        # Wait until all blocks have been computed.
        block_refs = list(blocks)
        ray.wait(block_refs, num_returns=len(block_refs), fetch_local=False)

        # Useful tables.
        node_ids_by_block = _get_node_ids_by_block(list(blocks))
        actors_by_node_id = _get_actors_by_node_id(locality_hints)
        addresses_by_node_id = _get_addresses_by_node_id()
        return cls(
            locality_hints, equal, metadata, node_ids_by_block,
            actors_by_node_id, addresses_by_node_id, blocks_per_split)

    def split(self) -> List[BlockList]:
        blocks = list(self.num_rows_by_block.keys())
        # Round 1: Allocate blocks to the actor with the most allocation
        # availability that's at the block's primary location.
        unallocated = self.allocate_to_primary_actors(
            blocks, overallocate=not self.equal)

        # Round 2: Allocate blocks to the actor with the most allocation
        # capacity that's at one of the block's secondary locations.
        unallocated = self.allocate_to_secondary_actors(
            unallocated, overallocate=not self.equal)

        if self.equal:
            # Round 3: Split the blocks, attempting to allocate the splits to
            # actors that are located on nodes that already have the block.
            unallocated, metadata_futures = self.split_and_allocate_to_actors(
                unallocated)

            # Round 4: Split the blocks, allocating the splits to arbitrary
            # nodes.
            new_metadata_futures = self.split_and_allocate_to_arbitrary_actors(
                unallocated)

            metadata_futures.extend(new_metadata_futures)

            # All blocks should be allocated at this point.
            assert all(
                allocated == self.target_num_rows_per_actor
                for allocated in self.num_rows_allocated_per_actor.values())

            self._update_metadata_from_futures(metadata_futures)

            # Round 5: Split blocks larger than the target block size into
            # target-sized blocks; merge blocks smaller than the target block
            # size into target-sized blocks.
            metadata_futures = self.rebalance_blocks()

            self._update_metadata_from_futures(metadata_futures)
        else:
            self.allocate_to_arbitrary_actors(unallocated)

        return [
                BlockList(
                    self.block_allocation_per_actor[actor],
                    [self.metadata_mapping[b]
                        for b in self.block_allocation_per_actor[actor]])
                for actor in self.locality_hints]

    def allocate_to_primary_actors(self, blocks, overallocate=False):
        primary_actors_by_block = {
            block: self.actors_by_node_id[self.node_ids_by_block[block][0]]
            for block in blocks}
        return self._allocate_to_actors(
            blocks, primary_actors_by_block, overallocate=overallocate)

    def allocate_to_secondary_actors(self, blocks, overallocate=False):
        secondary_actors_by_block = {
            block: itertools.chain.from_iterable(
                self.actors_by_node_id[node_id]
                for node_id in self.node_ids_by_block[block])
            for block in blocks}
        return self._allocate_to_actors(
            blocks, secondary_actors_by_block, overallocate=overallocate)

    def _allocate_to_actors(
            self, blocks, actors_by_block, overallocate=False):
        return _allocate_to_actors(
            blocks, self.num_rows_by_block, actors_by_block,
            self.target_num_rows_per_actor, self.num_rows_allocated_per_actor,
            self.block_allocation_per_actor, overallocate=overallocate)

    def split_and_allocate_to_actors(self, blocks):
        metadata_futures = []
        unallocated = []
        # Assumed invariant: No full block will fit in a local bin.
        # Split the blocks, attempting to allocate the splits to
        # actors that are located on nodes that already have the block.
        for block in blocks:
            size = self.num_rows_by_block[block]
            block_meta = self.metadata_mapping[block]
            block_locations = self.node_ids_by_block[block]
            # The offset into the block, caused by splits on previous
            # locations.
            offset = 0
            # Whether a split ever happens, for any of the locations. A split
            # won't happen if all actors on nodes that have the object are at
            # their allocation limit.
            split_made = False
            for node_id in block_locations:
                if offset >= size:
                    break
                # TODO(Clark): Optimize by keeping a node id --> actor dict
                # of priority queues instead of resorting.
                actors = self.actors_by_node_id[node_id]
                # TODO(Clark): Move _get_actor_slack() and
                # _order_actors_by_availability() calls into
                # _get_split_indices()?
                actor_slack = _get_actor_slack(
                    actors, self.target_num_rows_per_actor,
                    self.num_rows_allocated_per_actor)
                actors = _order_actors_by_availability(actors, actor_slack)

                split_indices, actors, offset = _get_split_indices(
                    actors, actor_slack, size, offset)
                if split_indices is None:
                    continue
                else:
                    split_made = True
                split_sizes = _get_split_sizes(split_indices)
                # Submit splitting task. We give a custom node resource in
                # order to ensure that this splitting task is scheduled on
                # the same node as the block.
                # TODO(Clark): Some Ray clusters won't have access to node
                # IDs (e.g. if their "node" primitive is a container); should
                # we expose a custom resource hook here? How would that work?
                # We could also fall back to relying on locality-aware
                # scheduling in that case, but that could result in a lot of
                # unnecessary data transfer if the cluster is busy.
                block_splits = _create_splits(
                    block, block_meta, split_indices, split_resources={
                        f"node:{self.addresses_by_node_id[node_id]}": 0.001})
                meta_futures = self._allocate_splits(
                    block_splits, actors, split_sizes)
                metadata_futures.extend(meta_futures)
            # TODO(Clark): Find a way to eagerly discard any references to the
            # old block in order to aggressively free up memory.
            if not split_made:
                # No splits were made, so we need to try to allocate the block
                # in the next round.
                unallocated.append(block)
            elif offset < size:
                # There's a leftover slice of the block that wasn't allocated
                # in this round, save it for allocation in the next round.
                (
                    leftover_block,
                    leftover_metadata) = _cached_split_block_at_indices(
                        ).options(
                            num_returns=2,
                            resources={
                                (
                                    "node:"
                                    f"{self.addresses_by_node_id[node_id]}"
                                ): 0.001}
                        ).remote(
                            block, block_meta, [offset, size])
                leftover_size = size - offset
                self.num_rows_by_block[leftover_block] = leftover_size
                unallocated.append(leftover_block)
                metadata_futures.append((leftover_block, leftover_metadata))
                del self.num_rows_by_block[block]
                del self.metadata_mapping[block]
            else:
                # Splits were made and the entire block was allocated.
                del self.num_rows_by_block[block]
                del self.metadata_mapping[block]
        return unallocated, metadata_futures

    def split_and_allocate_to_arbitrary_actors(self, blocks):
        blocks = sorted(
            blocks, key=lambda b: self.num_rows_by_block[b], reverse=True)
        metadata_futures = []
        # Split the blocks, allocating the splits to arbitrary nodes.
        for block in blocks:
            size = self.num_rows_by_block[block]
            block_meta = self.metadata_mapping[block]
            # TODO(Clark): Optimize by using a priority queue/balanced binary
            # tree instead of resorting on every block.
            actors = self.locality_hints
            actor_slack = _get_actor_slack(
                actors, self.target_num_rows_per_actor,
                self.num_rows_allocated_per_actor)
            actors = _order_actors_by_availability(
                actors, actor_slack)

            split_indices, actors, _ = _get_split_indices(
                actors, actor_slack, size, 0)
            assert split_indices is not None
            split_sizes = _get_split_sizes(split_indices)
            # Submit splitting task and allocate splits.
            block_splits = _create_splits(block, block_meta, split_indices)
            meta_futures = self._allocate_splits(
                block_splits, actors, split_sizes)
            del self.num_rows_by_block[block]
            del self.metadata_mapping[block]
            metadata_futures.extend(meta_futures)
        return metadata_futures

    def rebalance_blocks(self):
        metadata_futures = []

        if self.blocks_per_split is None:
            self.blocks_per_split = math.ceil(
                self.target_num_rows_per_actor / self.DEFAULT_ROWS_PER_BLOCK)
        rows_per_block = (
            self.target_num_rows_per_actor // self.blocks_per_split)

        for actor in self.locality_hints:
            blocks = self.block_allocation_per_actor[actor]
            blocks = sorted(
                blocks, key=lambda b: self.num_rows_by_block[b], reverse=True)
            new_allocation = []
            leftovers = []
            # First process blocks larger than or equal to target.
            for idx, block in enumerate(blocks):
                size = self.num_rows_by_block[block]
                if size < rows_per_block:
                    break
                if size == rows_per_block:
                    new_allocation.append(block)
                    continue
                # Case: size > rows_per_block
                split_indices = list(range(0, size, rows_per_block))
                if split_indices[-1] < size:
                    split_indices.append(size)
                split_sizes = _get_split_sizes(split_indices)
                block_splits = _create_splits(
                    block, self.metadata_mapping[block], split_indices)
                self.num_rows_by_block.update(
                    zip(zip(*block_splits)[0], split_sizes))
                metadata_futures.extend(block_splits)
                last_split_size = split_sizes[-1]
                if last_split_size < rows_per_block:
                    # Handle leftover block.
                    leftover_block, _ = block_splits.pop()
                    split_sizes.pop()
                    self.num_rows_by_block[leftover_block] = last_split_size
                    leftovers.append(leftover_block)
                new_allocation.extend(block_splits)
                del self.num_rows_by_block[block]
                del self.metadata_mapping[block]
            # Move on to blocks smaller than target.
            blocks = blocks[idx:] + leftovers
            # Order blocks largest to smallest for first-fit-decreasing
            # packing.
            blocks = sorted(
                blocks, lambda b: self.num_rows_by_block[b], reverse=True)
            # This merge buffer is added to until we've reached the
            # rows_per_block target, at which point we merge all of the
            # blocks and clear the buffer.
            # TODO(Clark): Add an abstraction (class) for the merge buffer?
            merge_buffer = []
            merge_buffer_size = 0
            # We use two-pointer loop to allocate blocks to the buffer, where
            # we first try to allocate the larger (left) block, then try to
            # allocate the smaller (right) block, and if neither fit, then we
            # split the smaller (right) block to add a sub-block that does fit.
            # This naive method approximates an optimal packing of blocks into
            # rows_per_block bins that minimizes splitting.
            left = 0
            right = len(blocks) - 1
            while left < right:
                left_block = blocks[left]
                left_size = self.num_rows_by_block[left_block]
                right_block = blocks[right]
                right_size = self.num_rows_by_block[right_block]
                if merge_buffer_size + left_size <= rows_per_block:
                    # The left (larger) block can fit into the merge buffer,
                    # so add it to the buffer and increment the left pointer.
                    merge_buffer.append(left_block)
                    merge_buffer_size += left_size
                    left += 1
                elif merge_buffer_size + right_size <= rows_per_block:
                    # The right (smaller) block can fit into the merge buffer,
                    # so add it to the buffer and increment the right pointer.
                    merge_buffer.append(right_block)
                    merge_buffer_size += right_size
                    right -= 1
                else:
                    # Neither the left nor the right block fit in the merge
                    # buffer, so we split right (smaller) block into a block
                    # that will fit into the buffer and a leftover block that
                    # we add back into the candidate block list.

                    # Submit splitting task.
                    diff = rows_per_block - merge_buffer_size
                    (
                        block, meta, leftover_block,
                        leftover_meta) = _cached_split_block_at_indices(
                            ).options(num_return=4).remote(
                                right_block,
                                self.metadata_mapping[right_block],
                                [0, diff, right_size])
                    del self.num_rows_by_block[right_block]
                    del self.metadata_mapping[right_block]
                    self.num_rows_by_block[block] = diff
                    self.num_rows_by_block[leftover_block] = right_size - diff
                    metadata_futures.append((block, meta))
                    merge_buffer.append(block)
                    merge_buffer_size += diff
                    assert merge_buffer_size == rows_per_block
                    # Keep leftover for later processing.
                    metadata_futures.append((leftover_block, leftover_meta))
                    # We overwrite the old rightmost block and don't advance
                    # the right pointer since (1) the old block can be
                    # discarded, (2) the leftover block is guaranteed to be
                    # smaller than the old block, and (3) the rightmost block
                    # should be the smallest block in the to-be-allocated set.
                    blocks[right] = leftover_block
                if merge_buffer_size == rows_per_block:
                    input_files = set(
                        itertools.chain(
                            self.metadata_mapping[b].input_files
                            for b in merge_buffer))
                    merged_block, merged_metadata = _cached_merge_blocks(
                        ).remote(*merge_buffer, input_files=input_files)
                    for b in merge_buffer:
                        del self.num_rows_by_block[b]
                        del self.metadata_mapping[b]
                    self.num_rows_by_block[merged_block] = rows_per_block
                    new_allocation.append(merged_block)
                    metadata_futures.append((merged_block, merged_metadata))
                    merge_buffer = []
                    merge_buffer_size = 0
            assert len(new_allocation) == self.blocks_per_split
            assert all(
                self.num_rows_by_block[b] == rows_per_block
                for b in new_allocation)
            self.block_allocation_per_actor[actor] = new_allocation
        return metadata_futures

    def _update_metadata_from_futures(self, metadata_futures):
        # Fetch new metadata for block splits.
        blocks, metas = zip(*metadata_futures)
        if metas:
            metas = ray.get(metas)
            self.metadata_mapping.update(dict(zip(blocks, metas)))

    def _allocate_splits(self, block_splits, actors, split_sizes):
        metadata_futures = []
        # Allocate the splits to the corresponding actors.
        for actor, num_rows, (block_split, meta) in zip(
                actors, split_sizes, block_splits):
            self.block_allocation_per_actor[actor].append(block_split)
            self.num_rows_allocated_per_actor[actor] += num_rows
            self.num_rows_by_block[block_split] = num_rows
            # Save these block metadata futures; we'll fetch them in
            # bulk later.
            metadata_futures.append((block_split, meta))
        return metadata_futures


def _allocate_to_actors(
        blocks, num_rows_by_block, actors_by_block, target_num_rows_per_actor,
        num_rows_allocated_per_actor, block_allocation_per_actor,
        overallocate=False):
    # Sort blocks into decreasing order, yielding first-fit-decreasing
    # packing.
    blocks = sorted(
        blocks, key=lambda b: num_rows_by_block[b], reverse=True)
    unallocated = []
    # Allocate blocks to the actor with the most allocation
    # availability that resides within the given location slice.
    for block in blocks:
        size = num_rows_by_block[block]
        actors = actors_by_block[block]
        # Get the actor with the most allocation availability.
        actor, curr_num_rows = _get_actor_with_highest_availability(
            actors, num_rows_allocated_per_actor)
        # If this would put the actor past its allocation limit,
        # don't allocate the block.
        cap = target_num_rows_per_actor
        if not overallocate:
            cap -= size
        if curr_num_rows > cap:
            unallocated.append(block)
            continue
        # Allocate the block.
        block_allocation_per_actor[actor].append(block)
        num_rows_allocated_per_actor[actor] += size
    return unallocated


def _create_splits(block, block_metadata, split_indices, split_resources=None):
    block_splits = _cached_split_block_at_indices().options(
        num_return=2 * (len(split_indices) - 1),
        resources=split_resources
    ).remote(block, block_metadata, split_indices)
    # Pack [b0, m0, b1, m1] into [(b0, m0), (b1, m1)].
    return list(zip(block_splits[::2], block_splits[1::2]))


def _get_actor_slack(actors, cap, num_rows_allocated_by_actor):
    return {
        actor: cap - num_rows_allocated_by_actor[actor]
        for actor in actors}


def _order_actors_by_availability(
        actors, actor_slack, descending=True, drop_full=True):
    # Get actors with any remaining allocation slack.
    if drop_full:
        actors = list(filter(lambda actor: actor_slack[actor] > 0, actors))
    # Order the actors from most slack to least.
    return sorted(actors, lambda actor: actor_slack[actor], reverse=descending)


def _get_actor_with_highest_availability(actors, num_rows_allocated_by_actor):
    actor = min(
        actors,
        key=lambda actor: num_rows_allocated_by_actor[actor])
    curr_num_rows = num_rows_allocated_by_actor[actor]
    return actor, curr_num_rows


def _get_split_indices(actors, actor_slack, block_size, offset=0):
    # Get block splits that would fill some or all of the remaining
    # slack in some or all of the actors that are on the same node
    # as this block.
    assert not actors or block_size > actor_slack[actors[0]]
    splits = []
    cum_num_rows = offset
    idx = 0
    while cum_num_rows < block_size and idx < len(actors):
        actor = actors[idx]
        slack = actor_slack[actor]
        cum_num_rows += slack
        splits.append((cum_num_rows, actor))
        idx += 1
    if not splits:
        # No splits were made, so we move on to the next location.
        return None, None, None
    split_indices, out_actors = zip(*splits)
    split_indices = [offset] + list(split_indices)
    # Bound final block by block size. This will also be the returned
    # offset.
    split_indices[-1] = min(split_indices[-1], block_size)
    return split_indices, list(out_actors), split_indices[-1]


def _get_split_sizes(split_indices):
    # Get the size (number of rows) of each block split.
    return [
        high - low
        for low, high in zip(
            split_indices[:-1], split_indices[1:])]


def _get_metadata_for_blocks(blocks):
    metadata = {
        b: m
        for b, m in zip(blocks, blocks.get_metadata())
    }
    missing = [
        b for b, m in metadata.items() if m.num_rows is None]
    if missing:
        get_num_rows = cached_remote_fn(
            lambda block: BlockAccessor.for_block(block).num_rows())
        # Get missing row data.
        missing_num_rows = ray.get([
            get_num_rows.remote(b) for b in missing])
        for block, num_rows in zip(missing, missing_num_rows):
            metadata[block].num_rows = num_rows
    assert all(m.num_rows is not None for m in metadata.values())
    return metadata


def _get_node_ids_by_block(blocks: List[ObjectRef[Block]]
                           ) -> Dict[str, List[ObjectRef[Block]]]:
    """Build the reverse index from node_id to block_refs.
    """
    block_ref_locations = ray.experimental.get_object_locations(blocks)
    node_ids_by_block = collections.defaultdict(list)
    for block_ref in blocks:
        node_ids = block_ref_locations.get(block_ref, {}).get(
            "node_ids", [])
        node_ids_by_block[block_ref] = node_ids
    return node_ids_by_block


def _get_actors_by_node_id(actors: List[Any]) -> Dict[Any, str]:
    """Build a map from a actor to its node_id.
    """
    actors_state = ray.state.actors()
    actors_by_node_id = collections.defaultdict(list)
    for actor in actors:
        node_id = actors_state.get(actor._actor_id.hex(), {}).get(
            "Address", {}).get("NodeID")
        actors_by_node_id[node_id].append(actor)
    return actors_by_node_id


def _get_addresses_by_node_id():
    return {
        node["NodeID"]: node["NodeManagerAddress"]
        for node in ray.nodes()}


def _split_simple(
        blocks: BlockList,
        n: int,
        *,
        blocks_per_split: Optional[int]):
    metadata_mapping = {
        b: m
        for b, m in zip(blocks, blocks.get_metadata())
    }
    num_rows_by_block = {b: m.num_rows for b, m in metadata_mapping.items()}
    missing = [
        b for b, num_rows in num_rows_by_block.items() if num_rows is None]
    if missing:
        get_num_rows = cached_remote_fn(
            lambda block: BlockAccessor.for_block(block).num_rows())
        missing_num_rows = ray.get([get_num_rows.remote(b) for b in missing])
        num_rows_by_block.update(zip(missing, missing_num_rows))
    assert all(num_rows is not None for num_rows in num_rows_by_block.values())
    total_rows = sum(num_rows_by_block.values())

    split_block_at_indices = _cached_split_block_at_indices()
    # merge_blocks = _cached_merge_blocks()

    target_num_rows_per_slot = total_rows // n
    blocks = list(blocks)
    blocks = sorted(blocks, lambda b: num_rows_by_block[b], reverse=True)
    block_allocation_per_slot = [[] for _ in range(n)]
    num_rows_allocated_per_slot = [0] * n
    metadata_futures = []
    # Allocate blocks to dataset slots, considering largest blocks
    # first.
    for block in blocks:
        size = num_rows_by_block[block]
        # Get actors with any remaining allocation slack.
        slots_with_slack = [
            ((target_num_rows_per_slot -
                num_rows_allocated_per_slot[slot]), slot)
            for slot in range(n)]
        slots_with_slack = list(
            filter(lambda x: x[0] > 0, slots_with_slack))
        # Order the slots from most slack to least.
        slots_with_slack = sorted(slots_with_slack, reverse=True)
        # Get block splits that would fill some or all of the remaining
        # slack in some or all of the slots.
        splits = []
        cum_num_rows = 0
        idx = 0
        while cum_num_rows < size:
            assert idx < len(slots_with_slack)
            slack, slot = slots_with_slack[idx]
            cum_num_rows += slack
            splits.append((cum_num_rows, slot))
            idx += 1

        assert len(splits) > 0
        split_indices, slots = zip(*splits)
        # Bound final block by block size.
        split_indices[-1] = min(split_indices[-1], size)
        # Get the size (number of rows) of each block split.
        split_sizes = [
            high - low
            for low, high in zip(
                split_indices[:-1], split_indices[1:])]
        split_indices = [0] + split_indices
        # Submit splitting task.
        block_splits = split_block_at_indices.options(
            num_return=2 * (len(split_indices) - 1)).remote(
                block, metadata_mapping[block], split_indices)
        del num_rows_by_block[block]
        del metadata_mapping[block]
        # Pack [b0, m0, b1, m1] into [(b0, m0), (b1, m1)], and pair
        # with the corresponding block split sizes.
        block_splits = list(
            zip(split_sizes,
                zip(block_splits[::2], block_splits[1::2])))
        # Allocate the splits to the corresponding slots.
        for slot, (num_rows, (block_split, meta)) in zip(
                slots, block_splits):
            block_allocation_per_slot[slot].append(block_split)
            num_rows_allocated_per_slot[slot] += num_rows
            num_rows_by_block[block_split] = num_rows
            # Save these block metadata futures; we'll fetch them in
            # bulk later.
            metadata_futures.append((block_split, meta))
    # All blocks should be allocated at this point.
    assert all(
        allocated == target_num_rows_per_slot
        for allocated in num_rows_allocated_per_slot.values())

    # TODO(Clark): Add block balancing round, see locality-aware
    # splitting, round 5.

    # Fetch new metadata for block splits.
    blocks, metas = zip(*metadata_futures)
    if metas:
        metas = ray.get(metas)
        metadata_mapping.update(dict(zip(blocks, metas)))

    return [
        BlockList(
                block_allocation_per_slot[slot],
                [metadata_mapping[b]
                    for b in block_allocation_per_slot[slot]])
        for slot in range(n)]


def _split_block_at_indices(
        block: Block, meta: BlockMetadata, indices: List[int]):
    """Split the block at the provided indices, producing len(indices) - 1
    blocks. If given indices [a, b, c, d], this will return splits
    [a, b), [b, c), and [c, d).
    """
    block = BlockAccessor.for_block(block)
    assert len(indices) >= 2
    out = []
    for low, high in zip(indices[:-1], indices[1:]):
        b = block.slice(low, high, copy=True)
        a = BlockAccessor.for_block(b)
        m = BlockMetadata(
            num_rows=a.num_rows(),
            size_bytes=a.size_bytes(),
            schema=meta.schema,
            input_files=meta.input_files)
        out.append(b)
        out.append(m)
    return out


def _merge_blocks(*blocks: List[Block], input_files: List[str]):
    builder = DelegatingArrowBlockBuilder()
    for block in blocks:
        builder.add_block(block)
    out_block = builder.build()
    return out_block, BlockAccessor.for_block(
        out_block).get_metadata(input_files)


def locality_aware_fair_split(
        blocks, equal, locality_hints, blocks_per_split=None):
    # If locality hints are given, we use a five-round greedy algorithm
    # for creating a fair redistribution of blocks across the provided set
    # of actors such that:
    #
    #  1. All actors are allocated the same number of blocks each with the
    #     same number of rows (guarantees no dataset or block skew).
    #  2. Block locality is maximized (minimizes network transfer).
    #  3. Block splitting is minimized (minimizes in-memory copying).
    #
    # This algorithm has the following rounds:
    #
    # Round 1 - primary location allocation:
    #  - Allocate each block to the actors on the block's primary
    #    location, up to each actor's row allocation limit.
    #
    # Round 2 - secondary location allocation:
    #  - Allocate each block to the actors on the block's secondary
    #    locations, up to each actor's row allocation limit.
    #
    # Round 3 - locality-aware splitting:
    #  - Split the remaining blocks, attempting to allocate the splits to
    #    actors that are located on nodes that already have the block.
    #
    # Round 4 - locality-agnostic residual splitting:
    #  - Split the remaining blocks, allocating the splits to arbitrary
    #    nodes.
    #
    # Round 5 - rebalance blocks:
    #  - Split blocks larger than the target block size into target-sized
    #    blocks; merge blocks smaller than the target block size into
    #    target-sized blocks.

    # Wait for all dataset blocks to materialize.
    block_refs = list(blocks)
    ray.wait(block_refs, num_returns=len(block_refs), fetch_local=False)

    block_set = _build_block_set(blocks)
    total_rows = sum(
        block_datum.num_rows for block_datum in iter(block_set))
    capacity = total_rows // len(locality_hints)
    actor_bin_set = _build_actor_bin_set(
        locality_hints, capacity)
    actors_by_node_id = _get_actors_by_node_id(locality_hints)
    addresses_by_node_id = _get_addresses_by_node_id()

    # TODO(Clark): Explore BlockAllocator abstraction that wraps block_set
    # and actor_bin_set, holds actors_by_node_id and addresses_by_node_id,
    # and ensures certain invariants, such as:
    #  1. A block is removed from block_set when it's added to an actor bin's
    #     allocation.

    # Round 1: Allocate blocks to the actor with the most allocation
    # availability that's at the block's primary location.
    allocate_blocks_to_primary_actors(
        block_set, actor_bin_set, actors_by_node_id,
        overallocate=not equal)

    # Round 2: Allocate blocks to the actor with the most allocation
    # capacity that's at one of the block's secondary locations.
    allocate_blocks_to_secondary_actors(
        block_set, actor_bin_set, actors_by_node_id,
        overallocate=not equal)

    if equal:
        # Round 3: Split the blocks, attempting to allocate the splits to
        # actors that are located on nodes that already have the block.
        split_blocks_over_actors(
            block_set, actor_bin_set, actors_by_node_id, addresses_by_node_id)

        # Round 4: Split the blocks, allocating the splits to arbitrary
        # nodes.
        split_blocks_over_arbitrary_actors(
            block_set, actor_bin_set, actors_by_node_id, addresses_by_node_id)

        block_set.resolve_metadata_futures()

        if blocks_per_split is None:
            blocks_per_split = math.ceil(
                capacity / DEFAULT_ROWS_PER_BLOCK)
        block_size = capacity // blocks_per_split

        # Round 5: Split blocks larger than the target block size into
        # target-sized blocks; merge blocks smaller than the target block
        # size into target-sized blocks.
        rebalance_blocks(block_set, actor_bin_set, block_size)

        block_set.resolve_metadata_futures()
    else:
        # We don't want to split blocks, so we allocate the remaining blocks
        # to arbitrary actors.
        allocate_blocks_to_arbitrary_actors(block_set, actor_bin_set)

    block_lists = []
    for actor in locality_hints:
        allocated_blocks = actor_bin_set.get_bin(actor).allocated_blocks
        block_lists.append(
            BlockList(
                [block.block for block in allocated_blocks],
                [block.meta for block in allocated_blocks]))
    return block_lists


def _build_block_set(blocks, locality_hints):
    metadata = _get_metadata_for_blocks(blocks)
    node_ids_by_block = _get_node_ids_by_block(blocks)
    return BlockSet({
        b: BlockData(
            b, metadata[b], metadata[b].num_rows, node_ids_by_block[b])
        for b in blocks})


def _build_actor_bin_set(actors, capacity):
    return BinSet({
        actor: Bin(actor, capacity)
        for actor in actors})


class BlockData:
    block: ObjectRef[Block]
    meta: Union[BlockMetadata, ObjectRef[BlockMetadata]]
    num_rows: int
    preferred_nodes: Optional[List[str]]
    allocated: bool

    def __init__(
            self, block: ObjectRef[Block],
            meta: Union[BlockMetadata, ObjectRef[BlockMetadata]],
            num_rows: int, preferred_nodes: Optional[List[str]] = None):
        self.block = block
        self.meta = meta
        self.num_rows = num_rows
        self.preferred_nodes = preferred_nodes
        self.allocated = False

    def set_allocated(self):
        self.allocated = True

    def __str__(self):
        return (
            f"{type(self).__name__}: block={self.block}, "
            f"num_rows={self.num_rows}")

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.block)


class BlockSet:
    blocks: Dict[ObjectRef[Block], BlockData]
    blocks_missing_meta: List[ObjectRef[Block]]

    def __init__(self, blocks: List[BlockData]):
        self.blocks = {b.block: b for b in blocks}
        self.blocks_missing_meta = []

    def get_block_data(self, block: ObjectRef[Block]):
        return self.blocks[block]

    def get_unallocated_blocks(self):
        return [block for block in self.blocks.values() if not block.allocated]

    def get_unallocated_blocks_ordered_by_size(self, descending=True):
        return sorted(
            self.get_unallocated_blocks(), key=lambda block: block.num_rows,
            reverse=descending)

    def add_block_data(self, block_data: BlockData):
        self.blocks[block_data.block] = block_data
        if isinstance(block_data.meta, ray.ObjectRef):
            self.blocks_missing_meta.append(block_data.block)

    def resolve_metadata_futures(self):
        if self.blocks_missing_meta:
            meta_futures = [
                self.get_block_data(block).meta
                for block in self.blocks_missing_meta]
            for block, meta in zip(
                    self.blocks_missing_meta, ray.get(meta_futures)):
                self.get_block_data(block).meta = meta
            self.blocks_missing_meta = []

    def discard_block(self, block_data: BlockData):
        del self.blocks[block_data.block]


BinID = Any


class Bin:
    id: BinID
    capacity: int
    allocated_blocks: List[BlockData]
    allocated_size: int

    def __init__(self, id: BinID, capacity: int):
        self.id = id
        self.capacity = capacity
        self.allocated_blocks = []
        self.allocated_size = 0

    def __eq__(self, other):
        if not isinstance(other, Bin):
            return False
        return self.id == other.id

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}: id={self.id}, cap={self.capacity}"

    def __hash__(self):
        return hash(self.id)

    @property
    def slack(self):
        return self.capacity - self.allocated_size

    def can_allocate(self, size: int, overallocate=False):
        if overallocate:
            return self.slack > 0
        else:
            return size <= self.slack

    def try_allocate(self, block: BlockData, overallocate=False):
        if self.can_allocate(block.num_rows, overallocate=overallocate):
            self.allocate(block)
            return True
        return False

    def allocate(self, block: BlockData):
        self.allocated_blocks.append(block)
        self.allocated_size += block.num_rows
        block.set_allocated()


class BinSet:
    bins: Dict[BinID, Bin]

    def __init__(self, bins: List[Bin]):
        self.bins = {bin_.id: bin_ for bin_ in bins}

    def is_empty(self):
        return len(self.bins) == 0

    def get_bin(self, key: Union[BinID, int]):
        try:
            return self.bins[key]
        except KeyError:
            return list(self.bins.values())[key]

    def __iter__(self):
        return iter(self.bins.values())

    def __getitem__(self, key):
        return self.get_bin(key)

    def get_bin_subset(self, bin_ids: List[BinID]):
        return BinSet([self.get_bin(bin_id) for bin_id in bin_ids])

    def get_bin_with_largest_slack(self):
        return max(
            self.bins.values(), key=lambda bin_: bin_.slack)

    def get_bin_with_smallest_slack(self):
        return min(
            self.bins.values(), key=lambda bin_: bin_.slack)

    def get_bins_ordered_by_slack(self, filter_full=True, descending=True):
        bins = iter(self)
        if filter_full:
            bins = filter(lambda bin_: bin_.slack > 0, bins)
        return sorted(
            bins, key=lambda bin_: bin_.slack,
            reverse=descending)


def allocate_blocks_to_primary_actors(
        block_set, actor_bin_set, actors_by_node_id, overallocate=False):
    primary_actors_by_block = {
        b: actors_by_node_id[b.preferred_nodes[0]]
        for b in block_set.get_unallocated_blocks()}
    _allocate_blocks_to_actors(
        block_set, actor_bin_set, primary_actors_by_block,
        overallocate=overallocate)


def allocate_blocks_to_secondary_actors(
        block_set, actor_bin_set, actors_by_node_id, overallocate=False):
    secondary_actors_by_block = {
        b: itertools.chain.from_iterable(
            actors_by_node_id[node_id]
            for node_id in b.preferred_nodes[1:])
        for b in block_set.get_unallocated_blocks()}
    _allocate_blocks_to_actors(
        block_set, actor_bin_set, secondary_actors_by_block,
        overallocate=overallocate)


def allocate_blocks_to_arbitrary_actors(block_set, actor_bin_set):
    # Sort blocks into decreasing order, yielding first-fit-decreasing
    # packing.
    blocks = block_set.get_unallocated_blocks_ordered_by_size()
    for block in blocks:
        # Get the actor with the most allocation availability.
        # TODO(Clark): Try out best-fit rather than worst-fit.
        actor_bin = actor_bin_set.get_bin_with_largest_slack()
        # Allocate the block to this bin.
        actor_bin.allocate(block)


def _allocate_blocks_to_actors(
        block_set, actor_bin_set, actors_by_block, overallocate=False):
    # Sort blocks into decreasing order, yielding first-fit-decreasing
    # packing.
    blocks = block_set.get_unallocated_blocks_ordered_by_size()
    # Allocate blocks to the actor with the most allocation
    # availability that resides within the given location slice.
    for block in blocks:
        actors = actors_by_block[block]
        actor_bin_subset = actor_bin_set.get_bin_subset(actors)
        # Get the actor with the most allocation availability.
        # TODO(Clark): Try out best-fit rather than worst-fit.
        actor_bin = actor_bin_subset.get_bin_with_largest_slack()
        # Try to allocate the block to this bin.
        actor_bin.try_allocate(block, overallocate=overallocate)


def split_blocks_over_actors(
        block_set, actor_bin_set, actors_by_node_id, addresses_by_node_id):
    # Assumed invariant: No full block will fit in a local bin.
    # Split the blocks, attempting to allocate the splits to
    # actors that are located on nodes that already have the block.
    for block in block_set.get_unallocated_blocks():
        splits, leftover = _split_block_over_actor_bins_by_node(
            block, actor_bin_set, actors_by_node_id, addresses_by_node_id)
        if splits:
            block_set.discard_block(block)
        if leftover is not None:
            leftover_block, leftover_metadata, leftover_size = leftover
            leftover_block_data = BlockData(
                leftover_block,
                leftover_metadata,
                leftover_size)
            block_set.add_block_data(leftover_block_data)


def _split_block_over_actor_bins_by_node(
        block, actor_bin_set, actors_by_node_id, addresses_by_node_id):
    splits = []
    offset = 0
    for node_id in block.preferred_nodes:
        if offset >= block.num_rows:
            break
        node_address = addresses_by_node_id[node_id]
        actors = actors_by_node_id[node_id]
        actor_bin_subset = actor_bin_set.get_bin_subset(actors)
        # We give a custom node resource in order to ensure that this
        # splitting task is scheduled on the same node as the block.
        new_splits, offset = _split_block_over_actor_bins(
            block, actor_bin_subset, {f"node:{node_address}": 0.001}, offset)
        splits.extend(new_splits)
    if splits and offset < block.num_rows:
        # There's a leftover slice of the block that wasn't allocated
        # in this round, save it for allocation in the next round.
        (
            leftover_block,
            leftover_metadata) = _cached_split_block_at_indices(
                ).options(
                    num_returns=2,
                    resources={f"node:{node_address}": 0.001}
                ).remote(
                    block.block, block.meta, [offset, block.num_rows])
        leftover_size = block.num_rows - offset
        leftover = (leftover_block, leftover_metadata, leftover_size)
    else:
        leftover = None
    return splits, leftover


def _split_block_over_actor_bins(
        block, actor_bin_set, split_resources=None, offset=0):
    split_indices, actor_bins, offset = _get_block_split_indices(
        actor_bin_set, block.num_rows, offset)
    split_sizes = _get_split_sizes(split_indices)
    # Submit splitting task.
    # TODO(Clark): Some Ray clusters won't have access to node
    # IDs (e.g. if their "node" primitive is a container); should
    # we expose a custom resource hook here? How would that work?
    # We could also fall back to relying on locality-aware
    # scheduling in that case, but that could result in a lot of
    # unnecessary data transfer if the cluster is busy.
    block_splits = _create_splits(
        block.block, block.meta, split_indices,
        split_resources=split_resources)
    for (block, _), actor_bin, num_rows in zip(
            block_splits, actor_bins, split_sizes):
        actor_bin.allocate_block(block, num_rows)
    return block_splits, offset


def split_blocks_over_arbitrary_actors(block_set, actor_bin_set):
    # Get blocks sorted by size (increasing).
    blocks = block_set.get_unallocated_blocks_ordered_by_size()
    for block in blocks:
        splits, offset = _split_block_over_actor_bins(
            block, actor_bin_set)
        assert splits, splits
        block_set.discard_block(block)


def rebalance_blocks(block_set, actor_bin_set, target_block_size):
    for actor_bin in iter(actor_bin_set):
        blocks = actor_bin.allocated_blocks
        blocks = sorted(
            blocks, key=lambda b: b.num_rows, reverse=True)
        idx = next(
            i for i, b in enumerate(blocks) if b.num_rows < target_block_size)
        larger_blocks, smaller_blocks = blocks[:idx], blocks[idx:]
        new_allocation_large, small_leftovers = _rebalance_larger_blocks(
            larger_blocks, block_set, target_block_size)
        # Move on to blocks smaller than target.
        smaller_blocks += small_leftovers
        # Order blocks largest to smallest for first-fit-decreasing
        # packing.
        smaller_blocks = sorted(
            smaller_blocks, lambda b: b.num_rows, reverse=True)
        new_allocation_small = _rebalance_smaller_blocks(
            smaller_blocks, block_set, target_block_size)
        new_allocation = new_allocation_large + new_allocation_small
        # assert len(new_allocation) == blocks_per_split
        assert all(
            b.num_rows == target_block_size
            for b in new_allocation)
        actor_bin.allocated_blocks = new_allocation


def _rebalance_larger_blocks(blocks, block_set, target_block_size):
    new_allocation = []
    leftovers = []
    # First, process blocks larger than or equal to target.
    for block in enumerate(blocks):
        size = block.num_rows
        assert size >= target_block_size
        if size == target_block_size:
            new_allocation.append(block)
            continue
        # Case: size > target_block_size
        split_indices = list(range(0, size, target_block_size))
        if split_indices[-1] < size:
            split_indices.append(size)
        split_sizes = _get_split_sizes(split_indices)
        block_splits = _create_splits(
            block.block, block.meta, split_indices)
        last_split_size = split_sizes[-1]
        if last_split_size < target_block_size:
            # Handle leftover block.
            leftover_block, leftover_meta = block_splits.pop()
            split_sizes.pop()
            leftovers.append(
                BlockData(
                    leftover_block, leftover_meta, last_split_size))
        for (block, meta), split_size in zip(block_splits, split_sizes):
            new_block_data = BlockData(block, meta, split_size)
            new_allocation.append(new_block_data)
            block_set.add_block_data(new_block_data)
        block_set.discard_block(block)
    return new_allocation, leftovers


def _rebalance_smaller_blocks(blocks, block_set, target_block_size):
    new_allocation = []
    # This merge buffer is added to until we've reached the
    # target_block_size target, at which point we merge all of the
    # blocks and clear the buffer.
    # TODO(Clark): Add an abstraction (class) for the merge buffer?
    merge_buffer = []
    merge_buffer_size = 0
    # We use two-pointer loop to allocate blocks to the buffer, where
    # we first try to allocate the larger (left) block, then try to
    # allocate the smaller (right) block, and if neither fit, then we
    # split the smaller (right) block to add a sub-block that does fit.
    # This naive method approximates an optimal packing of blocks into
    # target_block_size bins that minimizes splitting.
    left = 0
    right = len(blocks) - 1
    while left < right:
        left_block = blocks[left]
        left_size = left_block.num_rows
        right_block = blocks[right]
        right_size = right_block.num_rows
        if merge_buffer_size + left_size <= target_block_size:
            # The left (larger) block can fit into the merge buffer,
            # so add it to the buffer and increment the left pointer.
            merge_buffer.append(left_block)
            merge_buffer_size += left_size
            left += 1
        elif merge_buffer_size + right_size <= target_block_size:
            # The right (smaller) block can fit into the merge buffer,
            # so add it to the buffer and increment the right pointer.
            merge_buffer.append(right_block)
            merge_buffer_size += right_size
            right -= 1
        else:
            # Neither the left nor the right block fit in the merge
            # buffer, so we split right (smaller) block into a block
            # that will fit into the buffer and a leftover block that
            # we add back into the candidate block list.

            # Submit splitting task.
            diff = target_block_size - merge_buffer_size
            (
                block, meta, leftover_block,
                leftover_meta) = _cached_split_block_at_indices(
                    ).options(num_return=4).remote(
                        right_block.block,
                        right_block.meta,
                        [0, diff, right_size])
            block_set.discard_block(right_block)
            leftover_block_data = BlockData(
                leftover_block, leftover_meta, right_size - diff)
            block_set.add_block_data(leftover_block_data)
            block_data = BlockData(block, meta, diff)
            block_set.add_block_data(block_data)
            merge_buffer.append(block)
            merge_buffer_size += diff
            assert merge_buffer_size == target_block_size
            # We overwrite the old rightmost block and don't advance
            # the right pointer since (1) the old block can be
            # discarded, (2) the leftover block is guaranteed to be
            # smaller than the old block, and (3) the rightmost block
            # should be the smallest block in the to-be-allocated set.
            blocks[right] = leftover_block
        if merge_buffer_size == target_block_size:
            # TODO(Clark): Resolve metadata for blocks in the merge buffer
            # that resulted from splitting?
            input_files = set(
                itertools.chain(
                    b.meta.input_files
                    for b in merge_buffer))
            merged_block, merged_metadata = _cached_merge_blocks(
                ).remote(*[
                    b.block for b in merge_buffer],
                    input_files=input_files)
            for b in merge_buffer:
                block_set.discard_block(b)
            block_set.add_block_data(BlockData(
                merged_block, merged_metadata, target_block_size))
            new_allocation.append(merged_block)
            merge_buffer = []
            merge_buffer_size = 0
    return new_allocation


def _get_block_split_indices(actor_bin_set, block_size, offset=0):
    # Get block splits that would fill some or all of the remaining
    # slack in some or all of the actors that are on the same node
    # as this block.
    assert actor_bin_set.is_empty() or block_size > actor_bin_set[0].slack
    sorted_bins = actor_bin_set.get_bins_ordered_by_slack()
    splits = []
    cum_num_rows = offset
    idx = 0
    while cum_num_rows < block_size and idx < len(sorted_bins):
        actor_bin = sorted_bins[idx]
        cum_num_rows += actor_bin.slack
        splits.append((cum_num_rows, actor_bin))
        idx += 1
    if not splits:
        # No splits were made, so we move on to the next location.
        return None, None, None
    split_indices, actor_bins = zip(*splits)
    split_indices = [offset] + list(split_indices)
    # Bound final block by block size. This will also be the returned
    # offset.
    split_indices[-1] = min(split_indices[-1], block_size)
    return split_indices, list(actor_bins), split_indices[-1]
