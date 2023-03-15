import collections
import pytest
import time
from unittest.mock import MagicMock

import ray
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    PhysicalOperator,
)
from ray.data._internal.execution.streaming_executor import (
    _debug_dump_topology,
    _validate_topology,
)
from ray.data._internal.execution.streaming_executor_state import (
    OpState,
    TopologyResourceUsage,
    DownstreamMemoryInfo,
    build_streaming_topology,
    process_completed_tasks,
    select_operator_to_run,
    _execution_allowed,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import make_ref_bundles
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


EMPTY_DOWNSTREAM_USAGE = collections.defaultdict(lambda: DownstreamMemoryInfo(0, 0))
NO_USAGE = TopologyResourceUsage(ExecutionResources(), EMPTY_DOWNSTREAM_USAGE)


@ray.remote
def sleep():
    time.sleep(999)


def make_transform(block_fn):
    def map_fn(block_iter):
        for block in block_iter:
            yield block_fn(block)

    return map_fn


def test_build_streaming_topology():
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(make_transform(lambda block: [b * 2 for b in block]), o2)
    topo = build_streaming_topology(o3, ExecutionOptions())
    assert len(topo) == 3, topo
    assert o1 in topo, topo
    assert not topo[o1].inqueues, topo
    assert topo[o1].outqueue == topo[o2].inqueues[0], topo
    assert topo[o2].outqueue == topo[o3].inqueues[0], topo
    assert list(topo) == [o1, o2, o3]


def test_disallow_non_unique_operators():
    inputs = make_ref_bundles([[x] for x in range(20)])
    # An operator [o1] cannot used in the same DAG twice.
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o4 = PhysicalOperator("test_combine", [o2, o3])
    with pytest.raises(ValueError):
        build_streaming_topology(o4, ExecutionOptions())


def test_process_completed_tasks():
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    topo = build_streaming_topology(o2, ExecutionOptions())

    # Test processing output bundles.
    assert len(topo[o1].outqueue) == 0, topo
    process_completed_tasks(topo)
    assert len(topo[o1].outqueue) == 20, topo

    # Test processing completed work items.
    sleep_ref = sleep.remote()
    done_ref = ray.put("done")
    o2.get_work_refs = MagicMock(return_value=[sleep_ref, done_ref])
    o2.notify_work_completed = MagicMock()
    o2.inputs_done = MagicMock()
    process_completed_tasks(topo)
    o2.notify_work_completed.assert_called_once_with(done_ref)
    o2.inputs_done.assert_not_called()

    # Test input finalization.
    o2.get_work_refs = MagicMock(return_value=[done_ref])
    o2.notify_work_completed = MagicMock()
    o2.inputs_done = MagicMock()
    o1.completed = MagicMock(return_value=True)
    topo[o1].outqueue.clear()
    process_completed_tasks(topo)
    o2.notify_work_completed.assert_called_once_with(done_ref)
    o2.inputs_done.assert_called_once()


def test_select_operator_to_run():
    opt = ExecutionOptions()
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(make_transform(lambda block: [b * 2 for b in block]), o2)
    topo = build_streaming_topology(o3, opt)

    # Test empty.
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        is None
    )

    # Test backpressure based on queue length between operators.
    topo[o1].outqueue.append("dummy1")
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o2
    )
    topo[o1].outqueue.append("dummy2")
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o2
    )
    topo[o2].outqueue.append("dummy3")
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o3
    )

    # Test backpressure includes num active tasks as well.
    o3.num_active_work_refs = MagicMock(return_value=2)
    o3.internal_queue_size = MagicMock(return_value=0)
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o2
    )
    # Internal queue size is added to num active tasks.
    o3.num_active_work_refs = MagicMock(return_value=0)
    o3.internal_queue_size = MagicMock(return_value=2)
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o2
    )
    o2.num_active_work_refs = MagicMock(return_value=2)
    o2.internal_queue_size = MagicMock(return_value=0)
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o3
    )
    o2.num_active_work_refs = MagicMock(return_value=0)
    o2.internal_queue_size = MagicMock(return_value=2)
    assert (
        select_operator_to_run(topo, NO_USAGE, ExecutionResources(), True, "dummy")
        == o3
    )


def test_dispatch_next_task():
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o1_state = OpState(o1, [])
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    op_state = OpState(o2, [o1_state.outqueue])

    # TODO: test multiple inqueues with the union operator.
    op_state.inqueues[0].append("dummy1")
    op_state.inqueues[0].append("dummy2")

    o2.add_input = MagicMock()
    op_state.dispatch_next_task()
    assert o2.add_input.called_once_with("dummy1")

    o2.add_input = MagicMock()
    op_state.dispatch_next_task()
    assert o2.add_input.called_once_with("dummy2")


def test_debug_dump_topology():
    opt = ExecutionOptions()
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(make_transform(lambda block: [b * 2 for b in block]), o2)
    topo = build_streaming_topology(o3, opt)
    # Just a sanity check to ensure it doesn't crash.
    _debug_dump_topology(topo)


def test_validate_topology():
    opt = ExecutionOptions()
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(
        make_transform(lambda block: [b * -1 for b in block]),
        o1,
        compute_strategy=ray.data.ActorPoolStrategy(8, 8),
    )
    o3 = MapOperator.create(
        make_transform(lambda block: [b * 2 for b in block]),
        o2,
        compute_strategy=ray.data.ActorPoolStrategy(4, 4),
    )
    topo = build_streaming_topology(o3, opt)
    _validate_topology(topo, ExecutionResources())
    _validate_topology(topo, ExecutionResources(cpu=20))
    _validate_topology(topo, ExecutionResources(gpu=0))
    with pytest.raises(ValueError):
        _validate_topology(topo, ExecutionResources(cpu=10))


def test_execution_allowed():
    op = InputDataBuffer([])

    def stub(res: ExecutionResources) -> TopologyResourceUsage:
        return TopologyResourceUsage(res, EMPTY_DOWNSTREAM_USAGE)

    # CPU.
    op.incremental_resource_usage = MagicMock(return_value=ExecutionResources(cpu=1))
    assert _execution_allowed(
        op, stub(ExecutionResources(cpu=1)), ExecutionResources(cpu=2)
    )
    assert not _execution_allowed(
        op, stub(ExecutionResources(cpu=2)), ExecutionResources(cpu=2)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(cpu=2)), ExecutionResources(gpu=2)
    )

    # GPU.
    op.incremental_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=1, gpu=1)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(gpu=1)), ExecutionResources(gpu=2)
    )
    assert not _execution_allowed(
        op, stub(ExecutionResources(gpu=2)), ExecutionResources(gpu=2)
    )

    # Test conversion to indicator (0/1).
    op.incremental_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=100, gpu=100)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(gpu=1)), ExecutionResources(gpu=2)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(gpu=1.5)), ExecutionResources(gpu=2)
    )
    assert not _execution_allowed(
        op, stub(ExecutionResources(gpu=2)), ExecutionResources(gpu=2)
    )

    # Test conversion to indicator (0/1).
    op.incremental_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=0.1, gpu=0.1)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(gpu=1)), ExecutionResources(gpu=2)
    )
    assert _execution_allowed(
        op, stub(ExecutionResources(gpu=1.5)), ExecutionResources(gpu=2)
    )
    assert not _execution_allowed(
        op, stub(ExecutionResources(gpu=2)), ExecutionResources(gpu=2)
    )


def test_resource_constrained_triggers_autoscaling():
    from ray.data._internal.execution.autoscaling_requester import (
        get_or_create_autoscaling_requester_actor,
    )

    # Test that dispatch not being possible due to resource limits triggers a scale-up
    # request to the autoscaler.
    opt = ExecutionOptions()
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(
        make_transform(lambda block: [b * -1 for b in block]),
        o1,
    )
    o2.num_active_work_refs = MagicMock(return_value=1)
    # Mock operator current resource usage to add object store memory usage.
    o2.current_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=1, object_store_memory=1000)
    )
    o3 = MapOperator.create(
        make_transform(lambda block: [b * 2 for b in block]),
        o2,
    )
    o3.num_active_work_refs = MagicMock(return_value=1)
    o4 = MapOperator.create(
        make_transform(lambda block: [b * 3 for b in block]),
        o3,
        compute_strategy=ray.data.ActorPoolStrategy(1, 2),
        ray_remote_args={"num_gpus": 1},
    )
    o4.num_active_work_refs = MagicMock(return_value=1)
    o4.incremental_resource_usage = MagicMock(return_value=ExecutionResources(gpu=1))
    topo = build_streaming_topology(o4, opt)
    # Make sure only two operator's inqueues has data.
    topo[o2].inqueues[0].append("dummy")
    topo[o4].inqueues[0].append("dummy")
    selected_op = select_operator_to_run(
        topo,
        TopologyResourceUsage(
            ExecutionResources(cpu=2, gpu=1, object_store_memory=1000),
            EMPTY_DOWNSTREAM_USAGE,
        ),
        ExecutionResources(cpu=2, gpu=1, object_store_memory=1000),
        True,
        "1",
    )
    assert selected_op is None
    # We should request incremental resources for only o2, since it's the only op that's
    # ready to dispatch.
    ac = get_or_create_autoscaling_requester_actor()
    assert ray.get(ac._get_resource_requests.remote())["1"][0] == {"CPU": 3, "GPU": 2}


def test_select_ops_ensure_at_least_one_live_operator():
    opt = ExecutionOptions()
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(
        make_transform(lambda block: [b * -1 for b in block]),
        o1,
    )
    o3 = MapOperator.create(
        make_transform(lambda block: [b * 2 for b in block]),
        o2,
    )
    topo = build_streaming_topology(o3, opt)
    topo[o2].outqueue.append("dummy1")
    o1.num_active_work_refs = MagicMock(return_value=2)
    assert (
        select_operator_to_run(
            topo,
            TopologyResourceUsage(ExecutionResources(cpu=1), EMPTY_DOWNSTREAM_USAGE),
            ExecutionResources(cpu=1),
            True,
            "dummy",
        )
        is None
    )
    o1.num_active_work_refs = MagicMock(return_value=0)
    assert (
        select_operator_to_run(
            topo,
            TopologyResourceUsage(ExecutionResources(cpu=1), EMPTY_DOWNSTREAM_USAGE),
            ExecutionResources(cpu=1),
            True,
            "dummy",
        )
        is o3
    )
    assert (
        select_operator_to_run(
            topo,
            TopologyResourceUsage(ExecutionResources(cpu=1), EMPTY_DOWNSTREAM_USAGE),
            ExecutionResources(cpu=1),
            False,
            "dummy",
        )
        is None
    )


def test_configure_output_locality():
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(
        make_transform(lambda block: [b * 2 for b in block]),
        o2,
        compute_strategy=ray.data.ActorPoolStrategy(1, 1),
    )
    # No locality.
    build_streaming_topology(o3, ExecutionOptions(locality_with_output=False))
    assert o2._ray_remote_args.get("scheduling_strategy") is None
    assert o3._ray_remote_args.get("scheduling_strategy") == "SPREAD"

    # Current node locality.
    build_streaming_topology(o3, ExecutionOptions(locality_with_output=True))
    s1 = o2._get_runtime_ray_remote_args()["scheduling_strategy"]
    assert isinstance(s1, NodeAffinitySchedulingStrategy)
    assert s1.node_id == ray.get_runtime_context().get_node_id()
    s2 = o3._get_runtime_ray_remote_args()["scheduling_strategy"]
    assert isinstance(s2, NodeAffinitySchedulingStrategy)
    assert s2.node_id == ray.get_runtime_context().get_node_id()

    # Multi node locality.
    build_streaming_topology(
        o3, ExecutionOptions(locality_with_output=["node1", "node2"])
    )
    s1a = o2._get_runtime_ray_remote_args()["scheduling_strategy"]
    s1b = o2._get_runtime_ray_remote_args()["scheduling_strategy"]
    s1c = o2._get_runtime_ray_remote_args()["scheduling_strategy"]
    assert s1a.node_id == "node1"
    assert s1b.node_id == "node2"
    assert s1c.node_id == "node1"
    s2a = o3._get_runtime_ray_remote_args()["scheduling_strategy"]
    s2b = o3._get_runtime_ray_remote_args()["scheduling_strategy"]
    s2c = o3._get_runtime_ray_remote_args()["scheduling_strategy"]
    assert s2a.node_id == "node1"
    assert s2b.node_id == "node2"
    assert s2c.node_id == "node1"


def test_calculate_topology_usage():
    inputs = make_ref_bundles([[x] for x in range(20)])
    o1 = InputDataBuffer(inputs)
    o2 = MapOperator.create(make_transform(lambda block: [b * -1 for b in block]), o1)
    o3 = MapOperator.create(make_transform(lambda block: [b * 2 for b in block]), o2)
    o2.current_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=5, object_store_memory=500)
    )
    o3.current_resource_usage = MagicMock(
        return_value=ExecutionResources(cpu=10, object_store_memory=1000)
    )
    topo = build_streaming_topology(o3, ExecutionOptions())
    inputs[0].size_bytes = MagicMock(return_value=200)
    topo[o2].outqueue = [inputs[0]]
    usage = TopologyResourceUsage.of(topo)
    assert len(usage.downstream_memory_usage) == 3, usage
    assert usage.overall == ExecutionResources(15, 0, 1700)
    assert usage.downstream_memory_usage[o1].object_store_memory == 1700, usage
    assert usage.downstream_memory_usage[o1].topology_fraction == 1, usage
    assert usage.downstream_memory_usage[o2].object_store_memory == 1700, usage
    assert usage.downstream_memory_usage[o2].topology_fraction == 1, usage
    assert usage.downstream_memory_usage[o3].object_store_memory == 1000, usage
    assert usage.downstream_memory_usage[o3].topology_fraction == 0.5, usage


def test_execution_allowed_downstream_aware_memory_throttling():
    op = InputDataBuffer([])
    op.incremental_resource_usage = MagicMock(return_value=ExecutionResources())
    # Below global.
    assert _execution_allowed(
        op,
        TopologyResourceUsage(
            ExecutionResources(object_store_memory=1000),
            {op: DownstreamMemoryInfo(1, 1000)},
        ),
        ExecutionResources(object_store_memory=1100),
    )
    # Above global.
    assert not _execution_allowed(
        op,
        TopologyResourceUsage(
            ExecutionResources(object_store_memory=1000),
            {op: DownstreamMemoryInfo(1, 1000)},
        ),
        ExecutionResources(object_store_memory=900),
    )
    # Above global, but below downstream quota of 50%.
    assert _execution_allowed(
        op,
        TopologyResourceUsage(
            ExecutionResources(object_store_memory=1000),
            {op: DownstreamMemoryInfo(0.5, 400)},
        ),
        ExecutionResources(object_store_memory=900),
    )
    # Above global, and above downstream quota of 50%.
    assert not _execution_allowed(
        op,
        TopologyResourceUsage(
            ExecutionResources(object_store_memory=1000),
            {op: DownstreamMemoryInfo(0.5, 600)},
        ),
        ExecutionResources(object_store_memory=900),
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
