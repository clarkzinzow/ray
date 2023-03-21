import time
from typing import Dict, List

import ray
from ray.data.context import DatasetContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Resource requests are considered stale after this number of seconds, and
# will be purged.
RESOURCE_REQUEST_TIMEOUT = 60


@ray.remote(num_cpus=0, max_restarts=-1, max_task_retries=-1)
class AutoscalingRequester:
    """Actor to make resource requests to autoscaler for the datasets.

    The resource requests are set to timeout for 60s.
    For those live requests (i.e. those made in last 60s), we keep track of the
    last request made for each execution, which overrides all previous requests it
    made; then sum the requested amounts across all executions as the final request
    to the autoscaler.
    """

    def __init__(self):
        # execution_id -> (List[Dict], expiration timestamp)
        self._resource_requests = {}
        # TTL for requests.
        self._timeout = RESOURCE_REQUEST_TIMEOUT

    def request_resources(self, req: List[Dict], execution_id: str):
        # Purge expired requests before making request to autoscaler.
        self._purge()
        # For the same execution_id, we track the latest resource request and
        # the its expiration timestamp.
        self._resource_requests[execution_id] = (
            req,
            time.time() + self._timeout,
        )
        # We aggregate the resource requests across all execution_id's to Ray
        # autoscaler.
        ray.autoscaler.sdk.request_resources(bundles=self._aggregate_requests())

    def _purge(self):
        # Purge requests that are stale.
        now = time.time()
        for k, (_, t) in list(self._resource_requests.items()):
            if t < now:
                self._resource_requests.pop(k)

    def _aggregate_requests(self) -> List[Dict]:
        self._purge()
        req = []
        for _, (r, _) in self._resource_requests.items():
            req.extend(r)
        return req

    def _test_set_timeout(self, ttl):
        """Set the timeout. This is for test only"""
        self._timeout = ttl


def get_or_create_autoscaling_requester_actor():
    ctx = DatasetContext.get_current()
    scheduling_strategy = ctx.scheduling_strategy
    # Pin the stats actor to the local node so it fate-shares with the driver.
    # Note: for Ray Client, the ray.get_runtime_context().get_node_id() should
    # point to the head node.
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        ray.get_runtime_context().get_node_id(),
        soft=True,
    )
    return AutoscalingRequester.options(
        name="AutoscalingRequester",
        namespace="AutoscalingRequester",
        get_if_exists=True,
        lifetime="detached",
        scheduling_strategy=scheduling_strategy,
    ).remote()
