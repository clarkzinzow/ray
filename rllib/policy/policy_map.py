from collections import deque
import gym
import os
import threading
from typing import Callable, Dict, Optional, Set, Type, TYPE_CHECKING, Union

import ray.cloudpickle as pickle
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.utils.annotations import PublicAPI, override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import create_policy_for_framework
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PolicyID,
)

tf1, tf, tfv = try_import_tf()

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


@PublicAPI
class PolicyMap(dict):
    """Maps policy IDs to Policy objects.

    Thereby, keeps n policies in memory and - when capacity is reached -
    writes the least recently used to disk. This allows adding 100s of
    policies to a Algorithm for league-based setups w/o running out of memory.
    """

    def __init__(
        self,
        worker_index: int,
        num_workers: int,
        capacity: Optional[int] = None,
        path: Optional[str] = None,
        policy_config=None,  # deprecated arg: All Policies now bring their own config.
        session_creator: Optional[Callable[[], "tf1.Session"]] = None,
        seed: Optional[int] = None,
    ):
        """Initializes a PolicyMap instance.

        Args:
            worker_index: The worker index of the RolloutWorker this map
                resides in.
            num_workers: The total number of remote workers in the
                WorkerSet to which this map's RolloutWorker belongs to.
            capacity: The maximum number of policies to hold in memory.
                The least used ones are written to disk/S3 and retrieved
                when needed.
            path: The path to store the policy pickle files to. Files
                will have the name: [policy_id].[worker idx].policy.pkl.
            session_creator: An optional
                tf1.Session creation callable.
            seed: An optional seed (used to seed tf policies).
        """
        if policy_config is not None:
            deprecation_warning(
                old="PolicyMap(policy_config=..)",
                error=True,
            )

        super().__init__()

        self.worker_index = worker_index
        self.num_workers = num_workers
        self.session_creator = session_creator
        self.seed = seed

        # The file extension for stashed policies (that are no longer available
        # in-memory but can be reinstated any time from storage).
        self.extension = f".{self.worker_index}.policy.pkl"

        # Dictionary of keys that may be looked up (cached or not).
        self.valid_keys: Set[str] = set()
        # The actual cache with the in-memory policy objects.
        self.cache: Dict[str, Policy] = {}
        # The doubly-linked list holding the currently in-memory objects.
        self.deque = deque(maxlen=capacity or 10)
        # The file path where to store overflowing policies.
        self.path = path or "."
        # The orig classes/obs+act spaces, and config overrides of the
        # Policies.
        self.policy_specs: Dict[PolicyID, PolicySpec] = {}

        # Lock used for locking some methods on the object-level.
        # This prevents possible race conditions when accessing the map
        # and the underlying structures, like self.deque and others.
        self._lock = threading.RLock()

    def insert_policy(
        self, policy_id: PolicyID, policy: Policy, config_override=None  # deprecated
    ) -> None:
        if config_override is not None:
            deprecation_warning(
                old="PolicyMap.insert_policy(config_override=..)",
                error=True,
            )

        self[policy_id] = policy

        # Store spec (class, obs-space, act-space, and config overrides) such
        # that the map will be able to reproduce on-the-fly added policies
        # from disk.
        self.policy_specs[policy_id] = PolicySpec(
            policy_class=type(policy),
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            config=policy.config,
        )

    def create_policy(
        self,
        policy_id: PolicyID,
        policy_cls: Type["Policy"],
        observation_space: gym.Space,
        action_space: gym.Space,
        config_override,  # deprecated arg
        merged_config: Union["AlgorithmConfig", AlgorithmConfigDict],
    ) -> None:
        """Creates a new policy and stores it to the cache.

        Args:
            policy_id: The policy ID. This is the key under which
                the created policy will be stored in this map.
            policy_cls: The (original) policy class to use.
                This may still be altered in case tf-eager (and tracing)
                is used.
            observation_space: The observation space of the
                policy.
            action_space: The action space of the policy.
            merged_config: The config object (or complete config dict) for the policy
                to use.
        """
        if config_override is not None:
            deprecation_warning(
                old="PolicyMap.create_policy(config_override=..)",
                error=True,
            )
        _class = get_tf_eager_cls_if_necessary(policy_cls, merged_config)

        policy = create_policy_for_framework(
            policy_id,
            _class,
            merged_config,
            observation_space,
            action_space,
            self.worker_index,
            self.session_creator,
            self.seed,
        )
        self.insert_policy(policy_id, policy)

    @with_lock
    @override(dict)
    def __getitem__(self, item):
        # Never seen this key -> Error.
        if item not in self.valid_keys:
            raise KeyError(f"PolicyID '{item}' not found in this PolicyMap!")

        # Item already in cache -> Rearrange deque (least recently used) and
        # return.
        if item in self.cache:
            self.deque.remove(item)
            self.deque.append(item)
        # Item not currently in cache -> Get from disk and - if at capacity -
        # remove leftmost one.
        else:
            self._read_from_disk(policy_id=item)

        return self.cache[item]

    @with_lock
    @override(dict)
    def __setitem__(self, key, value):
        # Item already in cache -> Rearrange deque (least recently used).
        if key in self.cache:
            self.deque.remove(key)
            self.deque.append(key)
            self.cache[key] = value
        # Item not currently in cache -> store new value and - if at capacity -
        # remove leftmost one.
        else:
            # Cache at capacity -> Drop leftmost item.
            if len(self.deque) == self.deque.maxlen:
                self._stash_to_disk()
            self.deque.append(key)
            self.cache[key] = value
        self.valid_keys.add(key)

    @with_lock
    @override(dict)
    def __delitem__(self, key):
        # Make key invalid.
        self.valid_keys.remove(key)
        # Remove policy from memory if currently cached.
        if key in self.cache:
            policy = self.cache[key]
            self._close_session(policy)
            del self.cache[key]
        # Remove file associated with the policy, if it exists.
        filename = self.path + "/" + key + self.extension
        if os.path.isfile(filename):
            os.remove(filename)

    @override(dict)
    def __iter__(self):
        return iter(self.keys())

    @override(dict)
    def items(self):
        """Iterates over all policies, even the stashed-to-disk ones."""

        def gen():
            for key in self.valid_keys:
                yield (key, self[key])

        return gen()

    @override(dict)
    def keys(self):
        self._lock.acquire()
        ks = list(self.valid_keys)
        self._lock.release()

        def gen():
            for key in ks:
                yield key

        return gen()

    @override(dict)
    def values(self):
        self._lock.acquire()
        vs = [self[k] for k in self.valid_keys]
        self._lock.release()

        def gen():
            for value in vs:
                yield value

        return gen()

    @with_lock
    @override(dict)
    def update(self, __m, **kwargs):
        for k, v in __m.items():
            self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    @with_lock
    @override(dict)
    def get(self, key):
        if key not in self.valid_keys:
            return None
        return self[key]

    @with_lock
    @override(dict)
    def __len__(self):
        """Returns number of all policies, including the stashed-to-disk ones."""
        return len(self.valid_keys)

    @with_lock
    @override(dict)
    def __contains__(self, item):
        return item in self.valid_keys

    def _stash_to_disk(self):
        """Writes the least-recently used policy to disk and rearranges cache.

        Also closes the session - if applicable - of the stashed policy.
        """
        # Get least recently used policy (all the way on the left in deque).
        delkey = self.deque.popleft()
        policy = self.cache[delkey]
        # Get its state for writing to disk.
        policy_state = policy.get_state()
        # Closes policy's tf session, if any.
        self._close_session(policy)
        # Remove from memory. This will clear the tf Graph as well.
        del self.cache[delkey]
        # Write state to disk.
        with open(self.path + "/" + delkey + self.extension, "wb") as f:
            pickle.dump(policy_state, file=f)

    def _read_from_disk(self, policy_id):
        """Reads a policy ID from disk and re-adds it to the cache."""
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        # Make sure this policy ID is not in the cache right now.
        assert policy_id not in self.cache
        # Read policy state from disk.
        with open(self.path + "/" + policy_id + self.extension, "rb") as f:
            policy_state = pickle.load(f)

        # Get class and config override.
        config = self.policy_specs[policy_id].config
        if isinstance(config, AlgorithmConfig):
            config = config.to_dict()

        # Create policy object (from its spec: cls, obs-space, act-space,
        # config).
        self.create_policy(
            policy_id,
            self.policy_specs[policy_id].policy_class,
            self.policy_specs[policy_id].observation_space,
            self.policy_specs[policy_id].action_space,
            config_override=None,  # deprecated, must be None
            merged_config=config,
        )
        # Restore policy's state.
        policy = self[policy_id]
        policy.set_state(policy_state)

    def _close_session(self, policy):
        sess = policy.get_session()
        # Closes the tf session, if any.
        if sess is not None:
            sess.close()
