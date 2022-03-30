import math
from typing import Callable, Optional, List, TYPE_CHECKING

from ray.util.annotations import PublicAPI
from ray.data.block import (
    T,
    U,
    Block,
    BlockAccessor,
    KeyType,
    AggType,
    KeyFn,
    _validate_key_fn,
)
from ray.data.impl.null_aggregate import (
    _null_wrap_init,
    _null_wrap_merge,
    _null_wrap_accumulate_block,
    _null_wrap_finalize,
)

if TYPE_CHECKING:
    from ray.data import Dataset


@PublicAPI
class AggregateFn(object):
    def __init__(
        self,
        init: Callable[[KeyType], AggType],
        merge: Callable[[AggType, AggType], AggType],
        accumulate: Callable[[AggType, T], AggType] = None,
        accumulate_block: Callable[[AggType, Block[T]], AggType] = None,
        finalize: Callable[[AggType], U] = lambda a: a,
        name: Optional[str] = None,
    ):
        """Defines an aggregate function in the accumulator style.

        Aggregates a collection of inputs of type T into
        a single output value of type U.
        See https://www.sigops.org/s/conferences/sosp/2009/papers/yu-sosp09.pdf
        for more details about accumulator-based aggregation.

        Args:
            init: This is called once for each group to return the empty accumulator.
                For example, an empty accumulator for a sum would be 0.
            merge: This may be called multiple times, each time to merge
                two accumulators into one.
            accumulate: This is called once per row of the same group.
                This combines the accumulator and the row, returns the updated
                accumulator. Exactly one of accumulate and accumulate_block must
                be provided.
            accumulate_block: This is used to calculate the aggregation for a
                single block, and is vectorized alternative to accumulate. This will be
                given the empty accumulator and the entire block, allowing for
                vectorized aggregation of the block. Exactly one of accumulate and
                accumulate_block must be provided.
            finalize: This is called once to compute the final aggregation
                result from the fully merged accumulator.
            name: The name of the aggregation. This will be used as the output
                column name in the case of Arrow dataset.
        """
        if (accumulate is None and accumulate_block is None) or (
            accumulate is not None and accumulate_block is not None
        ):
            raise ValueError(
                "Exactly one of accumulate or accumulate_block must be provided."
            )
        if accumulate_block is None:

            def accumulate_block(a: AggType, block_acc: BlockAccessor[T]) -> AggType:
                for r in block_acc.iter_rows():
                    a = accumulate(a, r)
                return a

        self.init = init
        self.merge = merge
        self.accumulate_block = accumulate_block
        self.finalize = finalize
        self.name = name

    def _validate(self, ds: "Dataset") -> None:
        """Raise an error if this cannot be applied to the given dataset."""
        pass


class _AggregateOnKeyBase(AggregateFn):
    def _set_key_fn(self, on: KeyFn):
        self._key_fn = on

    def _validate(self, ds: "Dataset") -> None:
        _validate_key_fn(ds, self._key_fn)


@PublicAPI
class Count(AggregateFn):
    """Defines count aggregation."""

    def __init__(self):
        super().__init__(
            init=lambda k: 0,
            accumulate=lambda a, r: a + 1,
            merge=lambda a1, a2: a1 + a2,
            name="count()",
        )


@PublicAPI
class Sum(_AggregateOnKeyBase):
    """Defines sum aggregation."""

    def __init__(self, on: Optional[KeyFn] = None, ignore_nulls: bool = True):
        self._set_key_fn(on)

        super().__init__(
            init=_null_wrap_init(lambda k: 0),
            merge=_null_wrap_merge(ignore_nulls, lambda a1, a2: a1 + a2),
            accumulate_block=_null_wrap_accumulate_block(
                ignore_nulls,
                lambda block_acc: block_acc.sum(on, ignore_nulls),
            ),
            finalize=_null_wrap_finalize(lambda a: a),
            name=(f"sum({str(on)})"),
        )


@PublicAPI
class Min(_AggregateOnKeyBase):
    """Defines min aggregation."""

    def __init__(self, on: Optional[KeyFn] = None, ignore_nulls: bool = True):
        self._set_key_fn(on)

        super().__init__(
            init=_null_wrap_init(lambda k: float("inf")),
            merge=_null_wrap_merge(ignore_nulls, min),
            accumulate_block=_null_wrap_accumulate_block(
                ignore_nulls,
                lambda block_acc: block_acc.min(on, ignore_nulls),
            ),
            finalize=_null_wrap_finalize(lambda a: a),
            name=(f"min({str(on)})"),
        )


@PublicAPI
class Max(_AggregateOnKeyBase):
    """Defines max aggregation."""

    def __init__(self, on: Optional[KeyFn] = None, ignore_nulls: bool = True):
        self._set_key_fn(on)

        super().__init__(
            init=_null_wrap_init(lambda k: float("-inf")),
            merge=_null_wrap_merge(ignore_nulls, max),
            accumulate_block=_null_wrap_accumulate_block(
                ignore_nulls,
                lambda block_acc: block_acc.max(on, ignore_nulls),
            ),
            finalize=_null_wrap_finalize(lambda a: a),
            name=(f"max({str(on)})"),
        )


@PublicAPI
class Mean(_AggregateOnKeyBase):
    """Defines mean aggregation."""

    def __init__(self, on: Optional[KeyFn] = None, ignore_nulls: bool = True):
        self._set_key_fn(on)

        def vectorized_mean(block_acc: BlockAccessor[T]) -> AggType:
            sum_ = block_acc.sum(on, ignore_nulls)
            if sum_ is None:
                return None
            count = block_acc.count(on, ignore_nulls)
            return [sum_, count]

        super().__init__(
            init=_null_wrap_init(lambda k: [0, 0]),
            merge=_null_wrap_merge(
                ignore_nulls, lambda a1, a2: [a1[0] + a2[0], a1[1] + a2[1]]
            ),
            accumulate_block=_null_wrap_accumulate_block(
                ignore_nulls,
                vectorized_mean,
            ),
            finalize=_null_wrap_finalize(lambda a: a[0] / a[1]),
            name=(f"mean({str(on)})"),
        )


@PublicAPI
class Std(_AggregateOnKeyBase):
    """Defines standard deviation aggregation.

    Uses Welford's online method for an accumulator-style computation of the
    standard deviation. This method was chosen due to it's numerical
    stability, and it being computable in a single pass.
    This may give different (but more accurate) results than NumPy, Pandas,
    and sklearn, which use a less numerically stable two-pass algorithm.
    See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(
        self,
        on: Optional[KeyFn] = None,
        ddof: int = 1,
        ignore_nulls: bool = True,
    ):
        self._set_key_fn(on)

        def vectorized_agg(block_acc: BlockAccessor[T]) -> AggType:
            count = block_acc.count(on, ignore_nulls)
            if count == 0:
                return None
            sum_ = block_acc.sum(on, ignore_nulls)
            if sum_ is None:
                return None
            mean = sum_ / count
            M2 = block_acc.sum_of_squared_diffs_from_mean(on, ignore_nulls, mean)
            return [M2, mean, count]

        def merge(a: List[float], b: List[float]):
            # Merges two accumulations into one.
            # See
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            M2_a, mean_a, count_a = a
            M2_b, mean_b, count_b = b
            delta = mean_b - mean_a
            count = count_a + count_b
            # NOTE: We use this mean calculation since it's more numerically
            # stable than mean_a + delta * count_b / count, which actually
            # deviates from Pandas in the ~15th decimal place and causes our
            # exact comparison tests to fail.
            mean = (mean_a * count_a + mean_b * count_b) / count
            # Update the sum of squared differences.
            M2 = M2_a + M2_b + (delta ** 2) * count_a * count_b / count
            return [M2, mean, count]

        def finalize(a: List[float]):
            # Compute the final standard deviation from the accumulated
            # sum of squared differences from current mean and the count.
            M2, mean, count = a
            if count < 2:
                return 0.0
            return math.sqrt(M2 / (count - ddof))

        super().__init__(
            init=_null_wrap_init(lambda k: [0, 0, 0]),
            merge=_null_wrap_merge(ignore_nulls, merge),
            accumulate_block=_null_wrap_accumulate_block(
                ignore_nulls,
                vectorized_agg,
            ),
            finalize=_null_wrap_finalize(finalize),
            name=(f"std({str(on)})"),
        )


def _to_on_fn(on: Optional[KeyFn]):
    if on is None:
        return lambda r: r
    elif isinstance(on, str):
        return lambda r: r[on]
    else:
        return on
