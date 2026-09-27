import uuid

from dev_double.providers.std.decision.clock_std_impl import ClockStdImpl
from dev_double.providers.std.decision.id_provider_std_impl import IdProviderStdImpl


def test_clock_is_monotonic():
    clock = ClockStdImpl()
    a, b = clock.now(), clock.now()
    assert b >= a


def test_ids_are_unique_uuids():
    ids = IdProviderStdImpl()
    a, b = ids.new_id(), ids.new_id()
    assert a != b
    assert str(uuid.UUID(a)) == a
