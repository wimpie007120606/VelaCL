from velacl.data import build_events


def test_stream_is_deterministic_and_complete():
    first = build_events(42)
    second = build_events(42)
    assert first == second
    assert {event.stage for event in first} == set(range(6))
    assert {event.language for event in first} == {"en", "af", "zu", "xh", "st"}
    assert all(event.id and event.timestamp and event.text for event in first)
    assert {event.split for event in first} == {"train", "test"}
