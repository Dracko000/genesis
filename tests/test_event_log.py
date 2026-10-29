from genesis.experience.event_log import EventLog


def test_event_log_is_append_only_and_sequential(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    first = log.append("experiment_started", {"id": "E001"})
    second = log.append("metric", {"mse": 0.5})
    assert first.sequence == 0
    assert second.sequence == 1
    assert [event.event_type for event in log.read_all()] == [
        "experiment_started",
        "metric",
    ]


def test_event_log_survives_reopen(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(path).append("a", {"x": 1})
    reopened = EventLog(path)
    reopened.append("b", {"x": 2})
    events = reopened.read_all()
    assert [event.sequence for event in events] == [0, 1]
    assert reopened.tail(1)[0].payload == {"x": 2}


def test_event_log_tail_rejects_negative(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    try:
        log.tail(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
