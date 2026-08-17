"""Tests for the drift-free log/eval cadence."""

from common.schedule import should_log_and_eval


def test_aligned_cadences() -> None:
    do_log, do_eval = should_log_and_eval(200, log_every_steps=20, eval_every_steps=200)
    assert do_log and do_eval
    do_log, do_eval = should_log_and_eval(20, log_every_steps=20, eval_every_steps=200)
    assert do_log and not do_eval
    do_log, do_eval = should_log_and_eval(15, log_every_steps=20, eval_every_steps=200)
    assert not do_log and not do_eval


def test_misaligned_eval_never_skipped() -> None:
    # The bug this guards against: eval gated inside the logging branch is
    # silently skipped when eval_every_steps isn't a multiple of log_every_steps.
    eval_steps = [
        step
        for step in range(1, 1001)
        if should_log_and_eval(step, log_every_steps=100, eval_every_steps=250)[1]
    ]
    assert eval_steps == [250, 500, 750, 1000]
    # Every eval step also logs, so the eval results are always recorded.
    for step in eval_steps:
        assert should_log_and_eval(step, log_every_steps=100, eval_every_steps=250)[0]


def test_zero_disables_cadence() -> None:
    assert should_log_and_eval(100, log_every_steps=0, eval_every_steps=0) == (
        False,
        False,
    )
    assert should_log_and_eval(100, log_every_steps=0, eval_every_steps=100) == (
        True,
        True,
    )
