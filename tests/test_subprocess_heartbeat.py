"""The CPU meter behind the translation subprocess heartbeat.

The parent extends its silence timeout only when a heartbeat shows the
subprocess tree burning CPU, so what this meter counts decides both whether a
working translation survives and whether a stuck one still gets killed.
"""

import subprocess
import sys
import time

from pdf2zh_next.high_level import HEARTBEAT_MIN_CPU_DELTA_SECONDS
from pdf2zh_next.high_level import _cpu_progress_meter

# Enough iterations to still be running at the mid-flight sample on a fast
# machine.
BURN = "x=0\nfor i in range(200_000_000): x+=i"


def test_idle_process_stays_below_the_liveness_threshold():
    """A subprocess whose workers are all blocked must still time out."""
    sample = _cpu_progress_meter()
    base = sample()
    time.sleep(3.0)
    assert sample() - base < HEARTBEAT_MIN_CPU_DELTA_SECONDS


def test_in_process_work_counts():
    """pdf.save() runs in the subprocess itself, not in a helper."""
    sample = _cpu_progress_meter()
    base = sample()
    x = 0
    for i in range(100_000_000):
        x += i
    assert sample() - base >= HEARTBEAT_MIN_CPU_DELTA_SECONDS


def test_busy_descendant_counts_and_survives_its_exit():
    """Font subsetting and save(clean=True) each run in their own process.

    The counter must not fall back when one exits, or the next heartbeat looks
    like a stall.
    """
    sample = _cpu_progress_meter()
    base = sample()
    child = subprocess.Popen([sys.executable, "-c", BURN])
    try:
        time.sleep(3.0)
        during = sample() - base
    finally:
        child.wait()
    after_exit = sample() - base

    assert during >= HEARTBEAT_MIN_CPU_DELTA_SECONDS
    assert after_exit >= during


def test_meter_never_decreases_across_repeated_samples():
    sample = _cpu_progress_meter()
    readings = []
    child = subprocess.Popen([sys.executable, "-c", BURN])
    try:
        for _ in range(6):
            readings.append(sample())
            time.sleep(0.5)
    finally:
        child.wait()
    readings.append(sample())

    assert readings == sorted(readings)
