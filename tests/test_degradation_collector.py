"""Degradations BabelDOC reports through log records reach the done event."""

import logging

from pdf2zh_next.high_level import _DegradationCollector


def _log(logger, message, marker=None):
    extra = None if marker is None else {"babeldoc_degradation": marker}
    logger.warning(message, extra=extra)


def test_collects_marked_records_once_each():
    logger = logging.getLogger("test.degradation.collect")
    collector = _DegradationCollector()
    logger.addHandler(collector)
    try:
        _log(logger, "a", {"kind": "unknown_operator", "detail": "zz"})
        _log(logger, "a again", {"kind": "unknown_operator", "detail": "zz"})
        _log(logger, "b", {"kind": "text_outside_text_object", "detail": "Tj"})
        _log(logger, "unmarked")
    finally:
        logger.removeHandler(collector)
    assert collector.degradations() == [
        {"kind": "unknown_operator", "detail": "zz"},
        {"kind": "text_outside_text_object", "detail": "Tj"},
    ]


def test_ignores_records_below_warning():
    logger = logging.getLogger("test.degradation.level")
    logger.setLevel(logging.DEBUG)
    collector = _DegradationCollector()
    logger.addHandler(collector)
    try:
        logger.info(
            "info",
            extra={"babeldoc_degradation": {"kind": "k", "detail": "d"}},
        )
    finally:
        logger.removeHandler(collector)
    assert collector.degradations() == []
