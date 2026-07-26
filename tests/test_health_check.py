from __future__ import annotations

from health_check import classify_broker_error


def test_classify_broker_error_marks_unauthorized_as_failure():
    detail, is_warning = classify_broker_error(Exception('{"message":"unauthorized."}'))
    assert is_warning is False
    assert "credentials rejected" in detail


def test_classify_broker_error_marks_missing_env_as_warning():
    detail, is_warning = classify_broker_error(Exception("Missing environment variables: ['ALPACA_API_KEY']"))
    assert is_warning is True
    assert "missing locally" in detail
