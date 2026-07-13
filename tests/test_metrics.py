import pytest

from velacl.metrics import classification_metrics, continual_metrics


def test_classification_metrics_known_values():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_continual_forgetting():
    result = continual_metrics([[0.8, 0.7], [0.2, 0.6]])
    assert result["average_accuracy"] == pytest.approx(0.65)
    assert result["average_forgetting"] == pytest.approx(0.05)
