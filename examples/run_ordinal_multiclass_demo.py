"""Demonstrate multiclass SVOR/NPSVOR on a synthetic ordinal dataset."""

from __future__ import annotations

import json

import numpy as np

from hcesvm.models.npsvor import NPSVORQP
from hcesvm.models.svor import SVORImplicitQP
from hcesvm.utils.evaluator import calculate_accuracy
from hcesvm.utils.ordinal_synthetic import make_staircase_ordinal_data


def main() -> int:
    X, y = make_staircase_ordinal_data(
        num_classes=5,
        samples_per_class=5,
        labels=[10, 20, 30, 40, 50],
        two_dimensional=True,
    )

    svor = SVORImplicitQP(C=1.0, add_order_constraints=True)
    svor.fit(X, y)

    npsvor_min = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="min_distance")
    npsvor_min.fit(X, y)

    npsvor_ordered = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="ordered_sum")
    npsvor_ordered.fit(X, y)

    result = {
        "labels": sorted(set(y)),
        "svor": {
            "train_accuracy": float(calculate_accuracy(np.asarray(y), np.asarray(svor.predict(X)))),
            "thresholds": svor.thresholds_,
            "weights": svor.weights_,
        },
        "npsvor_min_distance": {
            "train_accuracy": float(calculate_accuracy(np.asarray(y), np.asarray(npsvor_min.predict(X)))),
            "biases": npsvor_min.biases_,
        },
        "npsvor_ordered_sum": {
            "train_accuracy": float(calculate_accuracy(np.asarray(y), np.asarray(npsvor_ordered.predict(X)))),
            "biases": npsvor_ordered.biases_,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
