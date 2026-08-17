"""
Probability calibration for the ensemble.

The audit's calibration curve showed systematic overconfidence on
home favorites (predicted 55% -> actual 46%). Isotonic regression
fitted on a held-out calibration window corrects the probability
scale without touching the underlying model.

Temporal discipline: base model trains on the OLDEST segment, the
calibrator fits on the NEXT segment, so no calibration information
flows backward in time.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import StandardScaler

from .ensemble import build_ensemble


def build_calibrated_ensemble(X, y, train_frac: float = 0.75,
                              weights=(1, 1, 2)):
    """
    Train the voting ensemble on the first `train_frac` of the data
    (chronological), then fit an isotonic calibrator on the remainder.

    Returns (calibrated_model, scaler). Use exactly like the raw
    ensemble: calibrated_model.predict_proba(scaler.transform(...)).
    """
    split = int(len(X) * train_frac)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_cal, y_cal = X.iloc[split:], y.iloc[split:]

    scaler = StandardScaler()
    ensemble, _ = build_ensemble(X, y, weights=weights, verbose=False)
    ensemble.fit(scaler.fit_transform(X_train), y_train)

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(ensemble), method="isotonic",
    )
    calibrated.fit(scaler.transform(X_cal), y_cal)
    return calibrated, scaler
