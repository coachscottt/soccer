from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


def build_ensemble(X, y, weights=(1, 1, 2), verbose=True):
    """
    Build an ensemble model with soft voting.
    Ensembles usually outperform individual models.

    weights: per-estimator voting weights (lr, rf, xgb).
    The article default (1, 1, 2) favors XGBoost.
    """
    scaler = StandardScaler()

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=0.5)),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42)),
            ("xgb", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=42, eval_metric="mlogloss")),
        ],
        voting="soft",  # use probabilities, not votes
        weights=list(weights),
    )

    # Final training on all data (for production)
    # In practice, keep a holdout set
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ensemble.fit(X_train_scaled, y_train)

    preds = ensemble.predict(X_test_scaled)
    proba = ensemble.predict_proba(X_test_scaled)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE (Soft Voting, weights={list(weights)})")
        print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
        print(f"  Log Loss:  {log_loss(y_test, proba):.4f}")
        print()
        print(classification_report(
            y_test, preds,
            target_names=["Away Win", "Draw", "Home Win"]))

    return ensemble, scaler
