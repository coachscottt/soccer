import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier


def train_and_evaluate(X, y):
    """
    Train multiple models with time series validation
    (no data leakage from the future).
    """
    # TimeSeriesSplit — correct validation for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5,
            # multi_class="multinomial" removed in sklearn 1.8 (now default)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.08, random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        fold_accuracies = []
        fold_log_losses = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)

            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)

            fold_accuracies.append(accuracy_score(y_test, preds))
            fold_log_losses.append(log_loss(y_test, proba))

        results[name] = {
            "accuracy_mean": np.mean(fold_accuracies),
            "accuracy_std": np.std(fold_accuracies),
            "log_loss_mean": np.mean(fold_log_losses),
            "log_loss_std": np.std(fold_log_losses),
        }

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  Accuracy:  {results[name]['accuracy_mean']:.4f} "
              f"± {results[name]['accuracy_std']:.4f}")
        print(f"  Log Loss:  {results[name]['log_loss_mean']:.4f} "
              f"± {results[name]['log_loss_std']:.4f}")

    return results, models
