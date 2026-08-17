import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss


class WalkForwardBacktest:
    """
    Walk-forward backtesting: train model on past data,
    predict the next round, shift the window.
    """

    def __init__(self, model, scaler, initial_train_size: int = 500,
                 step_size: int = 50):
        self.model = model
        self.scaler = scaler
        self.initial_train_size = initial_train_size
        self.step_size = step_size

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run the backtest."""
        all_preds = []
        all_proba = []
        all_true = []

        for start in range(self.initial_train_size,
                           len(X) - self.step_size,
                           self.step_size):
            end = start + self.step_size

            X_train = X.iloc[:start]
            y_train = y.iloc[:start]
            X_test = X.iloc[start:end]
            y_test = y.iloc[start:end]

            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)

            self.model.fit(X_train_s, y_train)

            preds = self.model.predict(X_test_s)
            proba = self.model.predict_proba(X_test_s)

            all_preds.extend(preds)
            all_proba.extend(proba)
            all_true.extend(y_test.values)

        all_preds = np.array(all_preds)
        all_proba = np.array(all_proba)
        all_true = np.array(all_true)

        accuracy = accuracy_score(all_true, all_preds)
        logloss = log_loss(all_true, all_proba)

        print(f"Walk-Forward Backtest Results:")
        print(f"  Total predictions: {len(all_preds)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print()
        print(classification_report(all_true, all_preds,
                                    target_names=["Away", "Draw", "Home"]))

        return {
            "predictions": all_preds,
            "probabilities": all_proba,
            "actuals": all_true,
            "accuracy": accuracy,
            "log_loss": logloss,
        }
