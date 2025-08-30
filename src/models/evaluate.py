from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred) -> dict:
    """
    Evaluate regression predictions with MAE and R².
    Returns a dict with both metrics.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def print_eval(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}")
