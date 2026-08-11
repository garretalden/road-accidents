"""Original Penn CIS 545 class-project models: Logistic Regression, Random
Forest, XGBoost. Kept as a fixed historical reference point — new models go
in experiments/ instead.
"""

from .models import train_lr, train_rf, train_xgb

MODELS = {
    "Logistic Regression": ("lr", train_lr),
    "Random Forest": ("rf", train_rf),
    "XGBoost": ("xgb", train_xgb),
}
