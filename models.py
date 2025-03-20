from pydantic import BaseModel, Field, model_validator
from typing import List

# Central list of available model names
AVAILABLE_MODELS = [
    "Neural Network", "Gradient Boosting", "HistGradientBoosting",
    "MLPClassifier", "KNN", "XGBoost", "SVM",
    "Logistic Regression", "CatBoost", "SGD Classifier",
    "Extra Trees", "Random Forest", "AdaBoost", "Decision Tree",
    "Naive Bayes"
]

class Transaction(BaseModel):
    """Pydantic model for transaction data with validation rules."""
    ta: float = Field(default=0.0, ge=0.0, le=100000.0)
    tt: int = Field(default=4, ge=0, le=10)
    tm: float = Field(default=0.0, ge=0.0, le=24.0)
    du: int = Field(default=4, ge=0, le=10)
    lc: int = Field(default=8, ge=0, le=10)
    pm: int = Field(default=5, ge=0, le=10)
    ui: int = Field(..., ge=1, le=10000)
    pf: int = Field(..., ge=0, le=10)
    aa: int = Field(..., ge=0, le=200)
    nt: int = Field(..., ge=0, le=50)

class PredictionRequest(BaseModel):
    """Request model including transaction data and selected models."""
    transaction: Transaction
    models: List[str]

    @model_validator(mode='after')
    def validate_models(self):
        invalid_models = [model for model in self.models if model not in AVAILABLE_MODELS]
        if invalid_models:
            raise ValueError(f"Invalid model names: {invalid_models}. Available models: {AVAILABLE_MODELS}")
        return self
