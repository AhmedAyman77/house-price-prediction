import pandas as pd
from src.feature_engineering import (
    FeatureEngineering,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
from zenml import step


@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = "log", features: list = None) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = FeatureEngineering(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineering(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineering(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineering(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df
