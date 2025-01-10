from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step

from zenml import Model, pipline, step

@pipline(
    model=Model(
        # The name uniquely identifies this model
        name = "Prices_Predictor"
    ),
)

def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""
    row_data = data_ingestion_step(
        file_path="../data/archive.zip"
    )

    filled_data = handle_missing_values_step(row_data)