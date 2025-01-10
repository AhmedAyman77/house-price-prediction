import pandas as pd
from src.ingest_data import DataIngestFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    
    file_extension = '.zip'

    data_ingest = DataIngestFactory.get_data_ingest(file_extension=file_extension)

    df = data_ingest.ingest(file_path=file_path)

    return df