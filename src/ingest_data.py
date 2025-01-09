import os
import zipfile
import pandas as pd
from abc import ABC, abstractmethod


class DataIngest(ABC):
    
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass


class ZipDataIngest(DataIngest):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith('.zip'):
            raise ValueError("The provided file is not a .zip file.")
        
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall("extracted_data")
        
        extracted_files = os.listdir("extracted_data")
        csv_files = [File for File in extracted_files if File.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        csv_file_path = os.path.join("extracted_data", csv_files[0])

        df = pd.read_csv(csv_file_path)

        return df
    

class DataIngestFactory:
    @staticmethod
    def get_data_ingest(file_extension: str) -> DataIngest:
        if file_extension == '.zip':
            return ZipDataIngest()
        else:
            raise ValueError("Unsupported file extension.")

if __name__ == "__main__":
    
    # Specify the file path
    file_path = "data/archive.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingest = DataIngestFactory.get_data_ingest(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingest.ingest(file_path)

    # Now df contains the DataFrame from the extracted CSV
    print(df.head())  # Display the first few rows of the DataFrame
    
    pass