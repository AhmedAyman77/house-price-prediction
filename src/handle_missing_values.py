import logging
from abc import ABC, abstractmethod
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValueStrategy(MissingValueHandlingStrategy):

    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")

        return df_cleaned

class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None) -> None:
        self.method = method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame):
        
        logging.info(f"Filling missing values using method: {self.method}")
        df_cleaned = df.copy()
        
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
        
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
        
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df_cleaned[column].mode().iloc[0], inplace=True)
        
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")
        
        logging.info("Missing values filled.")
        return df_cleaned

class MissingValueFillMissingValueStrategyHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy) -> None:
        self.strategy = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)

    pass
