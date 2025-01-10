from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass

    
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nData Types and Non-null Counts:")
        print(df.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy) -> None:
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataInspectionStrategy):
        self.strategy = strategy
    
    def execute_inspection(self, df: pd.DataFrame):
        self.strategy.inspect(df)

if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('./extracted_data/AmesHousing.csv')

    # Initialize the Data Inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and execute
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)
