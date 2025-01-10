from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BiVariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass

class NumericalVsNumericalAnalysis(BiVariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumericalAnalysis(BiVariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BiVariateAnalysisStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: BiVariateAnalysisStrategy):
        self._strategy = strategy
    
    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)



# Example usage
if __name__ == "__main__":
    # Example usage of the BivariateAnalyzer with different strategies.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Analyzing relationship between two numerical features
    # analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'Gr Liv Area', 'SalePrice')

    # Analyzing relationship between a categorical and a numerical feature
    # analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'Overall Qual', 'SalePrice')
    pass
