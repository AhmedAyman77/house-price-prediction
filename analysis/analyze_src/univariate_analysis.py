from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
    
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)




if __name__ == "__main__":
    # Example usage of the UnivariateAnalyzer with different strategies.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Analyzing a numerical feature
    # analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'SalePrice')

    # Analyzing a categorical feature
    # analyzer.set_strategy(CategoricalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Neighborhood')
    pass
