from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MultiVariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass
    
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultiVariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
    
    def generate_pairplot(self, df: pd.DataFrame):
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()



# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMultivariateAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Multivariate Analysis
    # multivariate_analyzer = SimpleMultivariateAnalysis()

    # Select important features for pair plot
    # selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    # multivariate_analyzer.analyze(selected_features)
    pass
