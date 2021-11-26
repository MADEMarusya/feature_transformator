from pandas import DataFrame
from pandas import Series
from numerizer import numerize


class EnNumerizeTransform:

    def __init__(
        self,
        process_column: str,
        numerize_column: str,
    ):
        self.process_column = process_column
        self.numerize_column = numerize_column

    def fit(self, X: DataFrame, y: Series=None):
        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X[self.numerize_column] =''
        for i, row in X.iterrows():
            try:
                X.at[i, self.numerize_column] = numerize(row[self.process_column])
            except:
                X.at[i, self.numerize_column] = row[self.process_column]
        return X
