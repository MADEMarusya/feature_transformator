from pandas import DataFrame
from pandas import Series
from src.constatns.constatns import REMOVE_WORDS
from src.constatns.constatns import REPLACE_WORDS
from src.constatns.constatns import SEPARATOR
from src.constatns.constatns import STOP_WORDS


class PreprocessorTransformer:

    def __init__(
            self,
            process_column: str,
            new_column: str,
    ):
        self.process_column = process_column
        self.new_column = new_column

    def preprocessing(self, string: str) -> str:
        for word in REMOVE_WORDS:
            string = string.replace(word, " ")
        for k, v in REPLACE_WORDS.items():
            string = string.replace(k, v)
        words = string.split(SEPARATOR)
        words = [
                    word
                    for word in words[:2]
                    if word not in STOP_WORDS
                ] + words[2:]
        return SEPARATOR.join(words)

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X[self.new_column] = X[self.process_column].apply(lambda x: self.preprocessing(x))
        return X