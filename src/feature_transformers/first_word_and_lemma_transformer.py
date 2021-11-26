from pandas import DataFrame
from pandas import Series
from spacy import Language

from src.constatns.constatns import EMPTY_VALUE


class FirstWordAndLemmaTransformer:

    def __init__(
            self,
            process_column: str,
            first_world_column: str,
            first_world_lemma_column: str,
            ru_spacy_udpipe_model: Language,
    ):
        self.process_column = process_column
        self.first_world_column = first_world_column,
        self.first_world_lemma_column = first_world_lemma_column,
        self.ru_spacy_udpipe_model = ru_spacy_udpipe_model

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X[self.first_world_column] = X[self.process_column].apply(
            lambda x: x.split()[0].lower() if len(x.split()) > 0 else EMPTY_VALUE
        )
        X[self.first_world_lemma_column] = X[self.process_column].apply(
            lambda x: self.ru_spacy_udpipe_model(x)[0].lemma_.lower() if len(x.split()) > 0 else EMPTY_VALUE
        )
        return X
