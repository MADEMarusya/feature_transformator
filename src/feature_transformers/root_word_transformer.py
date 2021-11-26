from pandas import DataFrame
from pandas import Series
from spacy import Language

from src.constatns.constatns import (
    EMPTY_VALUE,
    SEPARATOR,
    TIRE_SEPARATOR,
    SENTENCE_ROOT,
)


class RootWordTransformer:

    def __init__(
            self,
            process_column: str,
            root_column: str,
            root_lemma_column: str,
            ru_spacy_udpipe_model: Language,
    ):
        self.process_column = process_column
        self.root_column = root_column
        self.root_lemma_column = root_lemma_column
        self.ru_spacy_udpipe_model = ru_spacy_udpipe_model

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X["tmp"] = X[self.process_column].apply(
            lambda x: self.get_root_sent(x)
        )
        X[[self.root_column, self.root_lemma_column]] = X.tmp.apply(str.lower).str.split(
            SEPARATOR,
            expand=True
        )
        X.drop('tmp', inplace=True, errors='ignore', axis=1)
        return X

    def get_root_sent(self, string: str) -> str:
        doc = self.ru_spacy_udpipe_model(string)
        root = EMPTY_VALUE
        lemma = EMPTY_VALUE
        for token in doc:
            if token.dep_ == SENTENCE_ROOT:
                root = TIRE_SEPARATOR.join(token.text.split())
                lemma = TIRE_SEPARATOR.join(token.lemma_.split())
        return SEPARATOR.join([root, lemma]).lower()
