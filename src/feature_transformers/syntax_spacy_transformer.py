from pandas import DataFrame
from pandas import Series
from spacy import Language

from src.constatns.constatns import (
    SEPARATOR,
    TIRE_SEPARATOR,
)


class SyntaxSpacyTransformer:

    def __init__(
            self,
            process_column: str,
            syntax_spacy_column: str,
            syntax_pos_spacy_column: str,
            ru_spacy_udpipe_model: Language,
    ):
        self.process_column = process_column
        self.syntax_spacy_column = syntax_spacy_column
        self.syntax_pos_spacy_column = syntax_pos_spacy_column
        self.ru_spacy_udpipe_model = ru_spacy_udpipe_model

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X["tmp"] = X[self.process_column].apply(
            lambda x: self.syntax(x)
        )
        X[[self.syntax_spacy_column, self.syntax_pos_spacy_column]] = X.tmp.str.split(
            TIRE_SEPARATOR,
            expand=True
        )
        X.drop('tmp', inplace=True, errors='ignore', axis=1)
        return X

    def syntax(self, x: str) -> str:
        doc = self.ru_spacy_udpipe_model(x)
        syntax = []
        pos = []
        for token in doc:
            syntax.append(token.dep_)
            pos.append(token.pos_)
        return TIRE_SEPARATOR.join([
            SEPARATOR.join(syntax),
            SEPARATOR.join(pos)
        ])
