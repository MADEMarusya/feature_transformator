from pandas import DataFrame
from pandas import Series
from pandas import concat
from fasttext import load_model


class LanguageIdentificationTransformer:

    def __init__(
            self,
            process_column: str,
            path_to_lid_model: str,
    ):
        self.process_column = process_column
        self.lid_model = load_model(path_to_lid_model)
        self.all_languages = [s[9:] for s in self.lid_model.labels]

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        lang_probs = DataFrame(0.0, index=X.index, columns=self.all_languages)
        sentences = X[self.process_column].to_list()
        predictions = self.lid_model.predict(sentences, k=20)
        for index, probs in zip(X.index, zip(*predictions)):
            for lang, prob in zip(*probs):
                lang_probs.loc[index, lang[9:]] = prob
        return concat([X, lang_probs], axis=1)
