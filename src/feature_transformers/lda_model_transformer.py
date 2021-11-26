from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaMulticore
from typing import List

from src.feature_transformers.preprocessor_transformer import PreprocessorTransformer
from src.constatns.constatns import SMALL_FILL_VALUE


class LdaModelTransformer:

    def __init__(
            self,
            fit_data_for_topic_prediction: List[str],
            process_columns: str,
            topic_column: str,
            prefix_prob_columns: str,
            is_need_prob_columns: bool,
            num_topics: int
    ):
        self.fit_data_for_topic_prediction = fit_data_for_topic_prediction
        self.process_columns = process_columns
        self.topic_column = topic_column
        self.prefix_prob_columns = prefix_prob_columns
        self.num_topics = num_topics
        self.is_need_prob_columns = is_need_prob_columns
        self.vectorizer = CountVectorizer(
            min_df=10,
            max_df=0.2
        )
        preprocessed_column = "preprocessed_column"
        fit_column = "input"
        preprocessor = PreprocessorTransformer(
            process_column=fit_column,
            new_column=preprocessed_column
        )
        preprocessed = DataFrame(self.fit_data_for_topic_prediction, columns=[fit_column])
        preprocessed = preprocessor.fit(preprocessed).transform(preprocessed)
        preprocessed = preprocessed[preprocessed_column].to_list()
        corpus_dictionary = self.vectorizer.fit_transform(preprocessed)
        self.id_map = dict((v, k) for k, v in self.vectorizer.vocabulary_.items())

        corpus = Sparse2Corpus(
            corpus_dictionary,
            documents_columns=False
        )
        self.lda_model = LdaMulticore(
            corpus=corpus,
            id2word=self.id_map,
            passes=2,
            random_state=5,
            num_topics=self.num_topics,
            workers=2
        )

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X[self.topic_column] = self.topic_prediction(X[self.process_columns])
        if self.is_need_prob_columns:
            tmp_df = self.topic_distribution(
                num_topics=self.num_topics,
                arr_strings=X[self.process_columns].values,
                prefix=self.prefix_prob_columns
            ).fillna(SMALL_FILL_VALUE)
            X = concat([
                X.reset_index(drop=True),
                tmp_df.reset_index(drop=True)
            ], axis=1, copy=False).reset_index(drop=True)
        return X

    def topic_prediction(self, my_document: List[str]):
        string_input = list(my_document)
        X = self.vectorizer.transform(string_input)
        corpus = Sparse2Corpus(X, documents_columns=False)
        output = list(self.lda_model[corpus])
        topics = []
        for out in output:
            topics.append(sorted(out, key=lambda x: x[1], reverse=True)[0][0])
        return topics

    def topic_distribution(
            self,
            num_topics: int,
            arr_strings: List[str],
            prefix: str
    ) -> List[float]:
        arr_strings = list(arr_strings)
        X = self.vectorizer.transform(arr_strings)
        corpus = Sparse2Corpus(X, documents_columns=False)

        output = list(self.lda_model[corpus])
        probs = []
        for out in output:
            probs.append([pair[1] for pair in out])

        df = DataFrame(probs)
        df.columns = [f"{prefix}_{i}" for i in range(num_topics)]
        return df
