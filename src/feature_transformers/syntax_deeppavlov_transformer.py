from pandas import DataFrame
from pandas import Series
from pandas import concat
from deeppavlov import configs, build_model


class SyntaxDeeppavlovTransformer:

    def __init__(
        self,
        process_column: str,
        syntax_deeppavlov_column: str
    ):
        self.process_column = process_column
        self.syntax_deeppavlov_column = syntax_deeppavlov_column
        self.syntax_model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

    def fit(self, X: DataFrame, y: Series=None):
        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        result = DataFrame('', index=X.index, columns=[self.syntax_deeppavlov_column])

        parsing_sentences = self.syntax_model(X[self.process_column].values)
        for i in range(len(parsing_sentences)):
            parse = parsing_sentences[i].rstrip().split('\t_\t_\n')
            parsed = [p.split('\t_\t_\t_\t_\t')[-1].split('\t')[1] for p in parse ]
            result.loc[X.index[i], self.syntax_deeppavlov_column] = ' '.join(parsed)

        return concat([X, result], axis=1)
