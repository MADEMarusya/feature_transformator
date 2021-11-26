from pandas import DataFrame
from pandas import Series
from json import load
from torch import device as torch_device
from deeppavlov import configs, build_model

from src.constatns.constatns import EMPTY_VALUE


class NerDeeppavlovTransformer:

    def __init__(
            self,
            process_column: str,
            ner_deeppavlov_column: str,
            tokens_ner_deeppavlov_column: str,
            device: torch_device,
    ):
        self.process_column = process_column
        self.ner_deeppavlov_column = ner_deeppavlov_column
        self.tokens_ner_deeppavlov_column = tokens_ner_deeppavlov_column
        config_ner = load(open(configs.ner.ner_ontonotes_bert_mult_torch, "r"))
        self.ner_model = build_model(
            config_ner,
            download=True
        )
        self.ner_model = self.ner_model.to(device)
        self.device = device

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        all_tokens = []
        all_entity = []

        for _, txt in zip(X.index.values, X[self.process_column].values):
            result = self.ner_model([txt if len(txt) > 0 else EMPTY_VALUE])
            tokens = [" ".join(sent) for sent in result[0]]
            entity = [" ".join(sent) for sent in result[1]]

            all_tokens.extend(tokens)
            all_entity.extend(entity)

        X[self.ner_deeppavlov_column] = all_entity
        X[self.tokens_ner_deeppavlov_column] = all_tokens
        return X
