from pandas import DataFrame
from pandas import Series
from torch import device as torch_device
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from src.constatns.constatns import SEPARATOR

class CapitalizationPunctuationTransformer:

    PUNKTS = "?!.,"

    def __init__(
            self,
            path_to_model: str,
            process_column: str,
            new_column: str,
            batch_size: int,
            device: torch_device,
    ):
        self.path_to_model = path_to_model
        self.process_column = process_column
        self.new_column = new_column
        self.batch_size = batch_size
        self.device = device
        self.eval_test = PunctuationCapitalizationModel.restore_from(
            self.path_to_model
        )
        self.eval_test = self.eval_test.to(self.device)

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")
        result = self.eval_test.add_punctuation_capitalization(
            X[self.process_column].values,
            batch_size=self.batch_size,
        )
        result_proc = []
        for text in result:
            if text[-1] in self.PUNKTS:
                text = text[:-1] + SEPARATOR + text[-1]
            result_proc.append(text)
        X[self.new_column] = result_proc
        return X
