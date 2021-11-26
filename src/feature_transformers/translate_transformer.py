from pandas import DataFrame
from pandas import Series
from pandas import concat
from torch import device as torch_device
from transformers import FSMTForConditionalGeneration
from transformers import FSMTTokenizer


class TranslateTransformer:

    def __init__(
            self,
            process_column: str,
            translated_column: str,
            back_translated_column: str,
            device: torch_device
    ):
        self.process_column = process_column
        self.translated_column = translated_column
        self.back_translated_column = back_translated_column
        self.device = device

        transl_model = "facebook/wmt19-ru-en"
        self.translator_tokenizer = FSMTTokenizer.from_pretrained(transl_model)
        self.translator_model = FSMTForConditionalGeneration.from_pretrained(transl_model)
        self.translator_model = self.translator_model.to(self.device)

        back_transl_model = "facebook/wmt19-en-ru"
        self.back_translator_tokenizer = FSMTTokenizer.from_pretrained(back_transl_model)
        self.back_translator_model = FSMTForConditionalGeneration.from_pretrained(back_transl_model)
        self.back_translator_model = self.back_translator_model.to(self.device)

    def translating(self, phrase):
        input_ids = self.translator_tokenizer.encode(phrase, return_tensors="pt").to(self.device)
        outputs = self.translator_model.generate(input_ids)
        return self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def back_translating(self, phrase):
        input_ids = self.back_translator_tokenizer.encode(phrase, return_tensors="pt").to(self.device)
        outputs = self.back_translator_model.generate(input_ids)
        return self.back_translator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        result = DataFrame(index=X.index, columns=[self.translated_column,
                                                   self.back_translated_column
                                                   ])
        for i, row in X.iterrows():
            result.loc[i, self.translated_column] = self.translating(row[self.process_column])
            try:
                result.loc[i, self.back_translated_column] = self.back_translating(
                    result.loc[i, self.translated_column])
            except:
                result.loc[i, self.back_translated_column] = row[self.process_column]
        return concat([X, result], axis=1)
