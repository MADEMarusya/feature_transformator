from pandas import DataFrame
from pandas import Series
from torch import device as torch_device
from torch import cuda
from re import sub
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import pipeline as transformers_pipeline


class NerDslimTransformer:

    def __init__(
            self,
            process_column: str,
            misc_column: str,
            per_column: str,
            loc_column: str,
            org_column: str,
            len_misc_column: str,
            len_per_column: str,
            len_loc_column: str,
            len_org_column: str,
            device: torch_device
    ):
        self.process_column = process_column
        self.misc_column = misc_column
        self.per_column = per_column
        self.loc_column = loc_column
        self.org_column = org_column
        self.len_misc_column = len_misc_column
        self.len_per_column = len_per_column
        self.len_loc_column = len_loc_column
        self.len_org_column = len_org_column
        self.device = -1 if device.type == 'cpu' else cuda.current_device()

        ner_model = "dslim/bert-large-NER"
        tokenizer = AutoTokenizer.from_pretrained(ner_model)
        model = AutoModelForTokenClassification.from_pretrained(ner_model)
        self.nlp = transformers_pipeline("ner", model=model, tokenizer=tokenizer, device=self.device)

    def fit(self, X: DataFrame, y: Series = None):
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        if not isinstance(X, DataFrame):
            raise RuntimeError("X must be pandas DataFrame!!!")

        X.loc[:, self.misc_column] = ""
        X.loc[:, self.per_column] = ""
        X.loc[:, self.loc_column] = ""
        X.loc[:, self.org_column] = ""
        X.loc[:, self.len_misc_column] = 0
        X.loc[:, self.len_per_column] = 0
        X.loc[:, self.len_loc_column] = 0
        X.loc[:, self.len_org_column] = 0
        for j, row in X.iterrows():
            try:
                extraction_result = self.nlp(row['translated_sentence'])
                length_ner = len(extraction_result)
                location = []
                person = []
                misc = []
                organization = []
                for i in range(length_ner):
                    string = []
                    if extraction_result[i]['entity'] == 'B-LOC':
                        string.append(extraction_result[i]['word'])
                        next_ner = i + 1
                        if next_ner >= length_ner:
                            string = ' '.join(string)
                            location.append(sub(r' ##', '', string))
                            break
                        while extraction_result[next_ner]['entity'] == 'I-LOC':
                            string.append(extraction_result[next_ner]['word'])
                            next_ner += 1
                            if next_ner >= length_ner:
                                break
                        string = ' '.join(string)
                        location.append(sub(r' ##', '', string))
                    elif extraction_result[i]['entity'] == 'B-PER':
                        string.append(extraction_result[i]['word'])
                        next_ner = i + 1
                        if next_ner >= length_ner:
                            string = ' '.join(string)
                            person.append(sub(r' ##', '', string))
                            break
                        while extraction_result[next_ner]['entity'] == 'I-PER':
                            string.append(extraction_result[next_ner]['word'])
                            next_ner += 1
                            if next_ner >= length_ner:
                                break
                        string = ' '.join(string)
                        person.append(sub(r' ##', '', string))
                    elif extraction_result[i]['entity'] == 'B-MISC':
                        string.append(extraction_result[i]['word'])
                        next_ner = i + 1
                        if next_ner >= length_ner:
                            string = ' '.join(string)
                            misc.append(sub(r' ##', '', string))
                            break
                        while extraction_result[next_ner]['entity'] == 'I-MISC':
                            string.append(extraction_result[next_ner]['word'])
                            next_ner += 1
                            if next_ner >= length_ner:
                                break
                        string = ' '.join(string)
                        misc.append(sub(r' ##', '', string))
                    elif extraction_result[i]['entity'] == 'B-ORG':
                        string.append(extraction_result[i]['word'])
                        next_ner = i + 1
                        if next_ner >= length_ner:
                            string = ' '.join(string)
                            organization.append(sub(r' ##', '', string))
                            break
                        while extraction_result[next_ner]['entity'] == 'I-ORG':
                            string.append(extraction_result[next_ner]['word'])
                            next_ner += 1
                            if next_ner >= length_ner:
                                break
                        string = ' '.join(string)
                        organization.append(sub(r' ##', '', string))
                if len(location) != 0:
                    X.at[j, self.len_loc_column] = len(location)
                    X.at[j, self.loc_column] = ', '.join(location)
                if len(person) != 0:
                    X.at[j, self.len_per_column] = len(person)
                    X.at[j, self.per_column] = ', '.join(person)
                if len(misc) != 0:
                    X.at[j, self.len_misc_column] = len(misc)
                    X.at[j, self.misc_column] = ', '.join(misc)
                if len(organization) != 0:
                    X.at[j, self.len_org_column] = len(organization)
                    X.at[j, self.org_column] = ', '.join(organization)
            except:
                continue
        return X
