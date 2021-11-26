from pandas import DataFrame
from pandas import Series
from sklearn import pipeline
from spacy_udpipe import download as spacy_udpipe_download
from spacy_udpipe import load as spacy_udpipe_load
from typing import List
from torch import device as torch_device

from src.feature_transformers.preprocessor_transformer import PreprocessorTransformer
from src.feature_transformers.capitalization_punctuation_transformer import CapitalizationPunctuationTransformer
from src.feature_transformers.first_word_and_lemma_transformer import FirstWordAndLemmaTransformer
from src.feature_transformers.lda_model_transformer import LdaModelTransformer
from src.feature_transformers.root_word_transformer import RootWordTransformer
from src.feature_transformers.translate_transformer import TranslateTransformer
from src.feature_transformers.syntax_spacy_transformer import SyntaxSpacyTransformer
from src.feature_transformers.syntax_deeppavlov_transformer import SyntaxDeeppavlovTransformer
from src.feature_transformers.ner_deeppavlov_transformer import NerDeeppavlovTransformer
from src.feature_transformers.ner_dslim_transformer import NerDslimTransformer
from src.feature_transformers.language_identification_transformer import LanguageIdentificationTransformer
from src.feature_transformers.en_numerize_transformer import EnNumerizeTransform

from src.constatns.column_names import (
    ORIGINAL_SENTENCE_COLUMN,
    CLEAN_COLUMN,
    CAPIT_PUNKT_COLUMN,
    FIRST_WORLD_COLUMN,
    FIRST_WORLD_LEMMA_COLUMN,
    TOPIC_COLUMN,
    PREFIX_PROB_COLUMNS,
    ROOT_COLUMN,
    ROOT_LEMMA_COLUMN,
    TRANSLATED_SENTENCE_COLUMN,
    BACK_TRANSLATED_COLUMN,
    SYNTAX_SPACY_COLUMN,
    SYNTAX_POS_SPACY_COLUMN,
    SYNTAX_DEEPPAVLOV_COLUMN,
    NER_DEEPPAVLOV_COLUMN,
    TOKENS_NER_DEEPPAVLOV_COLUMN,
    NER_DSLIM_MISC_COLUMN,
    NER_DSLIM_PER_COLUMN,
    NER_DSLIM_LOC_COLUMN,
    NER_DSLIM_ORG_COLUMN,
    NER_DSLIM_LEN_MISC_COLUMN,
    NER_DSLIM_LEN_PER_COLUMN,
    NER_DSLIM_LEN_LOC_COLUMN,
    NER_DSLIM_LEN_ORG_COLUMN,
    NUMERIZE_COLUMN,
)


class FeaturePipeline:
    def __init__(
            self,
            path_to_capit_punkt_model: str,
            path_to_lid_model: str,
            fit_data_for_topic_prediction: List[str],
            device: torch_device,
            original_sentence_column: str = ORIGINAL_SENTENCE_COLUMN,
            clean_column: str = CLEAN_COLUMN,
            capit_punkt_column: str = CAPIT_PUNKT_COLUMN,
            first_world_column: str = FIRST_WORLD_COLUMN,
            first_world_lemma_column: str = FIRST_WORLD_LEMMA_COLUMN,
            translated_sentence_column: str = TRANSLATED_SENTENCE_COLUMN,
            topic_column: str = TOPIC_COLUMN,
            prefix_prob_columns: str = PREFIX_PROB_COLUMNS,
            root_column=ROOT_COLUMN,
            root_lemma_column=ROOT_LEMMA_COLUMN,
            translated_column=TRANSLATED_SENTENCE_COLUMN,
            back_translated_column=BACK_TRANSLATED_COLUMN,
            syntax_spacy_column=SYNTAX_SPACY_COLUMN,
            syntax_pos_spacy_column=SYNTAX_POS_SPACY_COLUMN,
            syntax_deeppavlov_column=SYNTAX_DEEPPAVLOV_COLUMN,
            ner_deeppavlov_column=NER_DEEPPAVLOV_COLUMN,
            tokens_ner_deeppavlov_column=TOKENS_NER_DEEPPAVLOV_COLUMN,
            misc_column=NER_DSLIM_MISC_COLUMN,
            per_column=NER_DSLIM_PER_COLUMN,
            loc_column=NER_DSLIM_LOC_COLUMN,
            org_column=NER_DSLIM_ORG_COLUMN,
            len_misc_column=NER_DSLIM_LEN_MISC_COLUMN,
            len_per_column=NER_DSLIM_LEN_PER_COLUMN,
            len_loc_column=NER_DSLIM_LEN_LOC_COLUMN,
            len_org_column=NER_DSLIM_LEN_ORG_COLUMN,
            numerize_column=NUMERIZE_COLUMN,
    ):
        num_lda_topics = 3
        spacy_udpipe_download("ru")
        self.ru_spacy_udpipe_model = spacy_udpipe_load("ru")

        self.pipeline = pipeline.Pipeline([
            ("preprocessor_transformer", PreprocessorTransformer(
                process_column=original_sentence_column,
                new_column=clean_column
            )),
            ("capitalization_punctuation_transformer", CapitalizationPunctuationTransformer(
                path_to_model=path_to_capit_punkt_model,
                process_column=clean_column,
                new_column=capit_punkt_column,
                batch_size=32,
                device=device
            )),
            ("first_word_and_lemma_transformer", FirstWordAndLemmaTransformer(
                process_column=clean_column,
                first_world_column=first_world_column,
                first_world_lemma_column=first_world_lemma_column,
                ru_spacy_udpipe_model=self.ru_spacy_udpipe_model
            )),
            ("lda_transformer_3", LdaModelTransformer(
                fit_data_for_topic_prediction=fit_data_for_topic_prediction,
                process_columns=original_sentence_column,
                topic_column=topic_column + str(num_lda_topics),
                prefix_prob_columns=prefix_prob_columns,
                is_need_prob_columns=True,
                num_topics=num_lda_topics
            )),
            ('root_word_transformer', RootWordTransformer(
                process_column=clean_column,
                root_column=root_column,
                root_lemma_column=root_lemma_column,
                ru_spacy_udpipe_model=self.ru_spacy_udpipe_model
            )),
            ("translate_transformer", TranslateTransformer(
                process_column=capit_punkt_column,
                translated_column=translated_column,
                back_translated_column=back_translated_column,
                device=device
            )),
            ("syntax_spacy_transformer", SyntaxSpacyTransformer(
                process_column=clean_column,
                syntax_spacy_column=syntax_spacy_column,
                syntax_pos_spacy_column=syntax_pos_spacy_column,
                ru_spacy_udpipe_model=self.ru_spacy_udpipe_model
            )),
            ("syntax_deeppavlov_transformer", SyntaxDeeppavlovTransformer(
                process_column=clean_column,
                syntax_deeppavlov_column=syntax_deeppavlov_column,
            )),
            ("ner_deeppavlov_transformer", NerDeeppavlovTransformer(
                    process_column=capit_punkt_column,
                    ner_deeppavlov_column=ner_deeppavlov_column,
                    tokens_ner_deeppavlov_column=tokens_ner_deeppavlov_column,
                    device=device
            )),
            ("ner_dslim_transformer", NerDslimTransformer(
                process_column=translated_sentence_column,
                misc_column=misc_column,
                per_column=per_column,
                loc_column=loc_column,
                org_column=org_column,
                len_misc_column=len_misc_column,
                len_per_column=len_per_column,
                len_loc_column=len_loc_column,
                len_org_column=len_org_column,
                device=device
            )),
            ("en_numerize_transformer", EnNumerizeTransform(
                process_column=translated_sentence_column,
                numerize_column=numerize_column
            )),
            ("language_identification", LanguageIdentificationTransformer(
                process_column=capit_punkt_column,
                path_to_lid_model=path_to_lid_model
            ))
        ])

    def fit(self, X: DataFrame, y: Series = None):
        return self.pipeline.fit(X, y)

    def transform(self, X: DataFrame):
        return self.pipeline.transform(X)

    def fit_transform(self, X: DataFrame):
        return self.pipeline.fit_transform(X)
