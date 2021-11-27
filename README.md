# Модуль генерации признаков
Модуль генерации признаков для проекта улучшения запросов для вопросно ответной системы Маруси

Для работы с модулем необходимо установить зависимости вручную (надеюсь это временное решение) 
и скачать все нужные модели
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit[all]
pip install spacy_udpipe
pip install numerizer
pip install transformers
pip install fasttext
# установка модели синтактического анализа от deeppavlov
python -m deeppavlov install syntax_ru_syntagrus_bert
# Установка модели английского нумеризатора
python -m spacy download en_core_web_sm
# скачивание модель языковой идентификации 
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# Скачивание модели для пунктуации капитализации (модуль gdown указан в requirements)
gdown --id 1-1Usk7sM1aydyZFEyTetaY7tydwoG1dC
```

После подготовки можно использовать пайплайн
```python
import sys
sys.path.append("./feature_transformator")

from feature_transformator.src.feature_pipeline import FeaturePipeline
import torch
import pandas as pd

# Путь до модели капитализации пунктуации
PATH_TO_CAPIT_PUNKT_MODEL = './model-epoch=00-val_loss=0.0680.ckpt.nemo'

# Путь до модели языковой идентификации
PATH_TO_LID_MODEL = './lid.176.bin'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Данные для обучения модели классификации по топикам (разделение идет по трем топикам)
# FeaturePipeline принимает на вход тип List[str]
with open("./raw_query_flow.txt", "r", encoding='utf-8') as fin:
    rawqq = fin.readlines()
data_for_fit = pd.DataFrame(rawqq, columns=["input"])

pipeline = FeaturePipeline(
    path_to_capit_punkt_model=PATH_TO_CAPIT_PUNKT_MODEL,
    path_to_lid_model=PATH_TO_LID_MODEL,
    fit_data_for_topic_prediction=data_for_fit["input"].values,
    device=device,
)

data = pd.DataFrame([{"input": "что ты знаешь о митингах двадцать первого тридцать первого января"},
                     {"input": "маруся найди мне пожалуйста невесту"}])

transformed_data = pipeline.fit_transform(data)
```
