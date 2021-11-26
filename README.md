# Модуль генерации признаков
Модуль генерации признаков для проекта улучшения запросов для вопросно ответной системы Маруси

Для работы с молулем необходимо установить его и скачать в папку модуля необходимые для работы готовые модели 
```bash
python setup.py install
pip install feature_transformer_made_marusya
# установка модели синтактического анализа от deeppavlov
python -m deeppavlov install syntax_ru_syntagrus_bert
# Установка модели английского нумеризатора
python -m spacy download en_core_web_sm
# скачивание модель языковой идентификации 
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# Скачивание модели для пунктуации капитализации (модуль gdown указан в requirements)
gdown --id 1-1Usk7sM1aydyZFEyTetaY7tydwoG1dC
```

После установки можно использовать модуль для трансформации признаков. Для этого нужно испортировать класс 