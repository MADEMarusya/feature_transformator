FIRST_STOP_WORDS = [
    "давай",
    "да",
    "так",
    "привет",
    "ну",
    "нет",
    "не надо",
    "маруся",
    "Маруся",
    "Марусь",
    "марусь",
    "алиса",
    "Маруська",
    "маруська"
    "ты",
    "знаешь",
    "а",
    "а",
    "и",
    "еще",
    "ка",
    "ой",
    "siri",
    "мне",
    "скажи",
    "скажите",
    "очень",
    "нужно",
    "надо",
    "нужен",
    "расскажи",
    "подскажите",
    "подскажи",
    "спасибо",
    "пожалуйста",
    "хорошо",
    "короче",
    "говори",
    "говорите",
    "ok",
    "google",
    "но",
    "же",
    "бе",
    "ли",
    "блин",
    "или",
    "ладно",
    "раз"
]

REMOVE_WORDS = [
    "объясни пожалуйста",
    "спасибо большое",
    "марусь",
    "маруся",
    "\n"
]

REPLACE_WORDS = {
    "што": "что",
    "шо": "что"
}

STOP_WORDS = set(FIRST_STOP_WORDS)


EMPTY_VALUE = "EMPTY_VALUE"
SEPARATOR = " "
TIRE_SEPARATOR = "-"
SENTENCE_ROOT = "ROOT"
SMALL_FILL_VALUE = 1e-10
