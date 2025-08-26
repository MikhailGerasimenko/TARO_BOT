# Telegram-бот для гадания на Таро (LLM + ML-фильтрация)

## Возможности
- Расклад и подробная трактовка через Groq LLM (`llama-3.3-70b-versatile`)
- Фильтрация неподходящих вопросов русскоязычным BERT-классификатором + правила
- Диалог в Telegram с кнопками: добор карт, финальный анализ, перезапуск

---

## Требования
- Python 3.9+
- Аккаунт и API Key Groq
- Токен Telegram-бота (от @BotFather)

---

## Установка

1) Клонируйте проект и создайте окружение
```bash
python3 -m venv tarot-venv
source tarot-venv/bin/activate  # Mac/Linux
# .\\tarot-venv\\Scripts\\activate  # Windows
```

2) Установите зависимости
```bash
pip install -r requirements.txt
```

3) Переменные окружения
Создайте файл `.env` в корне:
```
TELEGRAM_BOT_TOKEN=ваш_токен_бота
GROQ_API_KEY=ваш_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
# Необязательно: порог уверенности классификатора
CLASSIFIER_THRESHOLD=0.65
```
Загрузите .env (локально):
```bash
pip install python-dotenv
export $(grep -v '^#' .env | xargs)
```

---

## Обучение/подготовка классификатора (опционально)
Если вы хотите включить ML-фильтрацию:
- Подготовьте датасет `deepseek_json_20250707_b9011f.json` (список объектов `{ "text": ..., "label": 0/1 }`).
- Запустите обучение:
```bash
python train_classifier.py
```
- Артефакты сохранятся в `./russian-bert-tarot-classifier/`.
- Без модели фильтрация будет работать по правилам (ключевые слова), а ML — пропущен.

---

## Запуск бота
```bash
python -m bot.main
```
Бот стартует и начинает получать апдейты через polling.

---

## Структура
- `bot/main.py` — логика Telegram, Groq API, фильтрация (правила + BERT)
- `bot/tarot_data.py` — справочник карт и сочетаний
- `train_classifier.py` — обучение русскоязычного BERT-классификатора
- `russian-bert-tarot-classifier/` — сохранённые веса/токенизатор

---

## Заметки по безопасности
- Никогда не коммитьте `.env` и ключи в репозиторий
- Переменные окружения обязательны для запуска (GROQ_API_KEY, TELEGRAM_BOT_TOKEN)

---

## Типичные проблемы
- 404 от Groq: проверьте endpoint `https://api.groq.com/openai/v1/chat/completions` и имя модели
- Ошибки при загрузке BERT: проверьте наличие папки `russian-bert-tarot-classifier`
- Задержки: модель и API-вызовы выполняются последовательно; проверьте сеть и лимиты Groq

---

## Лицензия
MIT (по желанию, обновите при необходимости)