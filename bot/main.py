import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, ConversationHandler
import random
from bot.tarot_data import TAROT_CARDS, COMBINATIONS
import requests
import asyncio
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from torch.nn.functional import softmax
import time
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Состояния для ConversationHandler
FIRST_DRAW, SECOND_DRAW, FINAL = range(3)

user_sessions = {}

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv('GROQ_MODEL', "llama-3.3-70b-versatile")  # Актуальная поддерживаемая модель Groq

# === Глобальная загрузка модели и токенизатора ===
MODEL_PATH = './russian-bert-tarot-classifier'
DEVICE = (
    'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
)

tokenizer = None
model = None
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    logging.warning(f"Не удалось загрузить классификатор из {MODEL_PATH}: {e}. Фильтрация будет ограниченной.")

# Простые правила до ML (явные запреты)
FORBIDDEN_KEYWORDS = [
    'когда', 'точная дата', 'дата', 'диагноз', 'беремен', 'умр', 'выиграю', 'лотер', 'сколько денег', 'сумма',
    'сделай за меня', 'реши за', 'порчу', 'сглаз'
]
CONFIDENCE_THRESHOLD = float(os.getenv('CLASSIFIER_THRESHOLD', '0.65'))

def generate_groq_answer(prompt, model=GROQ_MODEL, max_tokens=None, temperature=0.8):
    if not GROQ_API_KEY:
        return "Ошибка: отсутствует GROQ_API_KEY в переменных окружения."
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты — опытный таролог, отвечай на русском."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    backoff = 1.0
    for attempt in range(3):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=data, timeout=30)
            if response.status_code >= 400:
                logging.error(f"Groq API error {response.status_code}: {response.text}")
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                logging.exception("Ошибка при обращении к Groq API, попытки исчерпаны")
                return f"Ошибка при обращении к Groq API: {e}"
            time.sleep(backoff)
            backoff *= 2
    return "Ошибка: не удалось получить ответ от Groq API."

def is_question_suitable(question):
    text = (question or '').lower()
    if any(k in text for k in FORBIDDEN_KEYWORDS):
        return False
    if tokenizer is None or model is None:
        # fallback: если нет модели, пропускаем вопрос (минимум блок по ключевикам выше)
        return True
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze(0)
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())
    if predicted_class == 1 and confidence >= CONFIDENCE_THRESHOLD:
        return True
    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
Этот бот помогает получить совет от карт Таро, но помни:

❌ Не подходят – вопросы о точных датах, медицинских диагнозах, финансах (например, «Когда я разбогатею?») или решения за других людей.

✅ Лучше спрашивать – о личных переживаниях, отношениях, выборе пути («Что мне учесть в текущей ситуации?»), развитии себя.

📜 Как работает:

Выбираешь расклад (1 карта для ясности, 3 карты для анализа, «Кельтский крест» для глубины).
Формулируешь вопрос внутрь себя – бот не требует его писать, но фокус важен.
Получаешь карты с толкованием: они отражают тенденции, а не приговор.

⚠️ Важно:

Карты показывают энергию момента – интерпретируй их через призму своей жизни.
Не злоупотребляй частыми гаданиями на одну тему – ответ уже дан.
✨ Таро – зеркало твоего подсознания. Доверься мудрости символов!

(Примеры хороших вопросов: «Как мне понять свои чувства к Х?», «На что обратить внимание в работе?», «Какой урок я сейчас прохожу?»).

Итак, какой у вас вопрос?
""")
    return FIRST_DRAW

async def first_draw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    if not is_question_suitable(question):
        await update.message.reply_text('Извините, я могу гадать только на вопросы, связанные с личной жизнью, отношениями, работой, здоровьем, духовным развитием и будущим. Пожалуйста, переформулируйте ваш вопрос.')
        return FIRST_DRAW
    cards = random.sample(TAROT_CARDS, 3)
    user_sessions[update.effective_user.id] = {
        'question': question,
        'cards': cards,
        'step': 1
    }
    response = f'Ваш вопрос: {question}\n\nПервые 3 карты:'
    for idx, card in enumerate(cards, 1):
        response += f"\n{idx}. {card['name']}: {card['meaning']}\n{card['description']}\n"
    response += '\nЕсли хотите добрать еще карты, нажмите кнопку "Ещё". Если хотите завершить гадание — нажмите "Завершить".'
    keyboard = [['Ещё', 'Завершить']]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text(response, reply_markup=reply_markup)
    return SECOND_DRAW

async def second_draw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)
    if not session:
        await update.message.reply_text('Сначала задайте вопрос!')
        return FIRST_DRAW
    user_text = update.message.text.lower()
    if user_text in ['еще', 'ещё']:
        # добираем еще 2 карты
        already_drawn = session['cards']
        remaining = [c for c in TAROT_CARDS if c not in already_drawn]
        extra_cards = random.sample(remaining, 2)
        session['cards'].extend(extra_cards)
        session['step'] = 2
        response = 'Добраны еще 2 карты:'
        for idx, card in enumerate(extra_cards, len(already_drawn)+1):
            response += f"\n{idx}. {card['name']}: {card['meaning']}\n{card['description']}\n"
        response += '\nЕсли хотите завершить гадание — нажмите "Завершить".'
        keyboard = [['Завершить']]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text(response, reply_markup=reply_markup)
        return FINAL
    elif user_text == 'завершить':
        return await final_analysis(update, context)
    else:
        keyboard = [['Ещё', 'Завершить']]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text('Пожалуйста, используйте кнопки ниже: "Ещё" чтобы добрать карты или "Завершить" чтобы получить итоговый анализ.', reply_markup=reply_markup)
        return SECOND_DRAW

def analyze_combinations(cards):
    names = [card['name'] for card in cards]
    found = []
    auto = []
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            pair = tuple(sorted([names[i], names[j]]))
            if pair in COMBINATIONS:
                found.append(f'Сочетание {pair[0]} + {pair[1]}: {COMBINATIONS[pair]}')
            else:
                c1 = cards[i]
                c2 = cards[j]
                auto.append(f'Сочетание {c1["name"]} + {c2["name"]}: {c1["meaning"]} и {c2["meaning"]}. Вместе это может указывать на ситуацию, где {c1["meaning"].lower()} сочетается с {c2["meaning"].lower()}.')
    return '\n'.join(found + auto) if (found or auto) else None
 
async def final_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)
    if not session:
        await update.message.reply_text('Сначала задайте вопрос!')
        return FIRST_DRAW
    question = session['question']
    cards = session['cards']
    prompt = (
        'Сделай трактовку карт Таро для вопроса: "' + question + '".\n'
        'Карты:\n'
    )
    for idx, card in enumerate(cards, 1):
        prompt += f"{idx}. {card['name']}: {card['meaning']}\n"
    prompt += (
        "Дай максимально подробную, глубокую, развернутую и насыщенную трактовку именно этих карт применительно к вопросу пользователя. "
        "В ответе обязательно упоминай названия карт без какого-либо выделения. "
        "Используй тематические смайлики (например, 🔮, 🃏, ❤️, ✨, 🤔 и другие) для украшения и эмоционального окраса. "
        "Раскрой значения каждой карты, их взаимосвязи, возможные сценарии развития ситуации, психологические аспекты, а также дай практические советы и примеры, основанные на выпавших картах. "
        "Пиши грамотным, литературным и выразительным языком, избегай тавтологии, повторов и шаблонных фраз. "
        "Используй богатый и разнообразный словарный запас. "
        "Не используй жирный шрифт и не добавляй ** **. "
        "Не добавляй в конце фразу 'В целом' или 'В общем', если не требуется итоговое обобщение."
    )
    await update.message.reply_text('Вывод по выпавшим картам, это может занять несколько секунд:', reply_markup=ReplyKeyboardRemove())
    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, generate_groq_answer, prompt)
    except Exception as e:
        import logging
        logging.exception("Ошибка при генерации ответа через Groq API")
        answer = f"Ошибка при генерации ответа: {e}"
    await update.message.reply_text(answer, reply_markup=ReplyKeyboardRemove())
    # Добавим кнопку 'Начать заново' после завершения гадания
    keyboard = [['Начать заново']]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text('Хотите попробовать ещё раз? Нажмите "Начать заново".', reply_markup=reply_markup)
    user_sessions.pop(user_id, None)
    return ConversationHandler.END

async def handle_restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Обработчик кнопки "Начать заново" сработал. Пожалуйста, задайте новый вопрос для гадания на Таро.', reply_markup=ReplyKeyboardRemove())
    return FIRST_DRAW

conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        FIRST_DRAW: [MessageHandler(filters.Regex('(?i)^начать\s*заново$'), handle_restart),
                     MessageHandler(filters.TEXT & ~filters.COMMAND, first_draw)],
        SECOND_DRAW: [MessageHandler(filters.Regex('(?i)^начать\s*заново$'), handle_restart),
                      MessageHandler(filters.TEXT & ~filters.COMMAND, second_draw)],
        FINAL: [MessageHandler(filters.Regex('(?i)^начать\s*заново$'), handle_restart),
                MessageHandler(filters.TEXT & ~filters.COMMAND, final_analysis)],
    },
    fallbacks=[CommandHandler('start', start)]
)

if __name__ == '__main__':
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    if not TOKEN:
        raise RuntimeError('Отсутствует TELEGRAM_BOT_TOKEN в переменных окружения')
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(conv_handler)
    print('Бот запущен...')
    app.run_polling() 