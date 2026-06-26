"""
Preprocess and translate YouTube video titles.

This script was developed for Google Colab.

Input:
    A pandas DataFrame named `df` containing a column called `video_title`.

Output:
    A CSV file with cleaned titles, detected language, corrected language code,
    and English translation.

Generated columns:
    title_clean
    lang_detected
    lang_used
    title_english
"""

import re

import emoji
import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

OUTPUT_PATH = "/content/GenZ_2025-09-08_to_2025-09-13_TRANSLATED.csv"


# ---------------------------------------------------------------------
# Text cleaning and translation functions
# ---------------------------------------------------------------------

def clean_text(title):
    """Remove emojis, hashtags, and extra spaces from a video title."""
    text = emoji.replace_emoji(str(title), replace="")
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_lang_safe(text):
    """Detect language safely. Return 'unknown' if detection fails."""
    try:
        if not str(text).strip():
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


def choose_lang_for_translation(text, detected_lang):
    """
    Correct likely misclassification of romanized Hindi/Nepali text.

    Romanized South Asian titles may sometimes be detected as Indonesian,
    Malay, Tagalog, Swahili, or unknown. This heuristic checks for common
    romanized Hindi/Nepali words and assigns the language as Hindi (`hi`)
    before translation.
    """
    romanized_hi_words = [
        " andolan", " awaaz", " ka ", " ke ", " ki ",
        " sarkar", " neta", " pradesh", " desh",
        " kyun", " kya"
    ]

    text_lower = " " + str(text).lower() + " "

    if detected_lang in ["id", "ms", "tl", "sw", "unknown"]:
        if any(word in text_lower for word in romanized_hi_words):
            return "hi"

    return detected_lang


def translate_to_english(text, lang_used):
    """Translate cleaned title into English."""
    if not text:
        return None

    if lang_used == "en":
        return text

    if lang_used == "unknown":
        return None

    try:
        return GoogleTranslator(source=lang_used, target="en").translate(text)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Apply preprocessing
# ---------------------------------------------------------------------

if "video_title" not in df.columns:
    raise ValueError("The DataFrame must contain a column named 'video_title'.")

df["title_clean"] = df["video_title"].apply(clean_text)

df["lang_detected"] = df["title_clean"].apply(detect_lang_safe)

df["lang_used"] = df.apply(
    lambda row: choose_lang_for_translation(
        row["title_clean"],
        row["lang_detected"]
    ),
    axis=1
)

df["title_english"] = df.apply(
    lambda row: translate_to_english(
        row["title_clean"],
        row["lang_used"]
    ),
    axis=1
)


# ---------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)

print(
    df[
        [
            "video_title",
            "title_clean",
            "lang_detected",
            "lang_used",
            "title_english"
        ]
    ].head(5)
)
