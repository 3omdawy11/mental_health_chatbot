import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a translation engine.
Your only job is to translate the given text to English.
Rules:
- Output ONLY the translated text. No explanations, no notes, no preamble.
- If the text is already in English, return it exactly as-is.
- Preserve the original meaning, tone, and emotion faithfully.
- Do not add greetings or commentary of any kind."""


class Translator:
    def __init__(self, groq_api_key: str = None,
                 model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])
        self.model  = model

    def to_english(self, text: str, source_lang: str) -> str:
        if source_lang == "en":
            return text
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            translated = resp.choices[0].message.content.strip()
            logger.info(f"Translated [{source_lang}→en]: {translated[:80]}")
            return translated
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def to_lang(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Translate to language code '{target_lang}':\n\n{text}"},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Reverse translation failed: {e}")
            return text