"""Cheap interaction-language selection, independent of training-source language."""

import re
from collections.abc import Sequence

from ..models import MessageRole


_EXPLICIT = (
    (re.compile(r"(?:한국어|한글)(?:로|으로)?\s*(?:대답|답변|말|해|해주세요)|(?:answer|respond|reply|speak)\s+in\s+korean", re.I), "Korean"),
    (re.compile(r"(?:영어)(?:로|으로)?\s*(?:대답|답변|말|해|해주세요)|(?:answer|respond|reply|speak)\s+in\s+english", re.I), "English"),
    (re.compile(r"(?:프랑스어)(?:로|으로)?\s*(?:대답|답변|말|해|해주세요)|(?:answer|respond|reply|speak)\s+in\s+french|r[ée]ponds?\s+en\s+fran[çc]ais", re.I), "French"),
)
_FRENCH = re.compile(r"\b(?:qui|que|quoi|pourquoi|comment|bonjour|merci|est|suis|vous|avec|dans|pour|elle|il|je|tu|nous)\b", re.I)
_ENGLISH = re.compile(r"\b(?:who|what|where|when|why|how|is|are|was|were|do|did|have|has|the|you|your|me|my|tell|about)\b", re.I)
_OTHER_LATIN = re.compile(r"\b(?:hola|cómo|quién|buenos|wer|wie|ist|ciao|perché|olá|quem)\b", re.I)


def _clear_language(message: str) -> str | None:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", message)
    if re.search(r"[가-힣]", message):
        return "Korean"
    if len(words) < 2:
        return None
    french = len(_FRENCH.findall(message))
    english = len(_ENGLISH.findall(message))
    if french > english:
        return "French"
    if english > french:
        return "English"
    # Let the model infer other clearly expressed languages from the actual
    # message; never force the language of stored profile text onto it.
    return "language of the current user message" if len(words) >= 4 or _OTHER_LATIN.search(message) else None


def response_language(message: str, recent_messages: Sequence[tuple[MessageRole, str]] = (),
                      locale: str = "en") -> str:
    for pattern, language in _EXPLICIT:
        if pattern.search(message):
            return language
    current = _clear_language(message)
    if current:
        return current
    for role, content in reversed(recent_messages):
        if role == MessageRole.USER:
            prior = _clear_language(content)
            if prior:
                return prior
    return "Korean" if locale == "ko" else "English"
