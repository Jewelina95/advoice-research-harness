from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import jieba


TIME_MARK = re.compile(r"\x15\d+_\d+\x15")
CHAT_CODE = re.compile(r"\[[^\]]*\]|&[=+]?\w+|\([^)]*\)|\x15[^\x15]*\x15")
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
CJK = re.compile(r"[\u3400-\u9fff]")

FILLERS = {
    "en": {"uh", "um", "erm", "hmm", "well"},
    "es": {"eh", "em", "este", "pues", "bueno"},
    "zh": {"嗯", "呃", "这个", "那个"},
}
PRONOUNS = {
    "en": {"i", "you", "he", "she", "it", "we", "they", "this", "that", "someone", "something"},
    "es": {"yo", "tu", "tú", "él", "ella", "ello", "nosotros", "nosotras", "ellos", "ellas", "esto", "eso", "alguien", "algo"},
    "zh": {"我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "她们", "这个", "那个", "有人", "东西"},
}
STOPWORDS = {
    "en": {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "is", "are", "was", "were", "be", "been", "with", "for"},
    "es": {"el", "la", "los", "las", "un", "una", "y", "o", "de", "a", "en", "es", "son", "con", "por", "para"},
    "zh": {"的", "了", "和", "是", "在", "有", "也", "就", "都", "而", "及", "与", "一个", "这", "那"},
}

LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "zh": "zh",
    "zho": "zh",
    "cmn": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
}

PICTURE_CONTENT_UNITS = {
    "en": {
        "boy": ("boy", "son"),
        "girl": ("girl", "daughter", "sister"),
        "mother": ("mother", "mom", "woman", "lady"),
        "cookie": ("cookie", "cookies"),
        "jar": ("jar",),
        "stool": ("stool", "chair"),
        "falling": ("fall", "falling", "tipping", "tilting"),
        "reaching": ("reach", "reaching", "taking", "grabbing"),
        "sink": ("sink",),
        "overflow": ("overflow", "overflowing", "spilling", "running over"),
        "dishes": ("dish", "dishes", "plate", "plates"),
        "washing": ("washing", "drying", "wiping"),
        "window": ("window", "windows"),
        "curtains": ("curtain", "curtains"),
        "outside": ("outside", "yard", "garden", "grass", "tree", "bushes"),
        "water": ("water",),
    },
    "es": {
        "boy": ("niño", "chico", "hijo"),
        "girl": ("niña", "chica", "hija", "hermana"),
        "mother": ("madre", "mamá", "mujer", "señora"),
        "cookie": ("galleta", "galletas"),
        "jar": ("tarro", "frasco"),
        "stool": ("taburete", "banquito", "silla"),
        "falling": ("caer", "cayendo", "volcando", "inclinando"),
        "reaching": ("alcanzando", "cogiendo", "tomando"),
        "sink": ("fregadero", "lavabo"),
        "overflow": ("desbordando", "derramando"),
        "dishes": ("plato", "platos", "vajilla"),
        "washing": ("lavando", "secando", "limpiando"),
        "window": ("ventana", "ventanas"),
        "curtains": ("cortina", "cortinas"),
        "outside": ("afuera", "jardín", "césped", "árbol", "arbustos"),
        "water": ("agua",),
    },
    "zh": {
        "boy": ("男孩", "儿子"),
        "girl": ("女孩", "女儿", "妹妹"),
        "mother": ("妈妈", "母亲", "女人"),
        "cookie": ("饼干",),
        "jar": ("罐子", "罐"),
        "stool": ("凳子", "椅子"),
        "falling": ("摔倒", "跌倒", "倾斜", "翻倒"),
        "reaching": ("伸手", "拿", "抓"),
        "sink": ("水槽", "洗手池"),
        "overflow": ("溢出", "流出来", "洒出"),
        "dishes": ("盘子", "碟子", "餐具"),
        "washing": ("洗碗", "擦碗", "擦盘子"),
        "window": ("窗户", "窗"),
        "curtains": ("窗帘",),
        "outside": ("外面", "院子", "草地", "树", "灌木"),
        "water": ("水",),
    },
}

def repair_utf8_mojibake(text: object) -> str:
    """Repair UTF-8 text that was accidentally decoded and stored as Latin-1."""
    value = str(text)
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    old_markers = value.count("Ã") + value.count("Â")
    new_markers = repaired.count("Ã") + repaired.count("Â")
    return repaired if new_markers < old_markers else value

UNCERTAINTY_PHRASES = {
    "en": ("i don't know", "i do not know", "not sure", "something", "that thing"),
    "es": ("no sé", "no estoy seguro", "algo", "esa cosa"),
    "zh": ("不知道", "不确定", "什么东西", "那个东西"),
}


def canonical_language(language: str) -> str:
    """Map dataset language labels to the tokenizer/ASR language codes."""
    normalized = str(language or "").strip().lower().replace("_", "-")
    if normalized in {"", "auto", "unknown", "unspecified", "nan", "multilingual", "zh-en"}:
        return ""
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]
    base = normalized.split("-", 1)[0]
    return LANGUAGE_ALIASES.get(base, base)


def read_transcript(path_value: str) -> tuple[str, list[str], dict[str, int]]:
    if not path_value:
        return "", [], {"PAR": 0, "INV": 0}
    path = Path(path_value)
    if not path.exists():
        return "", [], {"PAR": 0, "INV": 0}
    if path.suffix.lower() == ".json":
        item = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        text = str(item.get("text", ""))
        return text, [text] if text else [], {"PAR": int(item.get("n_pac_seg", 1)), "INV": 0}
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".cha":
        return raw, [line for line in raw.splitlines() if line.strip()], {"PAR": 1 if raw.strip() else 0, "INV": 0}
    utterances: list[str] = []
    current_role = ""
    counts = {"PAR": 0, "INV": 0}
    for line in raw.splitlines():
        if line.startswith("*PAR:"):
            current_role = "PAR"
            counts["PAR"] += 1
            utterances.append(line.split(":", 1)[1])
        elif line.startswith("*INV:"):
            current_role = "INV"
            counts["INV"] += 1
        elif line.startswith("\t") and current_role == "PAR" and utterances:
            utterances[-1] += " " + line.strip()
    cleaned = [CHAT_CODE.sub(" ", TIME_MARK.sub(" ", utterance)).strip() for utterance in utterances]
    return " ".join(cleaned), cleaned, counts


def _tokens(text: str, language: str) -> list[str]:
    normalized = text.lower()
    if language.startswith("zh") and CJK.search(normalized):
        return [
            token.strip().lower()
            for token in jieba.lcut(normalized, cut_all=False)
            if token.strip() and (CJK.search(token) or WORD.search(token))
        ]
    return WORD.findall(normalized)


def _mattr(tokens: list[str], window: int = 50) -> float:
    if not tokens:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[index : index + window])) / window for index in range(len(tokens) - window + 1)]))


def _picture_description_metrics(
    text: str,
    tokens: list[str],
    language: str,
    task_type: str,
) -> dict[str, float]:
    unavailable = {
        "picture_content_unit_coverage": float("nan"),
        "picture_information_density": float("nan"),
        "picture_content_redundancy": float("nan"),
        "picture_uncertainty_rate_100w": float("nan"),
    }
    normalized_task = re.sub(r"[^a-z0-9]+", "_", str(task_type).lower()).strip("_")
    picture_tasks = {
        "picture_description",
        "cookie_theft_picture_description",
        "cookie_theft",
        "ctd",
    }
    if normalized_task not in picture_tasks or language not in PICTURE_CONTENT_UNITS:
        return unavailable
    normalized_text = text.lower()
    groups = PICTURE_CONTENT_UNITS[language]
    counts = {
        name: sum(normalized_text.count(term) for term in terms)
        for name, terms in groups.items()
    }
    unique_units = sum(count > 0 for count in counts.values())
    total_mentions = sum(counts.values())
    denominator = max(len(tokens), 1)
    uncertainty_count = sum(
        normalized_text.count(phrase) for phrase in UNCERTAINTY_PHRASES.get(language, ())
    )
    return {
        "picture_content_unit_coverage": float(unique_units / len(groups)),
        "picture_information_density": float(100.0 * unique_units / denominator),
        "picture_content_redundancy": float(
            max(total_mentions - unique_units, 0) / max(total_mentions, 1)
        ),
        "picture_uncertainty_rate_100w": float(100.0 * uncertainty_count / denominator),
    }


def transcript_metrics(
    path_value: str,
    language: str,
    duration_sec: float,
    task_type: str = "",
) -> dict[str, float]:
    text, utterances, counts = read_transcript(path_value)
    language_key = canonical_language(language)
    tokens = _tokens(text, language_key)
    utterance_lengths = [len(_tokens(utterance, language_key)) for utterance in utterances]
    denominator = max(len(tokens), 1)
    filler_count = sum(token in FILLERS.get(language_key, set()) for token in tokens)
    pronoun_count = sum(token in PRONOUNS.get(language_key, set()) for token in tokens)
    content_count = sum(token not in STOPWORDS.get(language_key, set()) for token in tokens)
    raw = Path(path_value).read_text(encoding="utf-8", errors="replace") if path_value and Path(path_value).exists() else ""
    repairs = raw.count("[/]") + raw.count("[//]")
    total_turns = counts["PAR"] + counts["INV"]
    metrics = {
        "word_count": float(len(tokens)),
        "speech_rate_wpm": float(len(tokens) / max(duration_sec / 60.0, 1e-6)) if tokens else float("nan"),
        "lexical_ttr": float(len(set(tokens)) / denominator) if tokens else float("nan"),
        "lexical_mattr50": _mattr(tokens),
        "filler_rate_100w": float(100.0 * filler_count / denominator) if tokens else float("nan"),
        "repair_rate_100w": float(100.0 * repairs / denominator) if tokens else float("nan"),
        "pronoun_ratio": float(pronoun_count / denominator) if tokens else float("nan"),
        "content_word_ratio": float(content_count / denominator) if tokens else float("nan"),
        "mean_utterance_words": float(np.mean(utterance_lengths)) if utterance_lengths else float("nan"),
        "patient_turn_count": float(counts["PAR"]),
        "interviewer_turn_count": float(counts["INV"]),
        "patient_turn_share": float(counts["PAR"] / total_turns) if total_turns else float("nan"),
        "transcript_available": float(bool(tokens)),
    }
    metrics.update(_picture_description_metrics(text, tokens, language_key, task_type))
    return metrics


def patient_intervals_from_cha(path: Path) -> list[list[float]]:
    intervals = []
    current_role = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("*PAR:"):
            current_role = "PAR"
        elif line.startswith("*INV:"):
            current_role = "INV"
        if current_role != "PAR":
            continue
        for start, end in re.findall(r"\x15(\d+)_(\d+)\x15", line):
            intervals.append([float(start) / 1000.0, float(end) / 1000.0])
    return intervals


def patient_intervals_from_segmentation(path: Path) -> list[list[float]]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [
        [float(row["begin"]) / 1000.0, float(row["end"]) / 1000.0]
        for row in frame.to_dict("records")
        if str(row.get("speaker", "")).upper() == "PAR"
    ]
