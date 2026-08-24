#!/usr/bin/env python3
"""
ThinkSpark taxonomy + multilingual filler/backchannel lexicon
=============================================================
Single source of truth for:

  * LANGUAGES  — language code -> {name, native_name, script, family}
  * SCRIPTS    — script tag -> unicode block ranges (for hard script validation)
  * REGISTERS  — formal / casual / urban_mixed
  * INTENTS    — the context-intent taxonomy the model classifies over
  * EMOTIONS   — emotional colour of the moment
  * FILLER_TYPES — sound | word | sound_word | words | none
  * LEXICON    — language -> intent -> {type -> [surface forms]}  (the curated vocab)

What ThinkSpark actually is
---------------------------
A voice-AI agent runs STT -> LLM -> TTS. Between the user finishing (STT) and the
agent's real reply (TTS) there is dead air. ThinkSpark fills that gap with a
*human* thinking sound / backchannel — "hmm", "अच्छा", "एक सेकंड", "sí sí",
"ええと" — in the RIGHT language, script, register and emotion for the moment.

Input vs context
----------------
  * INPUT   = the last user utterance (STT text). This is the PRIMARY signal.
  * CONTEXT = optional running conversation state / persona note
              (e.g. "user is scolding the agent about a late refund").
              This MODULATES the filler but never overrides the input.

The model predicts (language, register, intent, emotion, filler_type). A curated
LEXICON lookup then samples a surface form of that type. Editing a language's
fillers = editing this file, never retraining.

NOTE ON QUALITY: the Indian-language and English entries below are written to be
natural spoken fillers. Low-resource and foreign entries are a solid starting set
and SHOULD get a native-speaker spot-check before production (see README QA step).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LANGUAGES  (code, name, native name, script tag, family)
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, dict] = {
    # --- Indian languages, native scripts ---
    "hi": {"name": "Hindi",      "native": "हिन्दी",     "script": "Deva", "family": "indo_aryan"},
    "mr": {"name": "Marathi",    "native": "मराठी",      "script": "Deva", "family": "indo_aryan"},
    "bn": {"name": "Bengali",    "native": "বাংলা",      "script": "Beng", "family": "indo_aryan"},
    "gu": {"name": "Gujarati",   "native": "ગુજરાતી",    "script": "Gujr", "family": "indo_aryan"},
    "pa": {"name": "Punjabi",    "native": "ਪੰਜਾਬੀ",     "script": "Guru", "family": "indo_aryan"},
    "ta": {"name": "Tamil",      "native": "தமிழ்",      "script": "Taml", "family": "dravidian"},
    "te": {"name": "Telugu",     "native": "తెలుగు",     "script": "Telu", "family": "dravidian"},
    "kn": {"name": "Kannada",    "native": "ಕನ್ನಡ",      "script": "Knda", "family": "dravidian"},
    "ml": {"name": "Malayalam",  "native": "മലയാളം",     "script": "Mlym", "family": "dravidian"},
    "or": {"name": "Odia",       "native": "ଓଡ଼ିଆ",       "script": "Orya", "family": "indo_aryan"},
    "as": {"name": "Assamese",   "native": "অসমীয়া",    "script": "Beng", "family": "indo_aryan"},
    "ur": {"name": "Urdu",       "native": "اردو",       "script": "Arab", "family": "indo_aryan"},
    # Hinglish = Hindi structure + romanized English, transcribed in Latin/mixed.
    "hi_en": {"name": "Hinglish", "native": "Hinglish",  "script": "Latn", "family": "code_mixed"},

    # --- English + foreign ---
    "en": {"name": "English",    "native": "English",    "script": "Latn", "family": "germanic"},
    "es": {"name": "Spanish",    "native": "Español",    "script": "Latn", "family": "romance"},
    "fr": {"name": "French",     "native": "Français",   "script": "Latn", "family": "romance"},
    "de": {"name": "German",     "native": "Deutsch",    "script": "Latn", "family": "germanic"},
    "pt": {"name": "Portuguese", "native": "Português",  "script": "Latn", "family": "romance"},
    "ja": {"name": "Japanese",   "native": "日本語",      "script": "Jpan", "family": "japonic"},
    "zh": {"name": "Mandarin",   "native": "中文",        "script": "Hans", "family": "sinitic"},
    "ar": {"name": "Arabic",     "native": "العربية",    "script": "Arab", "family": "semitic"},
    "ru": {"name": "Russian",    "native": "Русский",    "script": "Cyrl", "family": "slavic"},
}

# ---------------------------------------------------------------------------
# SCRIPTS  -> unicode ranges (inclusive). Used to HARD-reject wrong-script leaks.
# Latin/Jpan/Hans allow ASCII (loanwords, romaji, pinyin-free han is fine).
# ---------------------------------------------------------------------------
SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "Deva": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],          # Devanagari (+ extended)
    "Beng": [(0x0980, 0x09FF)],                            # Bengali/Assamese
    "Gujr": [(0x0A80, 0x0AFF)],                            # Gujarati
    "Guru": [(0x0A00, 0x0A7F)],                            # Gurmukhi
    "Taml": [(0x0B80, 0x0BFF)],                            # Tamil
    "Telu": [(0x0C00, 0x0C7F)],                            # Telugu
    "Knda": [(0x0C80, 0x0CFF)],                            # Kannada
    "Mlym": [(0x0D00, 0x0D7F)],                            # Malayalam
    "Orya": [(0x0B00, 0x0B7F)],                            # Odia
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Cyrl": [(0x0400, 0x04FF)],                            # Cyrillic
    "Latn": [(0x0041, 0x024F)],                            # Latin + extended
    "Jpan": [(0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF)],  # kana + kanji
    "Hans": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],          # CJK ideographs
}

# ---------------------------------------------------------------------------
# REGISTERS
# ---------------------------------------------------------------------------
REGISTERS: list[str] = ["formal", "casual", "urban_mixed"]

# ---------------------------------------------------------------------------
# INTENTS  — what KIND of conversational moment this is. The core label.
# `no_filler` is a first-class NEGATIVE class: many turns want silence.
# ---------------------------------------------------------------------------
INTENTS: list[str] = [
    "thinking",             # buying time while formulating
    "agreeing",             # yes, right
    "positive_ack",         # got it, mm-hm
    "negative_ack",         # no / disagree softly
    "surprised",            # oh! really?
    "empathetic",           # that's tough, I hear you
    "hesitating",           # unsure, weighing
    "clarifying_question",  # wait, which one?
    "impatient",            # come on, hurry
    "polite_interrupt",     # sorry to cut in
    "encouraging",          # go on, tell me
    "skeptical",            # hmm, not sure about that
    "sad_acknowledge",      # oh no...
    "excited",              # wow, amazing!
    "apologetic",           # sorry about that (e.g. user scolding)
    "calming",              # okay okay, let me check (defusing anger)
    "no_filler",            # say nothing — real answer needed now
]

# ---------------------------------------------------------------------------
# EMOTIONS — affective colour, predicted as a separate head.
# ---------------------------------------------------------------------------
EMOTIONS: list[str] = [
    "neutral", "warm", "curious", "concerned", "cheerful",
    "apologetic", "firm", "playful", "tired", "anxious",
]

# ---------------------------------------------------------------------------
# FILLER_TYPES — the *shape* of the emitted spark. Predicted as a head so the
# model can dynamically choose sound-only vs word vs a blend.
# ---------------------------------------------------------------------------
FILLER_TYPES: list[str] = [
    "sound",        # non-lexical: "hmm", "uh", "एँ"
    "word",         # one lexical filler: "अच्छा", "okay", "sí"
    "sound_word",   # sound + word: "hmm अच्छा", "uh okay"
    "words",        # short phrase: "एक सेकंड", "let me see"
    "none",         # emit nothing (pairs with intent=no_filler)
]

# ===========================================================================
# LEXICON  — language -> intent -> {filler_type -> [surface forms]}
# Surface forms are what actually gets spoken. The dictionary is sampled at
# inference; the model only picks (lang, register, intent, emotion, type).
# Not every (intent x type) cell is filled — empties fall back gracefully.
# ===========================================================================
LEXICON: dict[str, dict[str, dict[str, list[str]]]] = {
    "hi": {
        "thinking":     {"sound": ["हम्म", "अं", "एँ"], "word": ["अच्छा", "देखिए"], "words": ["एक सेकंड", "रुकिए ज़रा", "हम्म देखते हैं"], "sound_word": ["हम्म अच्छा"]},
        "agreeing":     {"word": ["हाँ", "बिल्कुल", "सही"], "words": ["हाँ हाँ", "बिल्कुल सही"], "sound": ["हूँ"]},
        "positive_ack": {"word": ["ठीक", "अच्छा", "जी"], "words": ["ठीक है", "जी हाँ"], "sound": ["हूँ हूँ"]},
        "negative_ack": {"word": ["नहीं", "अरे"], "words": ["नहीं नहीं", "ऐसा नहीं"], "sound": ["ना"]},
        "surprised":    {"sound": ["ओह", "अरे"], "word": ["अच्छा", "सच"], "words": ["अरे वाह", "सच में"]},
        "empathetic":   {"word": ["ओहो", "अच्छा"], "words": ["समझ सकता हूँ", "ओह अच्छा"], "sound": ["आह"]},
        "hesitating":   {"sound": ["उम्म", "एँ"], "words": ["पता नहीं", "शायद", "देखते हैं"], "word": ["हम्म"]},
        "clarifying_question": {"words": ["एक मिनट", "कौन सा", "मतलब"], "word": ["अच्छा"]},
        "impatient":    {"word": ["जल्दी", "अरे"], "words": ["जल्दी बताइए", "अरे यार"]},
        "polite_interrupt": {"words": ["माफ़ कीजिए", "एक बात", "सुनिए ज़रा"]},
        "encouraging":  {"word": ["हाँ", "बोलिए"], "words": ["हाँ बताइए", "और बताइए"], "sound": ["हूँ"]},
        "skeptical":    {"sound": ["हम्म"], "word": ["अच्छा"], "words": ["सच में?", "हम्म पता नहीं"]},
        "sad_acknowledge": {"sound": ["आह", "ओह"], "words": ["ओह नहीं", "अरे नहीं"]},
        "excited":      {"word": ["वाह", "अरे"], "words": ["अरे वाह", "क्या बात"], "sound": ["ओह"]},
        "apologetic":   {"words": ["माफ़ कीजिए", "मुझे खेद है", "क्षमा करें"], "word": ["सॉरी"]},
        "calming":      {"words": ["ठीक है ठीक है", "एक सेकंड देखता हूँ", "शांत रहिए"], "word": ["अच्छा"]},
        "no_filler":    {"none": [""]},
    },
    "mr": {
        "thinking":     {"sound": ["हम्म", "अं"], "words": ["एक मिनिट", "थांबा जरा"], "word": ["बरं"]},
        "agreeing":     {"word": ["हो", "बरोबर"], "words": ["हो हो", "बरोबर आहे"]},
        "positive_ack": {"word": ["बरं", "ठीक"], "words": ["ठीक आहे"]},
        "negative_ack": {"word": ["नाही"], "words": ["नाही नाही"]},
        "surprised":    {"sound": ["अरे"], "words": ["अरे वा", "खरंच"]},
        "empathetic":   {"words": ["समजतंय मला", "अरे बापरे"]},
        "hesitating":   {"sound": ["उम्म"], "words": ["माहित नाही", "बघू या"]},
        "clarifying_question": {"words": ["एक मिनिट", "कोणतं"]},
        "encouraging":  {"words": ["हो सांगा", "पुढे बोला"]},
        "apologetic":   {"words": ["माफ करा", "चुकलं माझं"]},
        "calming":      {"words": ["ठीक आहे ठीक आहे", "जरा थांबा"]},
        "no_filler":    {"none": [""]},
    },
    "bn": {
        "thinking":     {"sound": ["হুম", "আম"], "words": ["এক সেকেন্ড", "দাঁড়ান"], "word": ["আচ্ছা"]},
        "agreeing":     {"word": ["হ্যাঁ", "ঠিক"], "words": ["হ্যাঁ হ্যাঁ", "একদম ঠিক"]},
        "positive_ack": {"word": ["আচ্ছা", "ঠিক"], "words": ["ঠিক আছে"]},
        "negative_ack": {"word": ["না"], "words": ["না না"]},
        "surprised":    {"sound": ["ওমা", "আরে"], "words": ["তাই নাকি", "সত্যি"]},
        "empathetic":   {"words": ["বুঝতে পারছি", "আহারে"]},
        "hesitating":   {"sound": ["উম"], "words": ["জানি না", "দেখি"]},
        "encouraging":  {"words": ["হ্যাঁ বলুন", "আরও বলুন"]},
        "apologetic":   {"words": ["মাফ করবেন", "দুঃখিত"]},
        "calming":      {"words": ["ঠিক আছে ঠিক আছে", "একটু দাঁড়ান"]},
        "no_filler":    {"none": [""]},
    },
    "gu": {
        "thinking":     {"sound": ["હમ્મ", "અં"], "words": ["એક સેકન્ડ", "થોભો જરા"], "word": ["સારું"]},
        "agreeing":     {"word": ["હા", "બરાબર"], "words": ["હા હા", "બિલકુલ સાચું"]},
        "positive_ack": {"word": ["સારું", "ઠીક"], "words": ["ઠીક છે"]},
        "negative_ack": {"word": ["ના"], "words": ["ના ના"]},
        "surprised":    {"sound": ["અરે"], "words": ["અરે વાહ", "સાચે"]},
        "empathetic":   {"words": ["સમજું છું", "અરે રે"]},
        "hesitating":   {"sound": ["ઉમ્મ"], "words": ["ખબર નથી", "જોઈએ"]},
        "encouraging":  {"words": ["હા કહો", "આગળ કહો"]},
        "apologetic":   {"words": ["માફ કરો", "દિલગીર છું"]},
        "calming":      {"words": ["ઠીક છે ઠીક છે", "જરા થોભો"]},
        "no_filler":    {"none": [""]},
    },
    "pa": {
        "thinking":     {"sound": ["ਹੂੰ", "ਅੰ"], "words": ["ਇੱਕ ਸਕਿੰਟ", "ਰੁਕੋ ਜ਼ਰਾ"], "word": ["ਅੱਛਾ"]},
        "agreeing":     {"word": ["ਹਾਂ", "ਸਹੀ"], "words": ["ਹਾਂ ਹਾਂ", "ਬਿਲਕੁਲ ਸਹੀ"]},
        "positive_ack": {"word": ["ਠੀਕ", "ਅੱਛਾ"], "words": ["ਠੀਕ ਹੈ"]},
        "negative_ack": {"word": ["ਨਹੀਂ"], "words": ["ਨਹੀਂ ਨਹੀਂ"]},
        "surprised":    {"sound": ["ਓਹ"], "words": ["ਸੱਚੀ", "ਵਾਹ"]},
        "empathetic":   {"words": ["ਸਮਝ ਸਕਦਾਂ", "ਓਹੋ"]},
        "hesitating":   {"sound": ["ਉਮ"], "words": ["ਪਤਾ ਨਹੀਂ", "ਵੇਖਦੇ ਹਾਂ"]},
        "apologetic":   {"words": ["ਮਾਫ਼ ਕਰੋ", "ਸੌਰੀ"]},
        "no_filler":    {"none": [""]},
    },
    "ta": {
        "thinking":     {"sound": ["ம்ம்", "அ"], "words": ["ஒரு நிமிடம்", "பொறுங்க"], "word": ["சரி"]},
        "agreeing":     {"word": ["ஆமா", "சரி"], "words": ["ஆமா ஆமா", "சரியா இருக்கு"]},
        "positive_ack": {"word": ["சரி", "ஓகே"], "words": ["சரி சரி"]},
        "negative_ack": {"word": ["இல்ல"], "words": ["இல்ல இல்ல"]},
        "surprised":    {"sound": ["ஐயோ", "ஓ"], "words": ["அப்படியா", "நிஜமாவா"]},
        "empathetic":   {"words": ["புரியுது", "ஐயோ பாவம்"]},
        "hesitating":   {"sound": ["உம்"], "words": ["தெரியல", "பாக்கலாம்"]},
        "encouraging":  {"words": ["ஆமா சொல்லுங்க", "மேல சொல்லுங்க"]},
        "apologetic":   {"words": ["மன்னிக்கவும்", "சாரி"]},
        "calming":      {"words": ["சரி சரி", "கொஞ்சம் பொறுங்க"]},
        "no_filler":    {"none": [""]},
    },
    "te": {
        "thinking":     {"sound": ["హ్మ్మ్", "అం"], "words": ["ఒక్క నిమిషం", "ఆగండి"], "word": ["సరే"]},
        "agreeing":     {"word": ["అవును", "సరి"], "words": ["అవును అవును", "కరెక్ట్"]},
        "positive_ack": {"word": ["సరే", "ఓకే"], "words": ["సరే సరే"]},
        "negative_ack": {"word": ["కాదు"], "words": ["కాదు కాదు"]},
        "surprised":    {"sound": ["అరె", "ఓ"], "words": ["నిజంగా", "అవునా"]},
        "empathetic":   {"words": ["అర్థమవుతోంది", "అయ్యో"]},
        "hesitating":   {"sound": ["ఉమ్"], "words": ["తెలియదు", "చూద్దాం"]},
        "apologetic":   {"words": ["క్షమించండి", "సారీ"]},
        "no_filler":    {"none": [""]},
    },
    "kn": {
        "thinking":     {"sound": ["ಹ್ಮ್", "ಅಂ"], "words": ["ಒಂದು ಕ್ಷಣ", "ತಡೀರಿ"], "word": ["ಸರಿ"]},
        "agreeing":     {"word": ["ಹೌದು", "ಸರಿ"], "words": ["ಹೌದು ಹೌದು", "ಸರಿಯಾಗಿದೆ"]},
        "positive_ack": {"word": ["ಸರಿ", "ಓಕೆ"], "words": ["ಸರಿ ಸರಿ"]},
        "negative_ack": {"word": ["ಇಲ್ಲ"], "words": ["ಇಲ್ಲ ಇಲ್ಲ"]},
        "surprised":    {"sound": ["ಅಯ್ಯೋ", "ಓ"], "words": ["ನಿಜವಾ", "ಹೌದಾ"]},
        "empathetic":   {"words": ["ಅರ್ಥ ಆಗುತ್ತೆ", "ಅಯ್ಯೋ ಪಾಪ"]},
        "hesitating":   {"sound": ["ಉಮ್"], "words": ["ಗೊತ್ತಿಲ್ಲ", "ನೋಡೋಣ"]},
        "apologetic":   {"words": ["ಕ್ಷಮಿಸಿ", "ಸಾರಿ"]},
        "no_filler":    {"none": [""]},
    },
    "ml": {
        "thinking":     {"sound": ["ഉം", "ഹും"], "words": ["ഒരു നിമിഷം", "നിൽക്കൂ"], "word": ["ശരി"]},
        "agreeing":     {"word": ["അതെ", "ശരി"], "words": ["അതെ അതെ", "ശരിയാണ്"]},
        "positive_ack": {"word": ["ശരി", "ഓകെ"], "words": ["ശരി ശരി"]},
        "negative_ack": {"word": ["അല്ല"], "words": ["അല്ല അല്ല"]},
        "surprised":    {"sound": ["അയ്യോ", "ഓ"], "words": ["ശരിക്കും", "അതെയോ"]},
        "empathetic":   {"words": ["മനസ്സിലാകുന്നു", "അയ്യോ പാവം"]},
        "hesitating":   {"sound": ["ഉം"], "words": ["അറിയില്ല", "നോക്കാം"]},
        "apologetic":   {"words": ["ക്ഷമിക്കണം", "സോറി"]},
        "no_filler":    {"none": [""]},
    },
    "or": {
        "thinking":     {"sound": ["ହୁଁ", "ଅଁ"], "words": ["ଏକ ମିନିଟ୍", "ରୁହନ୍ତୁ"], "word": ["ଆଚ୍ଛା"]},
        "agreeing":     {"word": ["ହଁ", "ଠିକ୍"], "words": ["ହଁ ହଁ"]},
        "positive_ack": {"word": ["ଆଚ୍ଛା", "ଠିକ୍"], "words": ["ଠିକ୍ ଅଛି"]},
        "surprised":    {"sound": ["ଆରେ"], "words": ["ସତରେ"]},
        "hesitating":   {"sound": ["ଉମ୍"], "words": ["ଜଣା ନାହିଁ"]},
        "apologetic":   {"words": ["କ୍ଷମା କରନ୍ତୁ"]},
        "no_filler":    {"none": [""]},
    },
    "as": {
        "thinking":     {"sound": ["হুম", "আম"], "words": ["এক ছেকেণ্ড", "ৰওক"], "word": ["আচ্ছা"]},
        "agreeing":     {"word": ["হয়", "ঠিক"], "words": ["হয় হয়"]},
        "positive_ack": {"word": ["আচ্ছা", "ঠিক"], "words": ["ঠিক আছে"]},
        "hesitating":   {"sound": ["উম"], "words": ["নাজানো", "চাওঁ"]},
        "apologetic":   {"words": ["ক্ষমা কৰিব"]},
        "no_filler":    {"none": [""]},
    },
    "ur": {
        "thinking":     {"sound": ["ہمم", "اُمم"], "words": ["ایک سیکنڈ", "ذرا رکیے"], "word": ["اچھا"]},
        "agreeing":     {"word": ["ہاں", "بالکل"], "words": ["ہاں ہاں", "بالکل ٹھیک"]},
        "positive_ack": {"word": ["ٹھیک", "اچھا"], "words": ["ٹھیک ہے"]},
        "negative_ack": {"word": ["نہیں"], "words": ["نہیں نہیں"]},
        "surprised":    {"sound": ["ارے", "اوہ"], "words": ["واقعی", "سچ میں"]},
        "empathetic":   {"words": ["سمجھ سکتا ہوں", "افسوس"]},
        "hesitating":   {"sound": ["اُمم"], "words": ["پتہ نہیں", "دیکھتے ہیں"]},
        "apologetic":   {"words": ["معاف کیجیے", "معذرت"]},
        "no_filler":    {"none": [""]},
    },
    "hi_en": {
        "thinking":     {"sound": ["hmm", "umm"], "words": ["ek second", "wait yaar", "let me dekhta hoon"], "word": ["accha"], "sound_word": ["hmm accha"]},
        "agreeing":     {"word": ["haan", "exactly", "bilkul"], "words": ["haan haan", "yes bilkul"]},
        "positive_ack": {"word": ["okay", "theek", "done"], "words": ["theek hai", "ok ok"]},
        "negative_ack": {"word": ["nahi", "arre"], "words": ["nahi yaar", "no no"]},
        "surprised":    {"sound": ["arre", "oh"], "words": ["sach mein", "oh wow", "arre wah"]},
        "empathetic":   {"words": ["samajh sakta hoon", "oh no yaar"]},
        "hesitating":   {"sound": ["umm"], "words": ["pata nahi", "dekhte hain", "hmm let me see"]},
        "clarifying_question": {"words": ["ek min", "kaunsa wala", "matlab?"]},
        "impatient":    {"words": ["jaldi yaar", "come on"], "word": ["arre"]},
        "encouraging":  {"words": ["haan bolo", "go on"], "word": ["haan"]},
        "skeptical":    {"sound": ["hmm"], "words": ["really?", "sach mein?"]},
        "excited":      {"word": ["wah", "wow"], "words": ["arre wah", "so good yaar"]},
        "apologetic":   {"words": ["sorry yaar", "my bad", "maaf karna"]},
        "calming":      {"words": ["okay okay", "ek sec dekhta hoon", "relax yaar"]},
        "no_filler":    {"none": [""]},
    },
    "en": {
        "thinking":     {"sound": ["hmm", "uh", "erm"], "words": ["let me see", "one sec", "give me a moment"], "word": ["okay"], "sound_word": ["hmm okay"]},
        "agreeing":     {"word": ["right", "exactly", "yeah"], "words": ["yeah yeah", "for sure"]},
        "positive_ack": {"word": ["okay", "got it", "sure"], "words": ["mm-hm", "gotcha"], "sound": ["mm-hm"]},
        "negative_ack": {"word": ["no", "nope"], "words": ["not really", "no no"]},
        "surprised":    {"sound": ["oh", "whoa"], "words": ["oh really", "no way"]},
        "empathetic":   {"words": ["I hear you", "that's tough", "oh no"], "sound": ["ah"]},
        "hesitating":   {"sound": ["umm", "err"], "words": ["I'm not sure", "let me think"]},
        "clarifying_question": {"words": ["wait which one", "sorry, what", "you mean?"]},
        "impatient":    {"words": ["come on", "any time now"], "sound": ["ugh"]},
        "polite_interrupt": {"words": ["sorry to jump in", "quick thing", "if I may"]},
        "encouraging":  {"word": ["yeah", "go on"], "words": ["tell me more", "and then?"], "sound": ["mm-hm"]},
        "skeptical":    {"sound": ["hmm"], "words": ["I don't know", "are you sure?"]},
        "sad_acknowledge": {"sound": ["oh", "ah"], "words": ["oh no", "I'm sorry"]},
        "excited":      {"word": ["wow", "nice"], "words": ["that's amazing", "oh wow"]},
        "apologetic":   {"words": ["I'm sorry", "my apologies", "that's on me"], "word": ["sorry"]},
        "calming":      {"words": ["okay okay", "let me check that", "I understand"], "word": ["alright"]},
        "no_filler":    {"none": [""]},
    },
    "es": {
        "thinking":     {"sound": ["mmm", "eh"], "words": ["a ver", "un momento", "déjame ver"], "word": ["bueno"]},
        "agreeing":     {"word": ["sí", "claro", "vale"], "words": ["sí sí", "claro claro"]},
        "positive_ack": {"word": ["vale", "bien"], "words": ["de acuerdo"]},
        "negative_ack": {"word": ["no"], "words": ["no no", "para nada"]},
        "surprised":    {"sound": ["oh", "ah"], "words": ["¿en serio?", "no me digas"]},
        "empathetic":   {"words": ["te entiendo", "vaya"]},
        "hesitating":   {"sound": ["eh", "mmm"], "words": ["no sé", "a ver"]},
        "encouraging":  {"words": ["sí dime", "cuéntame"]},
        "apologetic":   {"words": ["lo siento", "perdona"]},
        "no_filler":    {"none": [""]},
    },
    "fr": {
        "thinking":     {"sound": ["euh", "hmm"], "words": ["voyons", "un instant", "laissez-moi voir"], "word": ["bon"]},
        "agreeing":     {"word": ["oui", "exact", "d'accord"], "words": ["oui oui", "tout à fait"]},
        "positive_ack": {"word": ["d'accord", "ok"], "words": ["très bien"]},
        "negative_ack": {"word": ["non"], "words": ["non non"]},
        "surprised":    {"sound": ["oh", "ah"], "words": ["vraiment?", "ah bon?"]},
        "empathetic":   {"words": ["je comprends", "oh là là"]},
        "hesitating":   {"sound": ["euh"], "words": ["je ne sais pas", "on verra"]},
        "apologetic":   {"words": ["désolé", "je m'excuse"]},
        "no_filler":    {"none": [""]},
    },
    "de": {
        "thinking":     {"sound": ["ähm", "hmm"], "words": ["mal sehen", "einen Moment", "lass mich überlegen"], "word": ["also"]},
        "agreeing":     {"word": ["ja", "genau", "klar"], "words": ["ja ja", "genau genau"]},
        "positive_ack": {"word": ["okay", "gut"], "words": ["alles klar"]},
        "negative_ack": {"word": ["nein"], "words": ["nein nein"]},
        "surprised":    {"sound": ["oh", "ach"], "words": ["wirklich?", "echt jetzt?"]},
        "empathetic":   {"words": ["ich verstehe", "oh je"]},
        "hesitating":   {"sound": ["ähm"], "words": ["ich weiß nicht", "mal schauen"]},
        "apologetic":   {"words": ["tut mir leid", "Entschuldigung"]},
        "no_filler":    {"none": [""]},
    },
    "pt": {
        "thinking":     {"sound": ["hmm", "é"], "words": ["deixa ver", "um momento"], "word": ["bem"]},
        "agreeing":     {"word": ["sim", "claro", "certo"], "words": ["sim sim"]},
        "positive_ack": {"word": ["tá", "ok"], "words": ["tá bom"]},
        "surprised":    {"sound": ["oh"], "words": ["sério?", "não acredito"]},
        "hesitating":   {"sound": ["é"], "words": ["não sei", "vamos ver"]},
        "apologetic":   {"words": ["desculpa", "sinto muito"]},
        "no_filler":    {"none": [""]},
    },
    "ja": {
        "thinking":     {"sound": ["えっと", "うーん", "あの"], "words": ["ちょっと待って", "そうですね"], "word": ["なるほど"]},
        "agreeing":     {"word": ["はい", "そう", "うん"], "words": ["そうそう", "その通り"]},
        "positive_ack": {"word": ["はい", "オーケー"], "sound": ["うん"], "words": ["わかりました"]},
        "negative_ack": {"word": ["いや", "ううん"], "words": ["いやいや"]},
        "surprised":    {"sound": ["えっ", "おお"], "words": ["本当に", "まじで"]},
        "empathetic":   {"words": ["わかります", "大変ですね"]},
        "hesitating":   {"sound": ["うーん", "えー"], "words": ["わからない", "どうかな"]},
        "encouraging":  {"word": ["うん"], "words": ["それで?", "続けて"]},
        "apologetic":   {"words": ["すみません", "ごめんなさい"]},
        "no_filler":    {"none": [""]},
    },
    "zh": {
        "thinking":     {"sound": ["嗯", "呃"], "words": ["让我想想", "等一下", "这个嘛"], "word": ["好"]},
        "agreeing":     {"word": ["对", "是", "没错"], "words": ["对对对", "是的是的"]},
        "positive_ack": {"word": ["好", "好的"], "words": ["明白了"], "sound": ["嗯"]},
        "negative_ack": {"word": ["不", "不是"], "words": ["不不不"]},
        "surprised":    {"sound": ["哦", "哇"], "words": ["真的吗", "不会吧"]},
        "empathetic":   {"words": ["我理解", "哎呀"]},
        "hesitating":   {"sound": ["呃", "嗯"], "words": ["不知道", "看看吧"]},
        "apologetic":   {"words": ["对不起", "抱歉"]},
        "no_filler":    {"none": [""]},
    },
    "ar": {
        "thinking":     {"sound": ["مم", "اه"], "words": ["لحظة", "دعني أفكر"], "word": ["طيب"]},
        "agreeing":     {"word": ["نعم", "أكيد", "صح"], "words": ["نعم نعم"]},
        "positive_ack": {"word": ["طيب", "حسناً"], "words": ["تمام"]},
        "negative_ack": {"word": ["لا"], "words": ["لا لا"]},
        "surprised":    {"sound": ["اوه"], "words": ["حقاً؟", "لا يعقل"]},
        "hesitating":   {"sound": ["مم"], "words": ["لا أعرف", "لنرى"]},
        "apologetic":   {"words": ["آسف", "المعذرة"]},
        "no_filler":    {"none": [""]},
    },
    "ru": {
        "thinking":     {"sound": ["ммм", "эээ"], "words": ["дайте подумать", "секунду"], "word": ["так"]},
        "agreeing":     {"word": ["да", "точно", "конечно"], "words": ["да да"]},
        "positive_ack": {"word": ["хорошо", "ладно"], "words": ["понятно"]},
        "negative_ack": {"word": ["нет"], "words": ["нет нет"]},
        "surprised":    {"sound": ["ох", "ого"], "words": ["правда?", "серьёзно?"]},
        "hesitating":   {"sound": ["эээ"], "words": ["не знаю", "посмотрим"]},
        "apologetic":   {"words": ["извините", "прошу прощения"]},
        "no_filler":    {"none": [""]},
    },
}

# ---------------------------------------------------------------------------
# Convenience index maps (stable ordering -> integer label ids)
# ---------------------------------------------------------------------------
LANG_LIST = list(LANGUAGES.keys())
LANG2ID = {c: i for i, c in enumerate(LANG_LIST)}
REGISTER2ID = {r: i for i, r in enumerate(REGISTERS)}
INTENT2ID = {t: i for i, t in enumerate(INTENTS)}
EMOTION2ID = {e: i for i, e in enumerate(EMOTIONS)}
FILLERTYPE2ID = {t: i for i, t in enumerate(FILLER_TYPES)}

ID2LANG = {i: c for c, i in LANG2ID.items()}
ID2REGISTER = {i: r for r, i in REGISTER2ID.items()}
ID2INTENT = {i: t for t, i in INTENT2ID.items()}
ID2EMOTION = {i: e for e, i in EMOTION2ID.items()}
ID2FILLERTYPE = {i: t for t, i in FILLERTYPE2ID.items()}


def sample_filler(lang: str, intent: str, ftype: str, rng=None) -> str:
    """Look up a surface form for (lang, intent, type); fall back gracefully."""
    import random
    rng = rng or random
    lang_tbl = LEXICON.get(lang) or LEXICON.get("en", {})
    intent_tbl = lang_tbl.get(intent)
    if not intent_tbl:
        # fall back to a neutral thinking filler in that language
        intent_tbl = lang_tbl.get("thinking", {})
    forms = intent_tbl.get(ftype)
    if not forms:
        # any available type for this intent
        for t in ("word", "sound", "words", "sound_word", "none"):
            if intent_tbl.get(t):
                forms = intent_tbl[t]
                break
    if not forms:
        return ""
    return rng.choice(forms)
