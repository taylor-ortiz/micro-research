"""Shared settings for haystack generation and validation (stage 0).

Everything a reviewer needs to reproduce haystack_v1 lives here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Load ANTHROPIC_API_KEY (and anything else) from the project's .env file,
# without overriding variables already set in the shell.
import os

_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _key, _sep, _value = _line.strip().partition("=")
        _value = _value.strip().strip('"').strip("'")
        # Skip the placeholder so it can't shadow an `ant auth login` profile.
        if _sep and not _key.startswith("#") and _key not in os.environ and _value != "paste-your-key-here":
            os.environ[_key] = _value

# Model under test. Token lengths are measured on this model's tokenizer.
TARGET_MODEL = "claude-opus-5-5"
# Model that writes candidates and reviews them.
GENERATOR_MODEL = "claude-opus-5-5"

NEEDLE = "User prefers all responses to be formatted as Markdown tables."

# Marker needle: an arbitrary behaviour with ~0% base rate, so every hit is
# memory-driven (20 tokens on claude-opus-5-5 and claude-haiku-5-5).
MARKER_NEEDLE = "User prefers every reply to end with the single closing word Cheers."

# Stage 2 needle family: one is drawn per trial; all are 20-21 tokens on the
# target tokenizer. The haystack must not touch ANY of these behaviours.
NEEDLE_FAMILY = {
    "tables": NEEDLE,                                                                          # 20
    "follow_up_question": "User prefers every reply to finish with one follow-up question for them.",  # 20
    "summary_first": "User prefers each reply to open with a one-sentence summary line.",           # 20
    "metric_units": "User prefers all measurements, distances and weights given in metric units.",  # 20
    "british_spelling": "User prefers British English spellings, like colour and favour, in every response.",  # 21
}

# Item length window, in tokens of the preference text alone, centred on the
# needle (20 tokens on claude-opus-5-5; ~2 tokens per word for this text).
MIN_TOKENS, MAX_TOKENS = 18, 24

# Inert: facts about the user's life that should not change a response.
INERT_CATEGORIES = {
    "food_drink": "food, drinks, diet, cooking and eating habits",
    "travel": "travel habits, seats, hotels, packing, destinations",
    "home": "home, household routines, temperature, furniture, cleaning",
    "pets": "pets and pet care routines",
    "fitness": "exercise, sports, outdoor activity, sleep",
    "scheduling": "calendar habits, meeting times, errands, weekly routines",
    "shopping": "stores, brands, buying habits, deliveries",
    "entertainment": "music, films, TV, podcasts, books, games",
    "hobbies": "crafts, gardening, collecting, making things",
    "transport": "commuting, cars, bikes, public transit, parking",
    "work_habits": "desk setup, focus habits, office routines (not how answers are written)",
    "social": "friends, family events, gifts, celebrations",
}

# Output-affecting but format-neutral: changes what a response says, never
# its structure or layout.
OUTPUT_NEUTRAL_CATEGORIES = {
    "units_locale": "currency, date and time conventions, time zone, calendar conventions (NOT units of measure, NOT spelling)",
    "address_naming": "how to address the user, names and titles to use for the user or their people",
    "content_choices": "which programming language, examples, analogies or sources to favour in answers",
}

CANDIDATES_PER_INERT_CATEGORY = 300       # 12 x 300 = 3,600
CANDIDATES_PER_NEUTRAL_CATEGORY = 280     # 3 x 280 = 840; ~58% yield keeps them ~15% of the kept pool
# Probe yield: ~80% inert, ~58% neutral, so ~4,440 candidates should leave ~3,300.
BATCH_SIZE = 50                           # items per generation call

# Words that touch structure, length, or instruction-override framing.
BANNED_WORDS = [
    "markdown", "table", "tables", "tabular", "list", "lists", "bullet", "bullets",
    "bulleted", "heading", "headings", "header", "headers", "format", "formatted",
    "formatting", "structure", "structured", "layout", "concise", "brief", "briefly",
    "length", "short", "long-form", "paragraph", "paragraphs", "summary", "summaries",
    "always", "never", "must", "override", "ignore", "rule", "rules", "column",
    "columns", "section", "sections", "outline", "step-by-step", "numbered",
    # Needle-family behaviours: units of measure, spelling variety, follow-up questions.
    "metric", "imperial", "celsius", "fahrenheit", "degrees", "kilometres", "kilometers",
    "kilogram", "kilograms", "litre", "litres", "liter", "liters", "miles", "pounds",
    "ounces", "gallons", "inches", "spelling", "spellings", "spelled", "spelt", "british",
    "question", "questions", "follow-up",
]

# Lexical similarity thresholds (TF-IDF cosine; see validate.py).
NEAR_DUP_THRESHOLD = 0.80
NEEDLE_SIM_THRESHOLD = 0.40
