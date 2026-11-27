"""Hand-written style rules: procedural memories that apply to every reply.

Each rule has a gold version and a swap twin, a question where the rule shows,
and a deterministic check for each version. For position-based checks (units,
currencies, labels) the version whose first match comes first wins, so "21.1 km
(13.1 miles)" follows metric and "13.1 miles (21.1 km)" follows imperial. For
whole-reply checks (openings, closings) a version passes when its check holds
and the other's doesn't.
"""

import re


class _Find:
    """A check that also reports where its first match starts, for first-wins grading."""

    def __init__(self, pattern, flags=re.IGNORECASE):
        self.rx = re.compile(pattern, flags)

    def pos(self, reply):
        m = self.rx.search(reply)
        return m.start() if m else None

    def __call__(self, reply):
        return self.pos(reply) is not None


def _has(pattern):
    return _Find(pattern)


def _has_cs(pattern):  # case-sensitive
    return _Find(pattern, 0)


def _last_line(reply):
    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _first_line(reply):
    lines = [l.strip().lstrip("#*> ").strip() for l in reply.strip().splitlines() if l.strip()]
    return lines[0] if lines else ""


def _ends_with_word(word):
    rx = re.compile(rf"\b{word}\W*$", re.IGNORECASE)
    return lambda reply: bool(rx.search(reply.strip()))


def _starts_with(prefix):
    return lambda reply: _first_line(reply).lower().startswith(prefix.lower())


def _last_line_starts(prefix):
    return lambda reply: _last_line(reply).lstrip("*_ ").lower().startswith(prefix.lower())


TIME_24H = r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b(?!\s*(?:a\.?m\.?|p\.?m\.?))"
TIME_12H = r"\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)"

RULES = [
    dict(id="sr01", gold="User prefers all distances given in metric units like kilometres and metres.",
         twin="User prefers all distances given in imperial units like miles and feet.",
         question="How far is a half marathon, and how long is a typical training run for one?",
         gold_check=_has(r"\b(?:km|kilomet(?:re|er)s?|metres?|meters?)\b"),
         twin_check=_has(r"\b(?:miles?|mi|feet|foot|ft|yards?)\b")),
    dict(id="sr02", gold="User prefers temperatures stated in degrees Celsius only.",
         twin="User prefers temperatures stated in degrees Fahrenheit only.",
         question="What oven temperature should I use to roast a whole chicken, and what internal temperature means it's done?",
         gold_check=_has(r"°\s*C\b|\bdegrees? Celsius\b|\b\d+\s*C\b"),
         twin_check=_has(r"°\s*F\b|\bdegrees? Fahrenheit\b|\b\d+\s*F\b")),
    dict(id="sr03", gold="User prefers clock times written in the twenty-four hour format.",
         twin="User prefers clock times written in the twelve hour format with am and pm.",
         question="Can you sketch a daily schedule for a productive work-from-home day?",
         gold_check=_has(TIME_24H), twin_check=_has(TIME_12H)),
    dict(id="sr04", gold="User prefers calendar dates written in the year-month-day numeric format.",
         twin="User prefers calendar dates written out with the month name, like March fifth.",
         question="Give me a week-by-week timeline for planning a wedding that is six months away, starting the first week of January 2027.",
         gold_check=_has(r"\b20\d\d-[01]\d-[0-3]\d\b"),
         twin_check=_has(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b")),
    dict(id="sr05", gold="User prefers all prices and budgets given in euros.",
         twin="User prefers all prices and budgets given in US dollars.",
         question="What would a reasonable budget be for a one-week beach holiday for two people?",
         gold_check=_has(r"€|\beuros?\b|\bEUR\b"), twin_check=_has(r"\$|\bdollars?\b|\bUSD\b")),
    dict(id="sr06", gold="User prefers costs quoted in British pounds sterling.",
         twin="User prefers costs quoted in Japanese yen.",
         question="Roughly how much does it cost to build a decent gaming PC?",
         gold_check=_has(r"£|\bpounds?\b|\bGBP\b"), twin_check=_has(r"¥|\byen\b|\bJPY\b")),
    dict(id="sr07", gold="User prefers recipe ingredient amounts given in grams and millilitres.",
         twin="User prefers recipe ingredient amounts given in cups and ounces.",
         question="Can you give me a simple recipe for pancakes?",
         gold_check=_has(r"\b\d+\s*(?:g|grams?|ml|millilit(?:re|er)s?)\b"),
         twin_check=_has(r"\bcups?\b|\b(?:oz|ounces?)\b")),
    dict(id="sr08", gold="User prefers speeds given in kilometres per hour.",
         twin="User prefers speeds given in miles per hour.",
         question="How fast do cheetahs, racehorses and greyhounds run at top speed?",
         gold_check=_has(r"\bkm/h\b|\bkph\b|\bkilomet(?:re|er)s? (?:per|an) hour\b"),
         twin_check=_has(r"\bmph\b|\bmiles (?:per|an) hour\b")),
    dict(id="sr09", gold="User prefers fuel economy given in litres per hundred kilometres.",
         twin="User prefers fuel economy given in miles per gallon.",
         question="Is a hybrid car much more fuel efficient than a regular petrol car?",
         gold_check=_has(r"L/100\s*km|lit(?:re|er)s? per (?:100|hundred) kilomet"),
         twin_check=_has(r"\bmpg\b|\bmiles per gallon\b")),
    dict(id="sr10", gold="User prefers meeting times expressed in UTC rather than local time.",
         twin="User prefers meeting times expressed in Tokyo time, marked JST.",
         question="When should a team spread across London, New York and Singapore hold a weekly call?",
         gold_check=_has_cs(r"\bUTC\b"), twin_check=_has_cs(r"\bJST\b")),
    dict(id="sr11", gold="User prefers every reply to end with the single closing word Cheers.",
         twin="User prefers every reply to end with the closing words Take care.",
         question="What are some good ways to stay focused while studying?",
         gold_check=_ends_with_word("Cheers"), twin_check=_ends_with_word(r"Take care")),
    dict(id="sr12", gold="User prefers every reply to open with the words Short answer, then a colon.",
         twin="User prefers every reply to open with the words Quick take, then a colon.",
         question="Should I learn Python or JavaScript as my first programming language?",
         gold_check=_starts_with("short answer:"), twin_check=_starts_with("quick take:")),
    dict(id="sr13", gold="User prefers every reply to finish with a final line starting with P.S.",
         twin="User prefers every reply to finish with a final line starting with Note, then a colon.",
         question="How do I keep basil alive on a windowsill?",
         gold_check=_last_line_starts("p.s."), twin_check=_last_line_starts("note:")),
    dict(id="sr14", gold="User prefers being addressed as Dr. Rivera in every reply.",
         twin="User prefers being addressed as Captain Rivera in every reply.",
         question="Can you help me think through whether to accept a job offer in another city?",
         gold_check=_has(r"\bDr\.? Rivera\b"), twin_check=_has(r"\bCaptain Rivera\b")),
    dict(id="sr15", gold="User prefers instructions broken into parts labelled Step 1, Step 2 and so on.",
         twin="User prefers instructions broken into parts labelled Phase 1, Phase 2 and so on.",
         question="How do I repaint a bedroom wall?",
         gold_check=_has(r"\bStep 1\b"), twin_check=_has(r"\bPhase 1\b")),
    dict(id="sr16", gold="User prefers choices presented as Option A, Option B and Option C.",
         twin="User prefers choices presented as Choice 1, Choice 2 and Choice 3.",
         question="What are my options for getting from the airport to the city centre late at night?",
         gold_check=_has(r"\bOption A\b"), twin_check=_has(r"\bChoice 1\b")),
    dict(id="sr17", gold="User prefers every reply to begin with the greeting Hi Sam.",
         twin="User prefers every reply to begin with the greeting Hello Alex.",
         question="What should I pack for a weekend hiking trip?",
         gold_check=_starts_with("hi sam"), twin_check=_starts_with("hello alex")),
    dict(id="sr18", gold="User prefers furniture and room sizes given in centimetres.",
         twin="User prefers furniture and room sizes given in inches.",
         question="What size desk and chair fit comfortably in a small home office?",
         gold_check=_has(r"\b\d+\s*(?:cm|centimet(?:re|er)s?)\b"),
         twin_check=_has(r"\b\d+\s*(?:in\b|inch(?:es)?\b|\")")),
    dict(id="sr19", gold="User prefers each reply to state a confidence level, as Confidence: high, medium or low.",
         twin="User prefers each reply to state a certainty score, as Certainty: a number out of ten.",
         question="Will electric cars be cheaper than petrol cars to buy within ten years?",
         gold_check=_has(r"\bConfidence:\s*(?:high|medium|low)\b"),
         twin_check=_has(r"\bCertainty:\s*\d+\s*(?:/|out of)\s*10\b")),
    dict(id="sr20", gold="User prefers every reply to close with one follow-up question starting Want me to.",
         twin="User prefers every reply to close with one follow-up question starting Should I.",
         question="What are the basics of starting a vegetable garden?",
         gold_check=lambda r: _last_line(r).lstrip("*_ ").lower().startswith("want me to") and _last_line(r).rstrip("*_ ").endswith("?"),
         twin_check=lambda r: _last_line(r).lstrip("*_ ").lower().startswith("should i") and _last_line(r).rstrip("*_ ").endswith("?")),
]


def grade(rule, reply, version):
    """True if the reply follows `version` ("gold" or "twin")."""
    own_check = rule[f"{version}_check"]
    other_check = rule[("twin" if version == "gold" else "gold") + "_check"]
    if isinstance(own_check, _Find) and isinstance(other_check, _Find):
        own, other = own_check.pos(reply), other_check.pos(reply)
        return own is not None and (other is None or own < other)
    return own_check(reply) and not other_check(reply)


def _selftest():
    r = {x["id"]: x for x in RULES}
    cases = [
        ("sr01", "It is 21.1 km long. Training runs are 8 to 10 km.", "gold", True),
        ("sr01", "It is 13.1 miles (21.1 km).", "gold", False),
        ("sr01", "It is 21.1 km (about 13.1 miles).", "gold", True),
        ("sr01", "It is 13.1 miles (21.1 km).", "twin", True),
        ("sr03", "Start at 08:30, lunch at 12:30, stop at 17:00.", "gold", True),
        ("sr03", "Start at 8:30 am, lunch at 12:30 pm.", "twin", True),
        ("sr03", "Start at 8:30 am, lunch at 12:30 pm.", "gold", False),
        ("sr11", "Stay hydrated and take breaks.\n\nCheers", "gold", True),
        ("sr11", "Stay hydrated.\n\nTake care!", "twin", True),
        ("sr12", "**Short answer:** Python.", "gold", True),
        ("sr13", "Water it often.\n\nP.S. Pinch the flowers.", "gold", True),
        ("sr17", "Hi Sam! Pack layers.", "gold", True),
        ("sr19", "Probably. Confidence: medium", "gold", True),
        ("sr20", "Start small.\n\nWant me to suggest easy crops?", "gold", True),
        ("sr20", "Start small.\n\nShould I suggest easy crops?", "twin", True),
        ("sr04", "Week 1 (2027-01-04): book the venue.", "gold", True),
        ("sr04", "Week 1 (January 4): book the venue.", "twin", True),
    ]
    bad = [(i, t, v) for i, t, v, want in cases if grade(r[i], t, v) != want]
    assert not bad, bad
    assert len(RULES) == 20 and len({x["id"] for x in RULES}) == 20
    print(f"style rules: {len(RULES)} rules, {len(cases)} check cases pass")


if __name__ == "__main__":
    _selftest()
