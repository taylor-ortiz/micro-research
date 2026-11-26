"""Shared settings for study 02 (memory routing), stage 1 build.

Credentials come from the SDK's normal lookup (e.g. an `ant auth login`
profile). No keys are read from or written to this repository.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STUDY01 = ROOT.parent / "01-memory-position"

ANSWER_MODEL = "claude-haiku-5-5"      # model under test
GENERATOR_MODEL = "claude-opus-5-5"    # writes gold memories, questions, sessions
JUDGE_MODEL = "claude-haiku-5-5"       # grades free-text checks
AUDIT_MODEL = "claude-sonnet-5-5"      # re-grades a random 10% of judged replies
ALLERGY_JUDGE_MODEL = "claude-sonnet-5-5"  # decides allergy replies that hit the word list

# $ per million tokens (input, output), standard rate. Haiku prompts over 100k
# tokens cost 5x. Batches are 50% off.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00)}

# Store composition at 1,000 memories: ~60% semantic, 25% episodic, 15% procedural.
STORE_SIZE = 1000
N_SEMANTIC, N_EPISODIC, N_PROCEDURAL = 600, 250, 150
N_TASK_DECOYS = 100            # procedural distractors that use the mock tools
# haystack_v1 categories excluded from the store: their items set currency, time
# zones, dates and forms of address, which would contradict gold style rules.
EXCLUDED_HAYSTACK_CATEGORIES = {"units_locale", "address_naming"}

# Questions per scenario.
SCENARIOS = {
    "task_procedure": 40,   # procedural, graded by first tool call
    "style_rule": 20,       # procedural, graded by regex
    "fact": 40,             # semantic, judge
    "allergy": 40,          # semantic via a reasoning step, word list + judge
    "past_session": 40,     # episodic, judge
    "unclear_type": 40,     # ambiguous, judge
    "no_memory": 40,        # control, judge (over-application)
}

ALLERGENS = {
    "peanut": ["peanut", "groundnut", "satay", "pad thai", "kung pao", "gado-gado",
               "gado gado", "mafe", "groundnut stew", "peanut sauce", "peanut butter"],
    "tree nut": ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
                 "macadamia", "pine nut", "praline", "marzipan", "frangipane",
                 "nutella", "baklava", "pesto", "korma"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish",
                  "scallop", "mussel", "clam", "oyster", "langoustine", "shrimp paste",
                  "paella", "gumbo", "cioppino", "bouillabaisse"],
    "sesame": ["sesame", "tahini", "hummus", "halva", "halvah", "furikake",
               "gomasio", "za'atar", "zaatar", "baba ganoush", "bibimbap"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee",
              "paneer", "mozzarella", "parmesan", "ricotta", "feta", "whey",
              "custard", "ice cream", "alfredo", "bechamel", "béchamel", "raita",
              "queso", "burrata", "mascarpone"],
}

# Mock tools for task-procedure questions. Each task gets 8 of these, always
# including its target tool and the swap twin's tool.
TOOLS = {
    "travel_portal": "Search and book flights, trains and hotels through the company travel portal.",
    "web_search": "Search the public web and return the top results.",
    "linear_create_issue": "Create an issue in Linear.",
    "jira_create_issue": "Create a ticket in Jira.",
    "slack_post": "Post a message to a Slack channel or person.",
    "email_send": "Send an email.",
    "calendar_find_slot": "Find free time on the user's calendar.",
    "calendar_create_event": "Create a calendar event.",
    "notion_search": "Search pages in Notion.",
    "drive_search": "Search files in Google Drive.",
    "notion_create_page": "Create a page in Notion.",
    "docs_create": "Create a Google Doc.",
    "github_search_code": "Search code in GitHub repositories.",
    "expense_submit": "Submit an expense report with a receipt.",
    "spreadsheet_append": "Append a row to a spreadsheet.",
    "sql_query": "Run a read-only SQL query on the analytics warehouse.",
    "crm_lookup": "Look up a customer record in the CRM.",
    "support_ticket_search": "Search past customer support tickets.",
    "dropbox_upload": "Upload a file to Dropbox.",
    "drive_upload": "Upload a file to Google Drive.",
}
TOOLS_PER_TASK = 8
TOOL_SCREEN_MAX_RATE = 0.30     # keep a task if no-memory target-tool rate <= 30%
ALLERGY_SCREEN_MIN_RATE = 0.60  # keep an allergy query if no-memory hit rate >= 60%
SCREEN_SAMPLES = 5

MEMORY_INSTRUCTION = ("Below are memories you have stored about this user from earlier "
                      "conversations, as JSON. Use them when they are relevant.")
