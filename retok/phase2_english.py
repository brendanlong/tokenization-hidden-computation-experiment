"""English-only prompt set for the v2 survey (80 prompts, 8 registers).

Design goals, in order:

1. **Same task for every model.** All prompts are English-in, English-out, so
   cross-model differences measure the model, not which language it chose to
   answer in (the v1 survey's main confound — small models answer non-English
   prompts in English, and script determines the structurally-possible
   non-canonical rate).
2. **Every prompt carries a checkable instruction** (a length, a format, a
   topic constraint), so a judge can grade compliance and rates can be
   reported conditional on the model actually doing the task.
3. **Register diversity.** Non-canonical seams concentrate in specific
   surface phenomena (whitespace runs, punctuation+newline boundaries, rare
   words, formatting characters), so the set spans prose, dialogue, technical
   writing, list/markdown-heavy output, casual text, and rewrites.

The registers are the dict keys so the existing per-domain machinery reports
a per-register breakdown for free. Prompts deliberately avoid digit-heavy
tasks (measured separately in the RL arms) and non-English text.
"""

from __future__ import annotations

ENGLISH_PROMPTS: dict[str, list[str]] = {
    "explanation": [
        "Explain how a refrigerator keeps food cold.",
        "Explain why the sky is blue to a curious twelve-year-old.",
        "Explain the difference between weather and climate.",
        "Explain how vaccines train the immune system.",
        "Explain why ice floats on water.",
        "Explain how compound interest works, with one worked example.",
        "Explain what a supply chain is using a loaf of bread as the example.",
        "Explain how noise-cancelling headphones work.",
        "Explain the difference between a virus and a bacterium.",
        "Explain how tides are caused.",
    ],
    "story": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Write a two-paragraph story that begins with the sentence: The last train left without her.",
        "Write a short ghost story set in a public library.",
        "Write a story about a chef who loses their sense of taste, with a hopeful ending.",
        "Write a short fable about a stubborn river and a patient mountain, ending with a moral.",
        "Write the opening scene of a detective novel set in a small coastal town.",
        "Write a short story told entirely from the point of view of a house cat.",
        "Write a story about two strangers stuck in an elevator who discover they share a secret.",
        "Write a bedtime story about a dragon who is afraid of the dark.",
        "Write a short story where the twist is revealed only in the final sentence.",
    ],
    "dialogue": [
        "Write a dialogue between a barista and a customer who can't decide what to order.",
        "Write a phone conversation between a landlord and a tenant about a broken heater.",
        "Write a dialogue between two hikers arguing about whether to turn back before a storm.",
        "Write an interview between a journalist and a retired astronaut.",
        "Write a conversation between a driving instructor and a nervous student during a lesson.",
        "Write a dialogue between siblings deciding what to cook for their parents' anniversary.",
        "Write a customer-support chat about a package that arrived damaged.",
        "Write a conversation between a museum guide and a skeptical visitor about a famous painting.",
    ],
    "technical": [
        "Describe the steps to safely jump-start a car with cables.",
        "Write instructions for changing a bicycle tire.",
        "Describe how to set up a tent in high wind.",
        "Write a troubleshooting guide for a laptop that won't turn on.",
        "Describe the procedure for repotting a root-bound houseplant.",
        "Write instructions for brewing coffee with a French press.",
        "Describe how to read a nutrition label and what to look for.",
        "Write a beginner's guide to sharpening a kitchen knife with a whetstone.",
        "Describe how to bleed a household radiator.",
        "Write step-by-step instructions for backing up photos from a phone.",
    ],
    "structured": [
        "Make a packing list for a week-long winter trip, organized into categories.",
        "Write a recipe for vegetable soup with an ingredients list and numbered steps.",
        "Create a weekly workout plan for a beginner, formatted as a table or day-by-day list.",
        "List ten book recommendations for someone who liked mystery novels, with one sentence each.",
        "Write an agenda for a one-hour team meeting about a delayed project.",
        "Make a comparison list of pros and cons for renting versus buying a home.",
        "Write a checklist for moving out of an apartment.",
        "Create a study schedule for a student with exams in three subjects in two weeks.",
        "List the steps of the scientific method with a one-line example for each.",
        "Write a simple monthly budget outline for a college student, with categories and rough percentages.",
        "Make a glossary of ten common cooking terms with one-sentence definitions.",
        "Write a FAQ with five questions and answers for a neighborhood tool-lending library.",
    ],
    "casual": [
        "Write a text message to a friend cancelling dinner plans, and make it apologetic but funny.",
        "Write a social media post about finally finishing a marathon.",
        "Write a short review of a fictional diner that serves excellent pancakes.",
        "Write a group-chat message trying to convince your friends to go camping this weekend.",
        "Write a thank-you note to a neighbor who watered your plants for two weeks.",
        "Write a short speech for a friend's birthday toast.",
        "Write a note to leave on a car you accidentally bumped in a parking lot.",
        "Write an out-of-office auto-reply for a two-week vacation, with a little personality.",
    ],
    "rewrite": [
        'Rewrite this sentence to be more formal: "hey can u send me that report asap thx".',
        'Summarize this in one sentence: "The meeting ran long because the projector broke, the agenda was missing, and half the team joined late; we agreed to reconvene Thursday with a printed agenda and a working projector."',
        'Rewrite this passage in simpler words: "The municipality\'s remuneration framework necessitates comprehensive documentation prior to disbursement."',
        'Make this sound more enthusiastic: "The event was fine. The food was okay. Some people seemed to enjoy it."',
        'Rewrite this as polite feedback: "Your draft is confusing and way too long and the ending makes no sense."',
        'Turn these notes into a paragraph: "flight delayed 3h - missed connection - hotel voucher - arrived noon next day - luggage fine".',
        'Rewrite this warning as a friendly reminder: "Employees who fail to badge in will be reported to management."',
        'Expand this into two or three sentences: "Store closed Monday. Reopening Tuesday 9am."',
    ],
    "opinion": [
        "Write a persuasive paragraph arguing that public libraries are still essential.",
        "Argue for or against homework in elementary school, taking a clear side.",
        "Write a letter to a city council supporting more protected bike lanes.",
        "Make the strongest case you can that breakfast is overrated.",
        "Write a short opinion piece on whether remote work is better for most people.",
        "Argue that learning a musical instrument as an adult is worth it.",
        "Write a paragraph defending the unpopular opinion that winter is the best season.",
        "Make a balanced case for and against zoos, then state your own view.",
    ],
    "qa": [
        "What causes hiccups, and what actually helps stop them?",
        "Why do cats purr?",
        "What is the difference between baking soda and baking powder?",
        "Why do we dream?",
        "How do airplanes stay in the air?",
        "Why does bread go stale, and can you reverse it?",
        "What makes a rainbow, and why is it curved?",
        "Why do onions make you cry?",
    ],
}

# A minimal completion scaffold for models with no chat template (GPT-2,
# base Gemma). Recorded per-generation when used; the judge grades the same
# instruction either way.
BASE_SCAFFOLD = (
    "Below is a writing task. A response that completes the task follows.\n\n"
    "Task: {prompt}\n\nResponse:"
)


def n_prompts() -> int:
    return sum(len(v) for v in ENGLISH_PROMPTS.values())
