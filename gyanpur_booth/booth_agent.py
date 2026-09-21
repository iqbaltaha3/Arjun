#!/usr/bin/env python3
"""
Booth Agent — product face. Pure ReAct orchestrator.

The LLM decides when to call voter / history / finish.
No keyword routing. No deterministic plans.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import domain
from .common import GeminiClient, ReactStep, ToolFailure, safe_json
from .history_agent import HistoryAgent
from .textutil import to_plain_text
from .portfolio_agent import PortfolioAgent
from .voter_agent import VoterAgent

MAX_STEPS = 5

CAPABILITIES_TEXT = f"""
I am the {domain.CONSTITUENCY['product_name']} for
{domain.CONSTITUENCY['name']} Vidhan Sabha
({domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}).

I am a political consultant for Uttar Pradesh elections. I work only from
the data collected for this constituency, and I can help with these kinds of
questions:

1) Voter roll (live database)
   - Counts by region, caste, religion, gender, age
   - Surname searches, EPIC lookup, booth composition

2) Election history (Form-20)
   - Candidate / party vote totals by year (2017, 2019, 2022, 2024)
   - Winners and margins for the seat or a region (e.g. Khokhar)

3) Booth portfolios (precomputed analysis)
   - Full demographic cards for a part number (Part_1, …)
   - Named booth / school profiles from booth_analysis

4) Strategy advice
   - Recommendations built from the numbers above (margins, vote-share
     trends, community strength, booth profiles), with the reasoning shown

Ask a concrete question, for example:
  - "How many Yadav voters in Khokhar?"
  - "Who won in Khokhar in 2022?"
  - "Demographic breakdown for part number 1"
  - "How many times has BJP won Gyanpur?"
  - "Does Congress have any scope in Gyanpur?"
""".strip()



@dataclass
class ToolResult:
    tool: str
    question: str
    answer: str
    ok: bool
    error: Optional[str] = None   # plain-language reason, safe to show the user
    detail: str = ""              # technical info: logs / planner only


@dataclass
class BoothState:
    question: str
    steps: List[ReactStep] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)

    def memory(self) -> str:
        parts = []
        for s in self.steps:
            parts.append(
                f"Step {s.step} | action={s.action}\n"
                f"Thought: {s.thought}\n"
                f"Input: {s.action_input}\n"
                f"Observation: {s.observation[:3500]}"
            )
        return "\n\n".join(parts)


class BoothAgent:
    def __init__(self, llm: Optional[GeminiClient] = None, max_steps: int = MAX_STEPS):
        self.llm = llm or GeminiClient()
        self.max_steps = max_steps
        self.voter = VoterAgent(llm=self.llm)
        self.history = HistoryAgent(llm=self.llm)
        self.portfolio = PortfolioAgent(llm=self.llm)

    def _system(self) -> str:
        return f"""
You are the BOOTH AGENT — a highly skilled political consultant for
{domain.CONSTITUENCY['name']} Vidhan Sabha
({domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}), advising
candidates and campaign teams. You understand the electoral dynamics of Uttar
Pradesh: community arithmetic, booth-level organisation, alliance and
vote-transfer effects, winning margins, turnout and swing between elections.

Your expertise decides WHICH data to look at and HOW to interpret it. It is
never a source of facts. Every fact and number must come from the tools.

{domain.SYSTEM_ARCHITECTURE}

TOOLS
-----
1) voter
   Live electoral roll SQL (table `voters`). Counts, surnames, caste,
   religion, gender, age, EPIC, region composition.
   No votes/winners/years.

2) history
   Form-20 election results SQL. Votes, winners, party performance,
   how a region voted in a given year.

3) portfolio
   Precomputed booth portfolio JSON (from booth_analysis/).
   Use for: part number queries (Part_1), named booth profiles,
   detailed demographic cards, caste-gender cross-tabs, household
   structure already analysed per booth.
   Prefer portfolio over voter when the user asks for a booth
   "profile", "portfolio", "stats for part N", or a specific school/booth
   card that exists in the analysis folder.

4) finish
   Stop and answer from observations.dont return json when aswering. 
   Only return json when using tools. If you have enough evidence to answer the question, use finish.

Respond with ONLY JSON each turn while using tools:
{{
  "thought": "what the candidate still needs",
  "action": "voter" | "history" | "portfolio" | "finish",
  "action_input": "self-contained sub-question for the tool"
}}

ADVICE / STRATEGY QUESTIONS
Questions such as "does Congress have any scope", "where should I focus",
"how can I win Khokhar", "which booths are weak" need evidence first.
Gather the relevant data with the tools BEFORE finishing, for example:
- history: party vote totals and margins across all available elections,
  and how the region in question voted (trend / swing).
- voter or portfolio: community, gender and age composition of the area or
  booth concerned.
Ask each tool a complete, self-contained question so that few steps are
needed. Never finish an advice question with no evidence gathered.

ROUTING EXAMPLES
- "demographics in part number 1" → portfolio
- "who won in Khokhar in 2022" → history
- "how many Yadav voters in Khokhar" → voter (or portfolio if booth card asked)
- "how did Khokhar vote in 2022" → history
- Do not send election results to voter or portfolio.
- Do not repeat the same failed tool call with the same input.
- action_input must stand alone. Never invent numbers.
- When observations answer the question, action=finish.
""".strip()

    def plan(self, state: BoothState) -> Dict[str, Any]:
        user = f"""
Candidate question:
{state.question}

Research so far:
{state.memory() or "(none yet)"}

Next step as JSON only.
"""
        text = self.llm.chat(
            [
                {"role": "system", "content": self._system()},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            return {
                "thought": text,
                "action": "finish",
                "action_input": "",
            }

        parsed = safe_json(text)
        if not parsed:
            return {
                "thought": "Unparseable plan; finishing with available evidence.",
                "action": "finish",
                "action_input": "",
            }

        action = str(parsed.get("action", "")).lower().strip()
        if action not in {"voter", "history", "portfolio", "finish"}:
            action = "finish"
        return {
            "thought": str(parsed.get("thought", "")).strip() or "Planning.",
            "action": action,
            "action_input": str(parsed.get("action_input", state.question)).strip()
            or state.question,
        }

    def call_tool(self, name: str, question: str) -> ToolResult:
        try:
            if name == "voter":
                ans = self.voter.answer(question)
            elif name == "history":
                ans = self.history.answer(question)
            elif name == "portfolio":
                ans = self.portfolio.answer(question)
            else:
                return ToolResult(name, question, "", False, f"Unknown tool: {name}")
            return ToolResult(name, question, str(ans).strip(), True)
        except ToolFailure as e:
            print(f"  [{name}] failed: {e.message} | {e.detail[:200]}")
            return ToolResult(name, question, "", False, e.message, detail=e.detail)
        except Exception as e:
            traceback.print_exc()
            return ToolResult(
                name, question, "", False,
                f"the {name} lookup hit an unexpected error", detail=f"{type(e).__name__}: {e}",
            )

    _TOOL_LABELS = {
        "voter": "voter-roll",
        "history": "election-history",
        "portfolio": "booth-portfolio",
    }

    @staticmethod
    def _failure_message(failed: List[ToolResult]) -> str:
        reasons: List[str] = []
        for r in failed:
            if r.error and r.error not in reasons:
                reasons.append(r.error)
        return (
            "Sorry, I couldn't answer that this time: "
            + "; ".join(reasons)
            + ".\n\nThat's a problem on my side, not with your question. "
            "Please try again in a moment, or ask it in a slightly different way "
            "(for example one election at a time)."
        )

    def synthesise(self, state: BoothState) -> str:
        ok = [r for r in state.results if r.ok and r.answer]
        ok_tools = {r.tool for r in ok}
        # Report a failure only if that tool never succeeded later on a retry.
        failed = [r for r in state.results if not r.ok and r.tool not in ok_tools]

        if not ok:
            if failed:
                return self._failure_message(failed)
            # No tool was used: meta / capability question, or LLM/API failure
            q = state.question.lower()
            if any(
                x in q
                for x in (
                    "what can you",
                    "what do you do",
                    "help",
                    "capabilities",
                    "who are you",
                )
            ):
                return CAPABILITIES_TEXT
            err_bits = [
                s.thought for s in state.steps if s.thought.startswith("<<LLM_ERROR")
            ]
            if err_bits:
                print(f"[booth] LLM failure: {err_bits[0][:300]}")
                return (
                    "I could not reach the analysis service just now, so I was "
                    "unable to research that. Please try again in a moment."
                )
            return (
                "I could not gather enough evidence.\n\n" + CAPABILITIES_TEXT
            )

        evidence = "\n\n".join(
            f"[{r.tool.upper()}]\nQ: {r.question}\nA: {r.answer}" for r in ok
        )
        if failed:
            evidence += "\n\n[COULD NOT RETRIEVE]\n" + "\n".join(
                f"- {self._TOOL_LABELS.get(r.tool, r.tool)} data: {r.error}"
                for r in failed
            )
        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are Arjun, a senior political data consultant for "
                        f"Uttar Pradesh elections, advising a campaign team using "
                        f"{domain.CONSTITUENCY['name']} Vidhan Sabha data. You understand "
                        "electoral data, booth organisation, margins, vote-share trends, "
                        "turnout and demographic composition. Your job is to turn retrieved "
                        "evidence into a useful, structured written answer.\n\n"
                        "RULES\n"
                        "1. Use ONLY the tool evidence for every fact and number. Never invent numbers.\n"
                        "2. Detect the language of the user's question and write the answer in that same language. "
                        "Do not translate Hindi into English or English into Hindi unless asked.\n"
                        "3. Separate facts from interpretation. Start with the key factual findings, then add a short "
                        "evidence-based interpretation when useful. Do not merely repeat raw tool output.\n"
                        "4. For strategic questions, explain what the evidence indicates and what additional information "
                        "would be useful before making a stronger conclusion. Do not invent assumptions about voter behaviour.\n"
                        "5. If the question is ambiguous or too broad to answer reliably, say what is unclear and ask one "
                        "specific clarifying question instead of guessing.\n"
                        "6. NEVER show SQL, database errors, stack traces or internal tool names.\n"
                        "7. FORMAT: plain text only, but structured. Use short section labels such as 'मुख्य तथ्य', "
                        "'विश्लेषण', 'अगला कदम' for Hindi or 'Key facts', 'Analysis', 'Next step' for English. "
                        "Use hyphen bullets where useful. No markdown tables, asterisks, backticks or emojis."                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Candidate question:\n{state.question}\n\n"
                        f"Evidence:\n{evidence}\n\n"
                        "Write the final answer for the candidate in plain text."
                    ),
                },
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            print(f"[booth] synthesis LLM failed: {text[:200]}")
            body = "\n\n".join(r.answer for r in ok)
            note = (
                "\n\n(I couldn't write a full summary just now, so these are the "
                "raw results.)"
            )
            if failed:
                note += " " + self._failure_message(failed).split("\n\n")[0]
            return body + note
        return text

    @staticmethod
    def _voice_language(question: str, hint: Optional[str] = None) -> str:
        """Map the user's visible script to a Sarvam TTS locale when possible."""
        q = question or ""
        if any("\u0900" <= ch <= "\u097f" for ch in q):
            return "hi-IN"
        if any("\u0980" <= ch <= "\u09ff" for ch in q):
            return "bn-IN"
        if any("\u0b80" <= ch <= "\u0bff" for ch in q):
            return "ta-IN"
        if any("\u0c00" <= ch <= "\u0c7f" for ch in q):
            return "te-IN"
        if any("\u0c80" <= ch <= "\u0cff" for ch in q):
            return "kn-IN"
        if any("\u0d00" <= ch <= "\u0d7f" for ch in q):
            return "ml-IN"
        if any("\u0a80" <= ch <= "\u0aff" for ch in q):
            return "gu-IN"
        if any("\u0a00" <= ch <= "\u0a7f" for ch in q):
            return "pa-IN"
        if any("\u0b00" <= ch <= "\u0b7f" for ch in q):
            return "od-IN"
        return hint or "en-IN"

    def voice_brief(
        self,
        question: str,
        written_answer: str,
        language_hint: Optional[str] = None,
    ) -> tuple[str, str]:
        """Create a short spoken consultant brief instead of reading the screen aloud."""
        detected_locale = self._voice_language(question, language_hint)
        lang_hint = detected_locale
        prompt = f"""
User question:
{question}

Written answer shown on screen:
{written_answer}

Create the spoken response Arjun should give after the written answer has already been displayed.
The spoken response must NOT repeat or paraphrase the written answer line by line.
Instead, behave like a consultant speaking naturally:
- briefly acknowledge what the result means;
- point out one or two useful patterns, implications, caveats, or comparisons directly supported by the written answer;
- if the result is surprising or incomplete, explain what should be checked next;
- if the question is ambiguous or the evidence is insufficient, ask one concise, specific clarification instead of guessing;
- do not introduce any fact or number absent from the written answer;
- keep it conversational and concise, normally 2-5 sentences;
- speak in the same language as the user's question. A language hint may be {lang_hint}, but the question's language takes priority.
Return ONLY the spoken text. No heading, bullets, markdown, labels, or meta-commentary.
""".strip()
        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Arjun, a concise and thoughtful political data consultant. "
                        "The screen already contains the detailed factual answer. Your voice response "
                        "adds interpretation and guidance; it must never be a duplicate reading of the screen. "
                        "Use only the supplied answer and question as evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        if text.startswith("<<LLM_ERROR"):
            return (
                "I’ve given you the detailed result on screen. If you want, I can also break down "
                "what this result means and what we should examine next.",
                detected_locale,
            )
        return to_plain_text(text).strip(), detected_locale

    def run(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Please ask a question about voters or election results."

        state = BoothState(question=question)
        print(f"\n[booth] question: {question}")

        for step in range(1, self.max_steps + 1):
            print(f"\n[booth] step {step}/{self.max_steps}")
            plan = self.plan(state)
            thought = plan["thought"]
            action = plan["action"]
            inp = plan["action_input"]
            print(f"  thought: {thought[:220]}")
            print(f"  action:  {action}")
            print(f"  input:   {inp[:160]}")

            if action == "finish":
                state.steps.append(ReactStep(step, thought, "finish", inp, "(finish)"))
                break

            result = self.call_tool(action, inp)
            state.results.append(result)
            obs = (
                result.answer
                if result.ok
                else f"ERROR: {result.error}. {result.detail}".strip()
            )
            print(f"  tool_ok: {result.ok} ({len(obs)} chars)")
            state.steps.append(ReactStep(step, thought, action, inp, obs))

        print("[booth] synthesising…")
        return to_plain_text(self.synthesise(state))


def main() -> None:
    print("=" * 72)
    print(f"  {domain.CONSTITUENCY['product_name']}")
    print(f"  {domain.CONSTITUENCY['product_tagline']}")
    print("=" * 72)
    print(f"  Constituency : {domain.CONSTITUENCY['name']} Vidhan Sabha")
    print(f"  District     : {domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}")
    print(f"  Voters DB    : {domain.VOTERS_DB}")
    print(f"  History DB   : {domain.HISTORY_DB}")
    client = GeminiClient()
    print(f"  LLM          : Gemini / {client.model}")
    print(f"  API key set  : {'yes' if client.api_key else 'NO — export GEMINI_API_KEY'}")
    print("=" * 72)
    print("Type 'exit' to quit.\n")

    agent = BoothAgent(llm=client)

    if len(sys.argv) > 1:
        print(agent.run(" ".join(sys.argv[1:])))
        return

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        try:
            print("\n" + agent.run(q) + "\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()