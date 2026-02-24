import os
import json
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Load local .env if present (does NOT override Render env vars by default)
load_dotenv()

# Accept a few possible key names (Render should be OPENAI_API_KEY)
OPENAI_API_KEY = (
    (os.getenv("OPENAI_API_KEY") or "")
    or (os.getenv("OPENAI_KEY") or "")
    or (os.getenv("OPENAI_API_TOKEN") or "")
).strip()

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in Render > Service > Environment."
    )

MODEL_STRATEGY = os.getenv("MODEL_STRATEGY", "gpt-5-mini")
MODEL_COMPOSER = os.getenv("MODEL_COMPOSER", "gpt-5-mini")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="HERMANN v1", version="1.0.0")


# -------------------------
# Definitive JSON schema (V1 + V2 extensions, backward compatible)
# -------------------------
Momentum = Literal["LOW", "MED", "HIGH"]
Risk = Literal["LOW", "MED", "HIGH", "UNKNOWN"]          # ✅ allow UNKNOWN
Control = Literal["LOW", "MED", "HIGH", "UNKNOWN"]       # ✅ allow UNKNOWN
BudgetType = Literal["FINANCIAL", "POLITICAL", "STALL", "UNKNOWN"]
Posture = Literal["ENGAGED", "NEUTRAL", "RESISTANT", "AVOIDING"]
PrimaryMove = Literal["PUSH", "CLARIFY", "PAUSE", "DISENGAGE"]
RecommendedChannel = Literal["CALL", "EMAIL", "BOTH"]

DealMaturity = Literal[1, 2, 3, 4, 5]
SponsorStrength = Literal["WEAK", "MED", "STRONG"]
PowerBalance = Literal["SELLER_UP", "EVEN", "BUYER_UP"]
UrgencyDecay = Literal["LOW", "MED", "HIGH", "UNKNOWN"]  # ✅ allow UNKNOWN


class KeySignal(BaseModel):
    quote: str
    meaning: str


class Analysis(BaseModel):
    # V1 (keep)
    recontextualisation: str = ""
    momentum: Momentum
    risk: Risk
    control: Control
    budget_type: BudgetType
    posture: Posture
    key_signals: List[KeySignal] = Field(default_factory=list)

    # V2 extensions (add defaults => no breaking)
    deal_maturity: DealMaturity = 3
    sponsor_strength: SponsorStrength = "MED"
    power_balance: PowerBalance = "EVEN"
    urgency_decay: UrgencyDecay = "MED"
    hidden_risks: List[str] = Field(default_factory=list)
    political_risk: str = ""
    info_gaps: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    # V1 (keep)
    primary_move: PrimaryMove
    recommended_channel: RecommendedChannel
    reasoning_summary: str = ""
    what_success_looks_like: str = ""

    # V2 extensions
    scenario_a: str = ""
    scenario_b: str = ""
    failure_path: str = ""


class CallPlan(BaseModel):
    opening: str = ""
    objectives: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    pushbacks: List[str] = Field(default_factory=list)


class EmailPlan(BaseModel):
    # V1 shape (front expects it)
    angle: str = ""
    cta: str = ""


class EmailDraft(BaseModel):
    # V2 copy/paste ready
    subject: str = ""
    body: str = ""
    cta: str = ""


class Execution(BaseModel):
    call_plan: CallPlan

    # V1 (keep)
    email_short: EmailPlan
    email_standard: EmailPlan

    # V2 (add)
    email_option_a: EmailDraft = Field(default_factory=EmailDraft)
    email_option_b: EmailDraft = Field(default_factory=EmailDraft)


class HermannStrategyJSON(BaseModel):
    analysis: Analysis
    decision: Decision
    execution: Execution


# -------------------------
# API payloads
# -------------------------
class AnalyzeRequest(BaseModel):
    raw_input: str
    lang: str = "fr"  # default language, backward compatible


class AnalyzeResponse(BaseModel):
    strategy: HermannStrategyJSON
    output: str


SYSTEM_STRATEGY = """
You are HERMANN, a partner-level commercial decision engine.

You analyze sales or negotiation situations and produce structured JSON ONLY.

Non-negotiable rules:
- Cold, factual, structured.
- Never invent facts.
- If context is missing, list "info_gaps" as the questions we MUST ask.
- Extract 1–3 key quotes if available.
- Always provide a CALL plan.
- Always provide two alternative emails (Option A / Option B) as fallback paths.
- Keep V1 fields populated too ("email_short", "email_standard") for compatibility.

Return ONLY valid JSON matching this schema:
{
  "analysis": {
    "recontextualisation": "",
    "momentum": "LOW|MED|HIGH",
    "risk": "LOW|MED|HIGH|UNKNOWN",
    "control": "LOW|MED|HIGH|UNKNOWN",
    "budget_type": "FINANCIAL|POLITICAL|STALL|UNKNOWN",
    "posture": "ENGAGED|NEUTRAL|RESISTANT|AVOIDING",
    "key_signals": [{"quote": "...", "meaning": "..."}],

    "deal_maturity": 1|2|3|4|5,
    "sponsor_strength": "WEAK|MED|STRONG",
    "power_balance": "SELLER_UP|EVEN|BUYER_UP",
    "urgency_decay": "LOW|MED|HIGH|UNKNOWN",
    "hidden_risks": ["..."],
    "political_risk": "",
    "info_gaps": ["..."]
  },
  "decision": {
    "primary_move": "PUSH|CLARIFY|PAUSE|DISENGAGE",
    "recommended_channel": "CALL|EMAIL|BOTH",
    "reasoning_summary": "",
    "what_success_looks_like": "",

    "scenario_a": "",
    "scenario_b": "",
    "failure_path": ""
  },
  "execution": {
    "call_plan": {
      "opening": "",
      "objectives": [],
      "questions": [],
      "pushbacks": []
    },

    "email_short": {"angle": "", "cta": ""},
    "email_standard": {"angle": "", "cta": ""},

    "email_option_a": {"subject": "", "body": "", "cta": ""},
    "email_option_b": {"subject": "", "body": "", "cta": ""}
  }
}

Guidance:
- deal_maturity (1–5): 1=Discovery, 2=Fit confirmed, 3=Proposal, 4=Internal buy-in, 5=Closing.
- sponsor_strength: WEAK if unclear/absent, STRONG if clearly driving internally.
- power_balance: BUYER_UP if they have alternatives/time, SELLER_UP if you have leverage, else EVEN.
- urgency_decay: HIGH if timing window is closing or delay kills probability.
- scenario_a / scenario_b: two strategic paths (e.g. recover via call vs final email + exit line).
- failure_path: what to do if call fails / they keep stalling.
""".strip()


SYSTEM_COMPOSER = """
You are HERMANN in senior delivery mode.

Transform the structured strategy into a clear, decisive, professional output.

Structure (French):
1) Décision (Move + Canal)
2) Pourquoi (1–2 lignes)
3) Impulsion (Momentum / Risk / Control)
4) Diagnostic Partner (deal maturity, sponsor, power, urgency, hidden risks, info gaps)
5) Plan d'appel (opening + objectives + questions + pushbacks)
6) Emails alternatifs (Option A / Option B) copy/paste:
   - Objet
   - Corps
   - CTA

Rules:
- Tone: senior colleague, direct, no fluff.
- If info_gaps exist, list them as "Questions à verrouiller".
- No JSON in final output.
""".strip()


def openai_json_only(model: str, system: str, user: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned empty content for JSON response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"OpenAI returned non-JSON content: {e}. Content was: {content[:500]}"
        )


def openai_text(model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


@app.get("/health")
def health():
    return {"ok": True, "service": "hermann-engine"}


# ✅ Keep existing endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    raw = openai_json_only(
        model=MODEL_STRATEGY,
        system=SYSTEM_STRATEGY,
        user=req.raw_input,
    )

    try:
        strategy = HermannStrategyJSON.model_validate(raw)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Strategy JSON failed schema validation",
                "validation": str(e),
                "raw": raw,
            },
        )

    output = openai_text(
        model=MODEL_COMPOSER,
        system=SYSTEM_COMPOSER,
        user=json.dumps(strategy.model_dump(), ensure_ascii=False, indent=2),
    )

    return {"strategy": strategy, "output": output}


# ✅ Alias endpoint to match front/proxy expectations (no breaking)
@app.post("/api/decision", response_model=AnalyzeResponse)
def api_decision(req: AnalyzeRequest):
    return analyze(req)
