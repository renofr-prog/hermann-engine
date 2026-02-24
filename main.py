import os
import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
app = FastAPI(title="HERMANN v1", version="1.1.0")


# =========================
# ENUMS
# =========================

Momentum = Literal["LOW", "MED", "HIGH"]
Risk = Literal["LOW", "MED", "HIGH", "UNKNOWN"]
Control = Literal["LOW", "MED", "HIGH", "UNKNOWN"]
BudgetType = Literal["FINANCIAL", "POLITICAL", "STALL", "UNKNOWN"]
Posture = Literal["ENGAGED", "NEUTRAL", "RESISTANT", "AVOIDING"]
PrimaryMove = Literal["PUSH", "CLARIFY", "PAUSE", "DISENGAGE"]
RecommendedChannel = Literal["CALL", "EMAIL", "BOTH"]

DealMaturity = Literal[1, 2, 3, 4, 5]
SponsorStrength = Literal["WEAK", "MED", "STRONG"]
PowerBalance = Literal["SELLER_UP", "EVEN", "BUYER_UP"]
UrgencyDecay = Literal["LOW", "MED", "HIGH", "UNKNOWN"]


# =========================
# SCHEMA
# =========================

class KeySignal(BaseModel):
    quote: str
    meaning: str


class Analysis(BaseModel):
    recontextualisation: str = ""
    momentum: Momentum
    risk: Risk
    control: Control
    budget_type: BudgetType
    posture: Posture
    key_signals: List[KeySignal] = Field(default_factory=list)

    deal_maturity: DealMaturity = 3
    sponsor_strength: SponsorStrength = "MED"
    power_balance: PowerBalance = "EVEN"
    urgency_decay: UrgencyDecay = "MED"
    hidden_risks: List[str] = Field(default_factory=list)
    political_risk: str = ""
    info_gaps: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    primary_move: PrimaryMove
    recommended_channel: RecommendedChannel
    reasoning_summary: str = ""
    what_success_looks_like: str = ""

    scenario_a: str = ""
    scenario_b: str = ""
    failure_path: str = ""


class CallPlan(BaseModel):
    opening: str = ""
    objectives: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    pushbacks: List[str] = Field(default_factory=list)


class EmailPlan(BaseModel):
    angle: str = ""
    cta: str = ""


class EmailDraft(BaseModel):
    subject: str = ""
    body: str = ""
    cta: str = ""


class Execution(BaseModel):
    call_plan: CallPlan
    email_short: EmailPlan
    email_standard: EmailPlan
    email_option_a: EmailDraft = Field(default_factory=EmailDraft)
    email_option_b: EmailDraft = Field(default_factory=EmailDraft)


class HermannStrategyJSON(BaseModel):
    analysis: Analysis
    decision: Decision
    execution: Execution


# =========================
# API PAYLOAD
# =========================

class AnalyzeRequest(BaseModel):
    raw_input: str
    lang: str = "fr"
    attachment_text: Optional[str] = None  # texte extrait du document joint


class AnalyzeResponse(BaseModel):
    strategy: HermannStrategyJSON
    output: str


# =========================
# PROMPTS
# =========================

SYSTEM_STRATEGY = """
You are HERMANN, a partner-level commercial decision engine.

The user message begins with LANG=fr|en|es|de.

Rules:
- Write everything (including email drafts) in LANG.
- Cold, factual, structured.
- Never invent facts.
- If context is missing, list info_gaps.
- Always provide CALL plan.
- Always provide two alternative emails (Option A / B).
- Keep email_short and email_standard populated.

Return ONLY valid JSON matching the schema.
""".strip()


SYSTEM_COMPOSER = """
You are HERMANN in senior delivery mode.

The input begins with LANG=fr|en|es|de.

Write the entire output in LANG.

Structure:
1) Decision
2) Why
3) Pulse
4) Diagnostic
5) Call plan
6) Alternative emails (Option A / B copy-paste)

No JSON in final output.
Tone: senior, direct, concise.
""".strip()


# =========================
# OPENAI HELPERS
# =========================

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
    return json.loads(content)


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


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"ok": True, "service": "hermann-engine"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):

    lang = (req.lang or "fr").lower()
    if lang not in ["fr", "en", "es", "de"]:
        lang = "fr"

    combined_input = req.raw_input

    # If document attached → inject into analysis context
    if req.attachment_text:
        combined_input += "\n\nDOCUMENT ATTACHED:\n" + req.attachment_text

    strategy_raw = openai_json_only(
        model=MODEL_STRATEGY,
        system=SYSTEM_STRATEGY,
        user=f"LANG={lang}\n\n{combined_input}",
    )

    try:
        strategy = HermannStrategyJSON.model_validate(strategy_raw)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Strategy JSON failed schema validation",
                "validation": str(e),
                "raw": strategy_raw,
            },
        )

    output = openai_text(
        model=MODEL_COMPOSER,
        system=SYSTEM_COMPOSER,
        user=f"LANG={lang}\n\n" + json.dumps(strategy.model_dump(), ensure_ascii=False, indent=2),
    )

    return {"strategy": strategy, "output": output}


@app.post("/api/decision", response_model=AnalyzeResponse)
def api_decision(req: AnalyzeRequest):
    return analyze(req)
