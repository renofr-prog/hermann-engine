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
# Definitive JSON schema
# -------------------------
Momentum = Literal["LOW", "MED", "HIGH"]
Risk = Literal["LOW", "MED", "HIGH"]
Control = Literal["LOW", "MED", "HIGH"]
BudgetType = Literal["FINANCIAL", "POLITICAL", "STALL", "UNKNOWN"]
Posture = Literal["ENGAGED", "NEUTRAL", "RESISTANT", "AVOIDING"]
PrimaryMove = Literal["PUSH", "CLARIFY", "PAUSE", "DISENGAGE"]
RecommendedChannel = Literal["CALL", "EMAIL", "BOTH"]


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


class Decision(BaseModel):
    primary_move: PrimaryMove
    recommended_channel: RecommendedChannel
    reasoning_summary: str = ""
    what_success_looks_like: str = ""


class CallPlan(BaseModel):
    opening: str = ""
    objectives: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    pushbacks: List[str] = Field(default_factory=list)


class EmailPlan(BaseModel):
    angle: str = ""
    cta: str = ""


class Execution(BaseModel):
    call_plan: CallPlan
    email_short: EmailPlan
    email_standard: EmailPlan


class HermannStrategyJSON(BaseModel):
    analysis: Analysis
    decision: Decision
    execution: Execution


# -------------------------
# API payloads
# -------------------------
class AnalyzeRequest(BaseModel):
    raw_input: str


class AnalyzeResponse(BaseModel):
    strategy: HermannStrategyJSON
    output: str


SYSTEM_STRATEGY = """You are HERMANN, a senior commercial negotiation engine.

You analyze sales or negotiation situations and produce a structured JSON only.

Rules:
Be cold, factual, structured.
Never invent facts.
Extract 2–3 key quotes if available.
Classify budget blockage.
Always provide CALL plan AND email angles.
Decide primary channel based on negotiation rules.
No commentary outside JSON.

Return ONLY valid JSON matching this schema:
{
  "analysis": {
    "recontextualisation": "",
    "momentum": "LOW|MED|HIGH",
    "risk": "LOW|MED|HIGH",
    "control": "LOW|MED|HIGH",
    "budget_type": "FINANCIAL|POLITICAL|STALL|UNKNOWN",
    "posture": "ENGAGED|NEUTRAL|RESISTANT|AVOIDING",
    "key_signals": [{"quote": "...", "meaning": "..."}]
  },
  "decision": {
    "primary_move": "PUSH|CLARIFY|PAUSE|DISENGAGE",
    "recommended_channel": "CALL|EMAIL|BOTH",
    "reasoning_summary": "",
    "what_success_looks_like": ""
  },
  "execution": {
    "call_plan": {
      "opening": "",
      "objectives": [],
      "questions": [],
      "pushbacks": []
    },
    "email_short": {"angle": "", "cta": ""},
    "email_standard": {"angle": "", "cta": ""}
  }
}
"""

SYSTEM_COMPOSER = """You are HERMANN in senior delivery mode.

Transform the structured strategy into a clear, decisive, professional output.

Structure:
Recontextualisation (short)
Momentum / Risk / Control (visible)
Decision (2 lines)
Plan
Call script
Email version A (80–120 words)
Email version B (120–180 words)

Rules:
Tone: senior colleague
Concise
Always propose alternative channel
No JSON in final output
"""


def openai_json_only(model: str, system: str, user: str) -> Dict[str, Any]:
    # Use Chat Completions API and force JSON output
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
        raise RuntimeError(f"OpenAI returned non-JSON content: {e}. Content was: {content[:500]}")


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


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    raw = openai_json_only(
        model=MODEL_STRATEGY,
        system=SYSTEM_STRATEGY,
        user=req.raw_input,
    )

    # Pydantic v2 validation
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
