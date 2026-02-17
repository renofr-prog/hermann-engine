import os
import json
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # Render will show this in logs if you forgot env var
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

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
    json: HermannStrategyJSON
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
    """
    Forces a JSON object response via Responses API.
    """
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "use

