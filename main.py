import os
import json
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Load local .env if present (does NOT override Render env vars by default)
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
app = FastAPI(title="HERMANN v1", version="1.0.0")


# -------------------------
# Definitive JSON schema (V1 + V2 extensions, backward compatible)
# -------------------------
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

Lang = Literal["fr", "en", "es", "de"]


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


# -------------------------
# API payloads
# -------------------------
class AnalyzeRequest(BaseModel):
    raw_input: str
    lang: Lang = "fr"


class AnalyzeResponse(BaseModel):
    strategy: HermannStrategyJSON
    output: str


def build_system_strategy(lang: str) -> str:
    return f"""
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
- Email drafts (option_a/option_b + email_short/email_standard) MUST be written in lang="{lang}".

Return ONLY valid JSON matching this schema:
{{
  "analysis": {{
    "recontextualisation": "",
    "momentum": "LOW|MED|HIGH",
    "risk": "LOW|MED|HIGH|UNKNOWN",
    "control": "LOW|MED|HIGH|UNKNOWN",
    "budget_type": "FINANCIAL|POLITICAL|STALL|UNKNOWN",
    "posture": "ENGAGED|NEUTRAL|RESISTANT|AVOIDING",
    "key_signals": [{{"quote": "...", "meaning": "..."}}],
    "deal_maturity": 1|2|3|4|5,
    "sponsor_strength": "WEAK|MED|STRONG",
    "power_balance": "SELLER_UP|EVEN|BUYER_UP",
    "urgency_decay": "LOW|MED|HIGH|UNKNOWN",
    "hidden_risks": ["..."],
    "political_risk": "",
    "info_gaps": ["..."]
  }},
  "decision": {{
    "primary_move": "PUSH|CLARIFY|PAUSE|DISENGAGE",
    "recommended_channel": "CALL|EMAIL|BOTH",
    "reasoning_summary": "",
    "what_success_looks_like": "",
    "scenario_a": "",
    "scenario_b": "",
    "failure_path": ""
  }},
  "execution": {{
    "call_plan": {{
      "opening": "",
      "objectives": [],
      "questions": [],
      "pushbacks": []
    }},
    "email_short": {{"angle": "", "cta": ""}},
    "email_standard": {{"angle": "", "cta": ""}},
    "email_option_a": {{"subject": "", "body": "", "cta": ""}},
    "email_option_b": {{"subject": "", "body": "", "cta": ""}}
  }}
}}

Guidance:
- deal_maturity (1–5): 1=Discovery, 2=Fit confirmed, 3=Proposal, 4=Internal buy-in, 5=Closing.
- sponsor_strength: WEAK if unclear/absent, STRONG if clearly driving internally.
- power_balance: BUYER_UP if they have alternatives/time, SELLER_UP if you have leverage, else EVEN.
- urgency_decay: HIGH if timing window is closing or delay kills probability.
- scenario_a / scenario_b: two strategic paths (e.g. recover via call vs final email + exit line).
- failure_path: what to do if call fails / they keep stalling.

ABSOLUTE REQUIREMENT:
Return ONE object with top-level keys: analysis, decision, execution. Never return partial objects.
""".strip()


def build_system_composer(lang: str) -> str:
    """
    Premium final output:
    - No JSON
    - No code fences (no ```), to avoid UI "text" artifacts and improve readability
    - Adds perceived value blocks: Strategic Summary + Do Not + Next step
    - Ends with an explicit loop (add context / iterate / coach)
    """
    if lang == "en":
        return """
You are HERMANN in senior delivery mode.

Transform the structured strategy into a clear, decisive, premium output.
No JSON. No code blocks. No markdown fences (no ```).

Hard requirements:
- Start with a short "Strategic Summary" (2–4 lines): gravity + objective + time sensitivity.
- Add a "Do Not" section with exactly 3 bullets (common mistakes to avoid).
- End with a "Next step" line that explicitly asks the user to add missing context OR switch to Coach mode.

Structure:
1) Strategic Summary
2) Decision
   - Move
   - Channel
   - Why (1–2 lines)
   - What success looks like (1 line)
3) Pulse
   - Momentum / Risk / Control
4) Partner diagnosis
   - Deal maturity / Posture / Sponsor / Power / Urgency / Budget type
   - Hidden risks (bullets)
   - Political risk (1 line if any)
5) Call plan (script)
   - Opening
   - Objectives (bullets)
   - Questions (bullets)
   - Pushbacks (bullets)
   - Control close: propose 2 concrete time slots and ask them to pick one
6) Emails (copy/paste, editable)
   - Email Short: Angle + CTA
   - Email Standard: Angle + CTA
   - Option A: Subject / Body / CTA
   - Option B: Subject / Body / CTA
7) Questions to lock (if info_gaps exist)
8) Do Not (exactly 3 bullets)
9) Next step question (ask to add context or switch to Coach)

Tone:
- Senior colleague, direct, no fluff.
- Never invent facts.
""".strip()

    if lang == "es":
        return """
Eres HERMANN en modo entrega senior.

Convierte la estrategia estructurada en una salida clara, decisiva y premium.
Sin JSON. Sin bloques de código. Sin markdown fences (no ```).

Requisitos:
- Empieza con "Resumen estratégico" (2–4 líneas): gravedad + objetivo + urgencia.
- Añade "No hacer" con exactamente 3 bullets.
- Termina con una línea "Siguiente paso" que pida añadir contexto o pasar a modo Coach.

Estructura:
1) Resumen estratégico
2) Decisión (Move / Canal / Por qué / Éxito)
3) Pulso (Momentum / Risk / Control)
4) Diagnóstico Partner (madurez / postura / sponsor / poder / urgencia / presupuesto)
5) Plan de llamada (script) + cierre de control (2 horarios concretos)
6) Emails listos para copiar/pegar (Short, Standard, Opción A, Opción B)
7) Preguntas a cerrar (si hay info_gaps)
8) No hacer (3 bullets exactos)
9) Siguiente paso (pregunta)

Tono:
- Colega senior, directo, sin relleno.
- Nunca inventes hechos.
""".strip()

    if lang == "de":
        return """
Du bist HERMANN im Senior-Delivery-Modus.

Wandle die strukturierte Strategie in eine klare, entschlossene, premium Ausgabe um.
Kein JSON. Keine Code-Blöcke. Keine Markdown-Fences (kein ```).

Anforderungen:
- Starte mit "Strategische Zusammenfassung" (2–4 Zeilen): Schweregrad + Ziel + Dringlichkeit.
- Füge "Nicht tun" mit genau 3 Bullets hinzu.
- Beende mit "Nächster Schritt" (Frage): mehr Kontext hinzufügen oder in Coach-Modus wechseln.

Struktur:
1) Strategische Zusammenfassung
2) Entscheidung (Move / Kanal / Warum / Erfolg)
3) Pulse (Momentum / Risk / Control)
4) Partner-Diagnose (Reife / Posture / Sponsor / Power / Urgency / Budget)
5) Call-Plan (Script) + Control Close (2 konkrete Zeitfenster)
6) Emails copy/paste (Short, Standard, Option A, Option B)
7) Offene Fragen (wenn info_gaps existieren)
8) Nicht tun (genau 3 Bullets)
9) Nächster Schritt (Frage)

Ton:
- Senior-Kollege, direkt, kein Blabla.
- Niemals Fakten erfinden.
""".strip()

    # fr default
    return """
Tu es HERMANN en mode “delivery senior”.

Transforme la stratégie structurée en un rendu clair, tranché, premium.
Zéro JSON. Zéro bloc de code. Zéro markdown fence (pas de ```).

Exigences non négociables :
- Commencer par un "Résumé stratégique" (2–4 lignes) : gravité + objectif + urgence.
- Ajouter une section "À ne pas faire" avec exactement 3 bullets.
- Terminer par une question "Prochaine étape" qui demande explicitement d’ajouter du contexte OU de basculer en mode Coach.

Structure :
1) Résumé stratégique
2) Décision
   - Move
   - Canal
   - Pourquoi (1–2 lignes)
   - Objectif (what success looks like) (1 ligne)
3) Impulsion
   - Momentum / Risk / Control
4) Diagnostic Partner
   - Deal maturity / Posture / Sponsor / Power / Urgency / Budget type
   - Hidden risks (bullets)
   - Political risk (1 ligne si présent)
5) Plan d’appel (script)
   - Opening
   - Objectives (bullets)
   - Questions (bullets)
   - Pushbacks (bullets)
   - Control close : proposer 2 créneaux concrets et demander lequel convient
6) Emails (copier/coller, éditables)
   - Email Short : Angle + CTA
   - Email Standard : Angle + CTA
   - Option A : Objet / Corps / CTA
   - Option B : Objet / Corps / CTA
7) Questions à verrouiller (si info_gaps existe)
8) À ne pas faire (exactement 3 bullets)
9) Prochaine étape (question)

Ton :
- Collègue senior, direct, sans blabla.
- Ne jamais inventer.
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


def _normalize_key_signals(value: Any) -> List[Dict[str, str]]:
    """
    Ensures key_signals is always a list of {quote:str, meaning:str}.
    Prevents Pydantic crashes when model returns weird shapes.
    """
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            quote = item.get("quote", "")
            meaning = item.get("meaning", "")
            if not isinstance(quote, str):
                quote = str(quote)
            if not isinstance(meaning, str):
                meaning = str(meaning)
            if quote or meaning:
                out.append({"quote": quote, "meaning": meaning})
        elif isinstance(item, str):
            out.append({"quote": item, "meaning": ""})
        else:
            out.append({"quote": str(item), "meaning": ""})
    return out


def _normalize_email_draft(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        subject = value.get("subject", "")
        body = value.get("body", "")
        cta = value.get("cta", "")
        return {
            "subject": subject if isinstance(subject, str) else str(subject),
            "body": body if isinstance(body, str) else str(body),
            "cta": cta if isinstance(cta, str) else str(cta),
        }
    return {"subject": "", "body": "", "cta": ""}


def repair_strategy(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    HARDENING: sometimes the model returns partial objects (e.g. only info_gaps/call_plan/email_*).
    We normalize it into the required {analysis, decision, execution} shape with safe defaults.
    This avoids 422 "missing fields" regressions.
    """
    if not isinstance(raw, dict):
        return {
            "analysis": {
                "recontextualisation": "",
                "momentum": "LOW",
                "risk": "UNKNOWN",
                "control": "UNKNOWN",
                "budget_type": "UNKNOWN",
                "posture": "NEUTRAL",
                "key_signals": [],
                "deal_maturity": 1,
                "sponsor_strength": "WEAK",
                "power_balance": "EVEN",
                "urgency_decay": "UNKNOWN",
                "hidden_risks": [],
                "political_risk": "",
                "info_gaps": ["Invalid model output (non-object)"],
            },
            "decision": {
                "primary_move": "CLARIFY",
                "recommended_channel": "BOTH",
                "reasoning_summary": "Invalid model output shape; requesting clarification.",
                "what_success_looks_like": "We recover required context and proceed.",
                "scenario_a": "",
                "scenario_b": "",
                "failure_path": "",
            },
            "execution": {
                "call_plan": {"opening": "", "objectives": [], "questions": [], "pushbacks": []},
                "email_short": {"angle": "", "cta": ""},
                "email_standard": {"angle": "", "cta": ""},
                "email_option_a": {"subject": "", "body": "", "cta": ""},
                "email_option_b": {"subject": "", "body": "", "cta": ""},
            },
        }

    if "analysis" in raw and "decision" in raw and "execution" in raw:
        # Normalize nested shapes defensively
        try:
            raw_analysis = raw.get("analysis", {})
            if isinstance(raw_analysis, dict):
                raw_analysis["key_signals"] = _normalize_key_signals(raw_analysis.get("key_signals"))
                raw["analysis"] = raw_analysis
        except Exception:
            pass

        try:
            ex = raw.get("execution", {})
            if isinstance(ex, dict):
                ex["email_option_a"] = _normalize_email_draft(ex.get("email_option_a"))
                ex["email_option_b"] = _normalize_email_draft(ex.get("email_option_b"))
                raw["execution"] = ex
        except Exception:
            pass

        return raw

    # Map common “partial” keys into the right locations
    info_gaps = raw.get("info_gaps") if isinstance(raw.get("info_gaps"), list) else []
    call_plan = raw.get("call_plan") if isinstance(raw.get("call_plan"), dict) else {}
    email_short = raw.get("email_short") if isinstance(raw.get("email_short"), dict) else {}
    email_standard = raw.get("email_standard") if isinstance(raw.get("email_standard"), dict) else {}

    fixed: Dict[str, Any] = {
        "analysis": {
            "recontextualisation": raw.get("recontextualisation", "") if isinstance(raw.get("recontextualisation"), str) else "",
            "momentum": raw.get("momentum", "LOW") if raw.get("momentum") in ["LOW", "MED", "HIGH"] else "LOW",
            "risk": raw.get("risk", "UNKNOWN") if raw.get("risk") in ["LOW", "MED", "HIGH", "UNKNOWN"] else "UNKNOWN",
            "control": raw.get("control", "UNKNOWN") if raw.get("control") in ["LOW", "MED", "HIGH", "UNKNOWN"] else "UNKNOWN",
            "budget_type": raw.get("budget_type", "UNKNOWN") if raw.get("budget_type") in ["FINANCIAL", "POLITICAL", "STALL", "UNKNOWN"] else "UNKNOWN",
            "posture": raw.get("posture", "NEUTRAL") if raw.get("posture") in ["ENGAGED", "NEUTRAL", "RESISTANT", "AVOIDING"] else "NEUTRAL",
            "key_signals": _normalize_key_signals(raw.get("key_signals", [])),
            "deal_maturity": raw.get("deal_maturity", 1) if raw.get("deal_maturity") in [1, 2, 3, 4, 5] else 1,
            "sponsor_strength": raw.get("sponsor_strength", "WEAK") if raw.get("sponsor_strength") in ["WEAK", "MED", "STRONG"] else "WEAK",
            "power_balance": raw.get("power_balance", "EVEN") if raw.get("power_balance") in ["SELLER_UP", "EVEN", "BUYER_UP"] else "EVEN",
            "urgency_decay": raw.get("urgency_decay", "UNKNOWN") if raw.get("urgency_decay") in ["LOW", "MED", "HIGH", "UNKNOWN"] else "UNKNOWN",
            "hidden_risks": raw.get("hidden_risks", []) if isinstance(raw.get("hidden_risks"), list) else [],
            "political_risk": raw.get("political_risk", "") if isinstance(raw.get("political_risk"), str) else "",
            "info_gaps": info_gaps,
        },
        "decision": {
            "primary_move": raw.get("primary_move", "CLARIFY") if raw.get("primary_move") in ["PUSH", "CLARIFY", "PAUSE", "DISENGAGE"] else "CLARIFY",
            "recommended_channel": raw.get("recommended_channel", "BOTH") if raw.get("recommended_channel") in ["CALL", "EMAIL", "BOTH"] else "BOTH",
            "reasoning_summary": raw.get("reasoning_summary", "") if isinstance(raw.get("reasoning_summary"), str) else "",
            "what_success_looks_like": raw.get("what_success_looks_like", "") if isinstance(raw.get("what_success_looks_like"), str) else "",
            "scenario_a": raw.get("scenario_a", "") if isinstance(raw.get("scenario_a"), str) else "",
            "scenario_b": raw.get("scenario_b", "") if isinstance(raw.get("scenario_b"), str) else "",
            "failure_path": raw.get("failure_path", "") if isinstance(raw.get("failure_path"), str) else "",
        },
        "execution": {
            "call_plan": {
                "opening": call_plan.get("opening", "") if isinstance(call_plan.get("opening"), str) else "",
                "objectives": call_plan.get("objectives", []) if isinstance(call_plan.get("objectives"), list) else [],
                "questions": call_plan.get("questions", []) if isinstance(call_plan.get("questions"), list) else [],
                "pushbacks": call_plan.get("pushbacks", []) if isinstance(call_plan.get("pushbacks"), list) else [],
            },
            "email_short": {
                "angle": email_short.get("angle", "") if isinstance(email_short.get("angle"), str) else "",
                "cta": email_short.get("cta", "") if isinstance(email_short.get("cta"), str) else "",
            },
            "email_standard": {
                "angle": email_standard.get("angle", "") if isinstance(email_standard.get("angle"), str) else "",
                "cta": email_standard.get("cta", "") if isinstance(email_standard.get("cta"), str) else "",
            },
            "email_option_a": _normalize_email_draft(raw.get("email_option_a")),
            "email_option_b": _normalize_email_draft(raw.get("email_option_b")),
        },
    }

    return fixed


@app.get("/health")
def health():
    return {"ok": True, "service": "hermann-engine"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    system_strategy = build_system_strategy(req.lang)

    raw = openai_json_only(
        model=MODEL_STRATEGY,
        system=system_strategy,
        user=req.raw_input,
    )

    repaired = repair_strategy(raw)

    try:
        strategy = HermannStrategyJSON.model_validate(repaired)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Strategy JSON failed schema validation",
                "validation": str(e),
                "raw": repaired,
            },
        )

    system_composer = build_system_composer(req.lang)
    output = openai_text(
        model=MODEL_COMPOSER,
        system=system_composer,
        user=json.dumps(strategy.model_dump(), ensure_ascii=False, indent=2),
    )

    return {"strategy": strategy, "output": output}


# Alias endpoint to match proxy expectations (no breaking)
@app.post("/api/decision", response_model=AnalyzeResponse)
def api_decision(req: AnalyzeRequest):
    return analyze(req)
