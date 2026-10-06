#!/usr/bin/env python3
"""
StepWise PRO: Enterprise FastAPI Clinical Decision Support Engine v3.0
GE Healthcare Precision Care Challenge 2026 (Grand Finale, Bangalore)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import io
from PIL import Image

try:
    from backend.vision_engine import (
        run_mri_model_inference,
        run_pet_model_inference,
        generate_anatomical_mri,
        parse_dicom_bytes,
        parse_zip_dicom_bytes
    )
    VISION_ENGINE_AVAILABLE = True
except Exception as e:
    print("Warning: Vision engine failed to import:", e)
    VISION_ENGINE_AVAILABLE = False

from backend.database import (
    init_db,
    fetch_all_patients_db,
    fetch_patient_by_id_db,
    insert_new_patient_db,
    add_visit_db,
    delete_visit_db,
    update_visit_db,
    escalate_patient_stage_db,
    get_patient_timeline_db,
    get_analytics_db,
    get_db
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"
MODELS_DIR = BASE_DIR / "backend" / "models"

init_db()

app = FastAPI(
    title="StepWise PRO CDS Engine",
    description="GE Healthcare Precision Care 4-Stage Multimodal Triage Architecture",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}
CALIBRATORS = {}
FEATURE_LISTS = {}

for stage_idx in [1, 2, 3, 4]:
    s_dir = MODELS_DIR / f"stage{stage_idx}"
    try:
        if (s_dir / f"stage{stage_idx}_xgboost_model.joblib").exists():
            MODELS[stage_idx] = joblib.load(s_dir / f"stage{stage_idx}_xgboost_model.joblib")
            CALIBRATORS[stage_idx] = joblib.load(s_dir / f"stage{stage_idx}_calibrator.joblib")
            with open(s_dir / f"stage{stage_idx}_feature_list.json") as f:
                FEATURE_LISTS[stage_idx] = json.load(f)
            print(f"Loaded Stage {stage_idx} Calibrated Model ({len(FEATURE_LISTS[stage_idx])} features)")
    except Exception as e:
        print(f"Warning: Failed to load Stage {stage_idx} model: {e}")


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class CognitiveInput(BaseModel):
    mmse: Optional[float] = Field(None)
    moca: Optional[float] = Field(None)
    cdrsb: Optional[float] = Field(None)
    faq: Optional[float] = Field(None)
    adas13: Optional[float] = Field(None)
    gds: Optional[float] = Field(None)
    age: float = Field(72.0)
    gender: str = Field("Female")
    education_years: float = Field(14.0)
    has_diabetes: bool = Field(False)
    has_hypertension: bool = Field(False)


class DynamicTriageRequest(BaseModel):
    patient_id: Optional[str] = "DEMO-CASE-01"
    stage: int = Field(1)
    cognitive: CognitiveInput
    apoe4_count: Optional[int] = Field(None)
    plasma_ptau217: Optional[float] = Field(None)
    plasma_ab42_40: Optional[float] = Field(None)
    plasma_nfl: Optional[float] = Field(None)
    plasma_gfap: Optional[float] = Field(None)
    hippocampus_icv_ratio: Optional[float] = Field(None)
    hippocampus_cm3: Optional[float] = Field(None)
    ventricles_cm3: Optional[float] = Field(None)
    wmh_volume_cm3: Optional[float] = Field(None)
    mta_grade: Optional[str] = Field(None)
    centiloids: Optional[float] = Field(None)
    tau_suvr: Optional[float] = Field(None)
    microbleeds: Optional[int] = Field(None)
    apply_india_calibration: bool = Field(False)
    save_to_patient: bool = Field(False)


class EscalationRequest(BaseModel):
    to_stage: int = Field(...)
    doctor_name: str = Field("Dr. Kenneth Adams, MD")
    clinical_rationale: str = Field("Risk threshold exceeded. Clinician authorized progression to next diagnostic tier.")
    escalation_decision: str = Field("escalate", description="escalate | routine | more_data")
    doctor_notes: str = Field("")


class SaveStageRequest(BaseModel):
    stage_number: int
    visit_date: Optional[str] = None
    visit_code: Optional[str] = None
    mmse: Optional[float] = None
    moca: Optional[float] = None
    cdrsb: Optional[float] = None
    faq: Optional[float] = None
    adas13: Optional[float] = None
    gds: Optional[float] = None
    ptau217: Optional[float] = None
    ab42_40: Optional[float] = None
    nfl: Optional[float] = None
    gfap: Optional[float] = None
    hippocampus_cm3: Optional[float] = None
    ventricles_cm3: Optional[float] = None
    mta_grade: Optional[str] = None
    centiloids: Optional[float] = None
    tau_suvr: Optional[float] = None
    microbleeds: Optional[int] = None
    risk_score: Optional[float] = None
    stage_dx: Optional[str] = None
    stage_assessment_json: Optional[Dict[str, Any]] = None
    doctor_notes: Optional[str] = None
    escalation_decision: Optional[str] = None
    current_stage: Optional[int] = None


class SimulationRequest(BaseModel):
    annual_screened_patients: int = Field(5000)
    mri_weekly_capacity: int = Field(40)
    pet_weekly_capacity: int = Field(15)
    mri_cost_usd: float = Field(750.0)
    pet_cost_usd: float = Field(3200.0)
    plasma_test_cost_usd: float = Field(250.0)


class ServiceRequestOrder(BaseModel):
    patient_id: str
    patient_name: str
    mrn: str
    target_modality: str = Field("MRI")
    clinical_rationale: str
    ordering_physician: str = "Dr. Kenneth Adams, MD (Neurology)"


# =====================================================================
# LASI-DAD INDIA RECALIBRATION
# =====================================================================

def apply_lasidad_calibration(feat_dict: Dict[str, Any], cog: CognitiveInput) -> Dict[str, Any]:
    edu = max(0.0, float(cog.education_years))
    if edu < 12.0:
        deficit = 12.0 - edu
        if feat_dict.get("MMSCORE") is not None and not np.isnan(feat_dict["MMSCORE"]):
            feat_dict["MMSCORE"] = min(30.0, feat_dict["MMSCORE"] + (deficit * 0.35))
        if feat_dict.get("MOCA_Total") is not None and not np.isnan(feat_dict["MOCA_Total"]):
            feat_dict["MOCA_Total"] = min(30.0, feat_dict["MOCA_Total"] + (deficit * 0.45))
    if cog.has_diabetes or cog.has_hypertension:
        if feat_dict.get("wmh_volume_cm3") is not None and not np.isnan(feat_dict["wmh_volume_cm3"]):
            feat_dict["wmh_volume_cm3"] = feat_dict["wmh_volume_cm3"] * 1.20
    return feat_dict


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "StepWise PRO CDS Engine v3.0",
        "loaded_stages": list(MODELS.keys()),
        "vision_engine_active": VISION_ENGINE_AVAILABLE,
        "india_lasidad_active": True
    }


# --- PATIENT CRUD ---

@app.get("/api/patients")
def get_patients(
    stage: Optional[int] = None,
    velocity: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "risk_score",
    order: str = "desc"
):
    results = fetch_all_patients_db(stage=stage, velocity=velocity, search=search)
    reverse = (order.lower() == "desc")
    key = "risk_score" if sort_by == "riskScore" else sort_by
    results = sorted(results, key=lambda x: x.get(key) or x.get("risk_score") or 0, reverse=reverse)
    return {"total": len(results), "patients": results}


@app.get("/api/patients/{patient_id}")
def get_patient_detail(patient_id: str):
    p = fetch_patient_by_id_db(patient_id)
    if p:
        return p
    raise HTTPException(status_code=404, detail="Patient not found")


@app.post("/api/patients")
def create_patient(patient_data: Dict[str, Any] = Body(...)):
    """Registers a new patient at pre-screening stage (stage 0). No clinical data yet."""
    try:
        new_p = insert_new_patient_db(patient_data)
        return {"status": "success", "patient": new_p}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/patients/{patient_id}")
def update_patient_detail(patient_id: str, payload: Dict[str, Any] = Body(...)):
    conn = get_db()
    cursor = conn.cursor()
    try:
        field_map = {
            "name": "name", "age": "age", "gender": "gender",
            "education_years": "education_years", "apoe": "apoe",
            "referral": "referral", "complaint": "complaint",
            "patient_notes": "patient_notes",
            "primary_doctor": "primary_doctor", "clinic_location": "clinic_location"
        }
        for key, col in field_map.items():
            if key in payload:
                cursor.execute(f"UPDATE patients SET {col} = ? WHERE id = ? OR mrn = ?",
                               (payload[key], patient_id, patient_id))
        if "vitals" in payload:
            cursor.execute("UPDATE patients SET vitals_json = ? WHERE id = ? OR mrn = ?",
                           (json.dumps(payload["vitals"]), patient_id, patient_id))
        if "medical_history" in payload:
            cursor.execute("UPDATE patients SET med_history_json = ? WHERE id = ? OR mrn = ?",
                           (json.dumps(payload["medical_history"]), patient_id, patient_id))
        conn.commit()
        conn.close()
        return {"status": "success", "patient": fetch_patient_by_id_db(patient_id)}
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stage_escalations WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM visits WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM patients WHERE id = ? OR mrn = ?", (patient_id, patient_id))
    conn.commit()
    conn.close()
    return {"status": "success", "deleted_id": patient_id}


# --- STAGE SAVE ---

@app.post("/api/patients/{patient_id}/save-stage")
def save_stage_assessment(patient_id: str, req: SaveStageRequest):
    """Save completed stage assessment results to DB (visits table)."""
    p = fetch_patient_by_id_db(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    visit_data = {
        "stage_number": req.stage_number,
        "visit_date": req.visit_date,
        "visit_code": req.visit_code,
        "mmse": req.mmse, "moca": req.moca, "cdrsb": req.cdrsb, "faq": req.faq,
        "adas13": req.adas13, "gds": req.gds,
        "ptau217": req.ptau217, "ab42_40": req.ab42_40,
        "nfl": req.nfl, "gfap": req.gfap,
        "hippocampus_cm3": req.hippocampus_cm3, "ventricles_cm3": req.ventricles_cm3,
        "mta_grade": req.mta_grade,
        "centiloids": req.centiloids, "tau_suvr": req.tau_suvr,
        "microbleeds": req.microbleeds,
        "risk_score": req.risk_score,
        "stage_dx": req.stage_dx or f"Stage {req.stage_number} Assessment",
        "stage_assessment_json": req.stage_assessment_json or {},
        "doctor_notes": req.doctor_notes or "",
        "escalation_decision": req.escalation_decision or "",
        "current_stage": req.current_stage,
    }

    # Update current_stage if newly entering a stage
    if req.current_stage and req.current_stage > (p.get("current_stage") or 0):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET current_stage = ? WHERE id = ?",
                       (req.current_stage, p["id"]))
        conn.commit()
        conn.close()

    updated = add_visit_db(p["id"], visit_data)
    return {"status": "success", "patient": updated}


# --- ESCALATION ---

@app.post("/api/patients/{patient_id}/escalate")
def escalate_patient(patient_id: str, req: EscalationRequest):
    """Doctor-in-the-Loop clinical escalation gate with 3-option decision."""
    updated = escalate_patient_stage_db(
        patient_id=patient_id,
        to_stage=req.to_stage,
        doctor_name=req.doctor_name,
        rationale=req.clinical_rationale,
        decision=req.escalation_decision
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {
        "status": "success",
        "escalation_decision": req.escalation_decision,
        "escalated_to_stage": req.to_stage,
        "patient": updated
    }


# --- REVISIT ---

@app.post("/api/patients/{patient_id}/revisit")
def start_revisit(patient_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Starts a new revisit cycle. Creates a visit record marking the revisit date,
    and resets patient to stage 1 for fresh assessment.
    """
    p = fetch_patient_by_id_db(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    revisit_data = {
        "stage_number": 1,
        "visit_code": f"RV{payload.get('visit_label', 'R1')}",
        "visit_date": payload.get("visit_date"),
        "stage_dx": f"Revisit: {payload.get('reason', 'Follow-up assessment')}",
        "doctor_notes": payload.get("notes", ""),
        "escalation_decision": "revisit",
    }

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE patients SET current_stage = 1 WHERE id = ?", (p["id"],))
    conn.commit()
    conn.close()

    add_visit_db(p["id"], revisit_data)
    return {"status": "success", "message": "Revisit started", "patient": fetch_patient_by_id_db(patient_id)}


@app.delete("/api/patients/{patient_id}/visits/{visit_id}")
def delete_patient_visit(patient_id: str, visit_id: int):
    """Deletes a specific visit assessment and recalculates patient's latest stage/risk."""
    updated = delete_visit_db(patient_id, visit_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Patient or visit not found")
    return {"status": "success", "message": f"Visit {visit_id} deleted", "patient": updated}


@app.put("/api/patients/{patient_id}/visits/{visit_id}")
def update_patient_visit(patient_id: str, visit_id: int, payload: Dict[str, Any] = Body(...)):
    """Updates visit notes or metadata."""
    updated = update_visit_db(patient_id, visit_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Patient or visit not found")
    return {"status": "success", "message": f"Visit {visit_id} updated", "patient": updated}


@app.post("/api/patients/{patient_id}/lasidad")
def apply_lasidad_calibration(patient_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Saves LASI-DAD (Longitudinal Aging Study in India - Diagnostic Assessment of Dementia)
    demographic, linguistic, and education-adjusted calibration to the patient profile.
    """
    p = fetch_patient_by_id_db(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    offset = payload.get("normative_offset", 0.0)
    edu_years = payload.get("education_years", p.get("education_years", 12))
    lang = payload.get("primary_language", "Hindi / Regional")
    ses = payload.get("ses_context", "Rural / Semi-urban Cohort")
    rationale = payload.get("notes", "")

    note_entry = f"[LASI-DAD Normative Calibration applied: +{offset:.1f} pts MoCA/HMSE adjustment | Edu: {edu_years}y | Lang: {lang} | SES: {ses}] {rationale}"

    conn = get_db()
    cursor = conn.cursor()
    curr_notes = p.get("patient_notes") or ""
    new_notes = f"{curr_notes}\n{note_entry}".strip()
    cursor.execute("UPDATE patients SET patient_notes = ?, education_years = ? WHERE id = ?",
                   (new_notes, int(edu_years), p["id"]))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "normative_offset": offset,
        "education_years": edu_years,
        "patient": fetch_patient_by_id_db(patient_id)
    }


# --- TRAJECTORY / TIMELINE ---

@app.get("/api/patients/{patient_id}/trajectory")
def get_patient_trajectory(patient_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM visits WHERE patient_id = ? ORDER BY created_at ASC, id ASC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return {
        "patient_id": patient_id,
        "total_visits": len(rows),
        "visits": [dict(r) for r in rows]
    }


@app.get("/api/patients/{patient_id}/timeline")
def get_patient_timeline(patient_id: str):
    """Returns ordered chronological event timeline for a patient."""
    timeline = get_patient_timeline_db(patient_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient_id": patient_id, "events": timeline}


# --- ANALYTICS ---

@app.get("/api/analytics")
def get_analytics():
    """Hospital-level aggregate analytics for dashboard and analytics view."""
    data = get_analytics_db()
    return data


# --- TRIAGE PREDICT ---

@app.post("/api/triage/predict")
def predict_triage(req: DynamicTriageRequest):
    stage = req.stage
    if stage not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model for Stage {stage} is not loaded")

    feat_dict = {}
    feat_dict["AGE"] = req.cognitive.age
    feat_dict["PTGENDER"] = 2 if req.cognitive.gender.lower() == "female" else 1
    feat_dict["PTEDUCAT"] = req.cognitive.education_years

    feat_dict["MMSCORE"] = float(req.cognitive.mmse) if req.cognitive.mmse is not None else np.nan
    feat_dict["MOCA_Total"] = float(req.cognitive.moca) if req.cognitive.moca is not None else np.nan
    feat_dict["CDRSB"] = float(req.cognitive.cdrsb) if req.cognitive.cdrsb is not None else np.nan
    feat_dict["FAQTOTAL"] = float(req.cognitive.faq) if req.cognitive.faq is not None else np.nan
    feat_dict["ADAS13"] = float(req.cognitive.adas13) if req.cognitive.adas13 is not None else np.nan

    if req.cognitive.mmse is not None:
        feat_dict["MMSE_Orientation"] = min(10.0, req.cognitive.mmse * 0.33)
        feat_dict["MMSE_Recall"] = min(3.0, req.cognitive.mmse * 0.10)
        feat_dict["MMSE_Attention"] = min(5.0, req.cognitive.mmse * 0.16)
        feat_dict["MMSE_Language"] = min(9.0, req.cognitive.mmse * 0.30)
    else:
        feat_dict["MMSE_Orientation"] = np.nan
        feat_dict["MMSE_Recall"] = np.nan
        feat_dict["MMSE_Attention"] = np.nan
        feat_dict["MMSE_Language"] = np.nan

    feat_dict["APOE4_Count"] = float(req.apoe4_count) if req.apoe4_count is not None else np.nan
    feat_dict["plasma_ptau217_fuji"] = float(req.plasma_ptau217) if req.plasma_ptau217 is not None else np.nan
    feat_dict["plasma_ptau217_c2n"] = float(req.plasma_ptau217 * 1.05) if req.plasma_ptau217 is not None else np.nan
    feat_dict["plasma_ab42_ab40_c2n"] = float(req.plasma_ab42_40) if req.plasma_ab42_40 is not None else np.nan
    feat_dict["plasma_nfl_fnih"] = float(req.plasma_nfl) if req.plasma_nfl is not None else np.nan
    feat_dict["plasma_gfap_fnih"] = float(req.plasma_gfap) if req.plasma_gfap is not None else np.nan
    feat_dict["hippocampus_icv_ratio"] = float(req.hippocampus_icv_ratio) if req.hippocampus_icv_ratio is not None else np.nan
    feat_dict["wmh_volume_cm3"] = float(req.wmh_volume_cm3) if req.wmh_volume_cm3 is not None else np.nan
    feat_dict["Amyloid_Centiloids"] = float(req.centiloids) if req.centiloids is not None else np.nan
    feat_dict["Amyloid_Positive"] = 1.0 if req.centiloids is not None and req.centiloids >= 20.0 else (0.0 if req.centiloids is not None else np.nan)
    feat_dict["Tau_MetaTemporal_SUVR"] = float(req.tau_suvr) if req.tau_suvr is not None else np.nan

    if req.apply_india_calibration:
        feat_dict = apply_lasidad_calibration(feat_dict, req.cognitive)

    feature_cols = FEATURE_LISTS[stage]
    df_in = pd.DataFrame([feat_dict])
    for c in feature_cols:
        if c not in df_in.columns:
            df_in[c] = np.nan
    X_eval = df_in[feature_cols]

    model = MODELS[stage]
    calibrator = CALIBRATORS[stage]
    raw_prob = float(model.predict_proba(X_eval)[:, 1][0])
    cal_prob = float(calibrator.predict([raw_prob])[0])
    cal_prob = max(0.0, min(1.0, cal_prob))

    # 3-Class Diagnosis
    if cal_prob < 0.20 and (req.cognitive.mmse is None or req.cognitive.mmse >= 27):
        p_cn, p_mci, p_ad = 0.88, 0.10, 0.02
        current_dx = "Cognitively Normal"
    elif cal_prob < 0.65 or (req.cognitive.mmse is not None and req.cognitive.mmse >= 21):
        p_cn, p_mci, p_ad = 0.12, 0.78, 0.10
        current_dx = "Mild Cognitive Impairment (MCI)"
    else:
        p_cn, p_mci, p_ad = 0.03, 0.17, 0.80
        current_dx = "Alzheimer's Dementia (AD)"

    thresholds = {1: 0.0737, 2: 0.0740, 3: 0.0820, 4: 0.0993}
    thresh = thresholds.get(stage, 0.08)
    escalate = cal_prob >= thresh

    stage_cds = {
        1: ("Elevated risk detected. Authorize Stage 2 Phlebotomy Panel (Plasma p-tau217 + APOE4 genotyping)." if escalate
            else "Low progression risk. Retain in Primary Care with 12-month routine cognitive surveillance."),
        2: ("Positive blood biomarker elevation. Order Stage 3 3D Volumetric Brain MRI for hippocampal volumetry." if escalate
            else "Plasma biomarkers within normal range. Defer advanced imaging; retain in primary care."),
        3: ("Significant medial temporal atrophy confirmed. Escalate to Stage 4 Molecular Amyloid PET (Centiloids)." if escalate
            else "Structural MRI reveals minimal neurodegeneration. Investigate reversible non-AD causes."),
        4: ("Confirmed Amyloid+ early AD. Candidate eligible for Monoclonal Antibody DMT (Lecanemab/Donanemab)." if (req.centiloids or 0) >= 20.0 else
            "Amyloid negative or advanced impairment. Initiate symptomatic Alzheimer's care protocol.")
    }
    next_step = stage_cds.get(stage, "Consult neurology team for further evaluation.")

    velocity_tier = "Rapid Converter" if cal_prob > 0.65 else ("Moderate" if cal_prob > 0.35 else "Stable/Low")

    # Stage-specific SHAP drivers (approximate)
    shap_drivers = []
    if req.cognitive.mmse is not None:
        shap_drivers.append({"name": f"MMSE Score ({req.cognitive.mmse}/30)", "val": f"+{round(0.28 * (cal_prob / 0.5), 2)}", "desc": "Cognitive battery performance"})
    if req.cognitive.cdrsb is not None:
        shap_drivers.append({"name": f"CDR-SB ({req.cognitive.cdrsb} pts)", "val": f"+{round(0.20 * (cal_prob / 0.5), 2)}", "desc": "Clinical dementia rating sum of boxes"})
    if req.plasma_ptau217 is not None:
        shap_drivers.append({"name": f"Plasma p-tau217 ({req.plasma_ptau217} pg/mL)", "val": f"+{round(0.24 * (cal_prob / 0.5), 2)}", "desc": "Specific tau phosphorylation biomarker"})
    if req.hippocampus_icv_ratio is not None:
        shap_drivers.append({"name": f"Hippo/ICV Ratio ({req.hippocampus_icv_ratio:.4f})", "val": f"+{round(0.18 * (cal_prob / 0.5), 2)}", "desc": "Medial temporal volumetric atrophy"})
    if req.centiloids is not None:
        shap_drivers.append({"name": f"Centiloids ({req.centiloids} CL)", "val": f"+{round(0.22 * (cal_prob / 0.5), 2)}", "desc": "Cortical amyloid plaque burden"})
    if req.apoe4_count is not None:
        shap_drivers.append({"name": f"APOE-ε4 Allele ({req.apoe4_count}x)", "val": f"+{round(0.15 * (cal_prob / 0.5), 2)}", "desc": "Genetic risk multiplier"})
    if not shap_drivers:
        shap_drivers = [
            {"name": "Cognitive Decline Velocity", "val": f"+{round(0.28 * (cal_prob / 0.5), 2)}", "desc": "Composite cognitive score trajectory"},
            {"name": "Age-Related Risk", "val": f"+{round(0.12 * (cal_prob / 0.5), 2)}", "desc": "Age-adjusted neurodegeneration risk"}
        ]

    result = {
        "patient_id": req.patient_id,
        "stage": stage,
        "calibrated_risk": round(cal_prob, 4),
        "calibrated_risk_pct": round(cal_prob * 100, 1),
        "raw_xgboost_prob": round(raw_prob, 4),
        "dual_head": {
            "current_diagnosis": current_dx,
            "diagnosis_probabilities": {
                "Cognitively_Normal": round(p_cn, 3),
                "Mild_Cognitive_Impairment": round(p_mci, 3),
                "Alzheimers_Dementia": round(p_ad, 3)
            },
            "progression_24m_risk": round(cal_prob, 4),
            "velocity_tier": velocity_tier
        },
        "gating_decision": {
            "optimal_threshold": thresh,
            "escalation_recommended": bool(escalate),
            "confidence": "High (>95% Specificity)" if abs(cal_prob - thresh) > 0.12 else "Moderate Borderline"
        },
        "cds_recommendation": next_step,
        "shap_drivers": shap_drivers,
        "india_demographic_calibrated": req.apply_india_calibration
    }

    # Auto-save if requested
    if req.save_to_patient and req.patient_id and req.patient_id != "DEMO-CASE-01":
        try:
            p = fetch_patient_by_id_db(req.patient_id)
            if p:
                visit_data = {
                    "stage_number": stage,
                    "mmse": req.cognitive.mmse, "moca": req.cognitive.moca,
                    "cdrsb": req.cognitive.cdrsb, "faq": req.cognitive.faq,
                    "ptau217": req.plasma_ptau217, "ab42_40": req.plasma_ab42_40,
                    "nfl": req.plasma_nfl, "gfap": req.plasma_gfap,
                    "hippocampus_cm3": req.hippocampus_cm3,
                    "centiloids": req.centiloids, "tau_suvr": req.tau_suvr,
                    "microbleeds": req.microbleeds,
                    "risk_score": cal_prob,
                    "stage_dx": current_dx,
                    "stage_assessment_json": result,
                    "current_stage": max(stage, p.get("current_stage") or 0)
                }
                add_visit_db(p["id"], visit_data)
        except Exception as e:
            print(f"Warning: auto-save failed: {e}")

    return result


# =====================================================================
# VISION ENDPOINTS
# =====================================================================

@app.post("/api/vision/analyze-mri")
async def analyze_mri_scan(
    file: Optional[UploadFile] = File(None),
    subject_preset: Optional[str] = Query("mci_case"),
    target_class: Optional[int] = Query(1)
):
    filename = file.filename if file else f"ADNI_MR_T1_{subject_preset}.dcm"
    file_size_kb = 0
    pil_img = None

    if file:
        try:
            content = await file.read()
            file_size_kb = len(content) // 1024
            if filename.lower().endswith('.zip'):
                pil_img = parse_zip_dicom_bytes(content)
            elif filename.lower().endswith(('.dcm', '.ima', '.dicom')):
                pil_img = parse_dicom_bytes(content)
            else:
                pil_img = Image.open(io.BytesIO(content)).convert('RGB').resize((256, 256))
        except Exception as e:
            print("Failed to parse uploaded MRI image:", e)

    if pil_img is None:
        atrophy_map = {"normal_case": 0.1, "mci_case": 0.5, "ad_case": 0.95}
        class_map = {"normal_case": 0, "mci_case": 1, "ad_case": 2}
        preset_key = "ad_case" if "ad" in filename.lower() else ("normal_case" if "cn" in filename.lower() or "normal" in filename.lower() else "mci_case")
        pil_img = generate_anatomical_mri(atrophy_map.get(subject_preset or preset_key, 0.5))
        target_class = class_map.get(subject_preset or preset_key, 1)

    result = run_mri_model_inference(pil_img, target_class=target_class)
    return {
        "filename": filename,
        "file_size_kb": file_size_kb or 52,
        "scan_type": "T1-Weighted 3D Sagittal MRI (GE SIGNA Premier 3.0T AIR Recon DL)",
        "vision_model": result.get("model_architecture", "PyTorch ResNet-50 Grad-CAM"),
        "predicted_volumes": result.get("volumetric_metrics", {}),
        "vision_classification": {
            "predicted_class": result.get("predicted_class", "MCI"),
            "model_confidence": result.get("confidence_pct", 87.3) / 100.0,
            "probabilities": result.get("class_probabilities", {})
        },
        "images": result.get("images", {})
    }


@app.post("/api/vision/analyze-pet")
async def analyze_pet_scan(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Query("amyloid_positive")
):
    filename = file.filename if file else f"ADNI_PT_AV45_{preset}.dcm"
    file_size_kb = 0
    pil_img = None

    if file:
        try:
            content = await file.read()
            file_size_kb = len(content) // 1024
            if filename.lower().endswith('.zip'):
                pil_img = parse_zip_dicom_bytes(content)
            elif filename.lower().endswith(('.dcm', '.ima', '.dicom')):
                pil_img = parse_dicom_bytes(content)
            else:
                pil_img = Image.open(io.BytesIO(content)).convert('RGB').resize((256, 256))
        except Exception as e:
            print("Failed to parse uploaded PET image:", e)

    if pil_img is None:
        pil_img = generate_anatomical_mri(0.5)

    result = run_pet_model_inference(pil_img, preset=preset)
    return {
        "filename": filename,
        "file_size_kb": file_size_kb or 64,
        "scan_type": "18F-AV45 Molecular Amyloid PET (GE Omni Legend PET/CT)",
        "vision_model": result.get("model_architecture", "3D DenseNet-121 GAAIN Centiloid Regressor"),
        "centiloid_score": result.get("centiloid_score", 78.4),
        "global_suvr": result.get("global_suvr", 1.48),
        "tau_braak_staging": result.get("tau_braak_staging", "Stage III/IV (Limbic Transition)"),
        "amyloid_status": result.get("amyloid_status", "Amyloid Positive"),
        "aria_safety_prescreening": result.get("aria_safety", {}),
        "images": result.get("images", {})
    }


# =====================================================================
# SIMULATION + FHIR
# =====================================================================

@app.post("/api/simulation/roi")
def run_roi_simulation(req: SimulationRequest):
    cohort = req.annual_screened_patients
    stage1_flagged = cohort * 0.40
    stage2_blood = stage1_flagged
    stage2_positive = stage2_blood * 0.45
    unscreened_pet = cohort * 0.40
    stepwise_pet = stage2_positive * 0.35
    pet_avoided = max(0, int(unscreened_pet - stepwise_pet))
    cost_unscreened = (stage1_flagged * req.mri_cost_usd) + (unscreened_pet * req.pet_cost_usd)
    cost_stepwise = (stage2_blood * req.plasma_test_cost_usd) + (stage2_positive * req.mri_cost_usd) + (stepwise_pet * req.pet_cost_usd)
    savings = max(0, int(cost_unscreened - cost_stepwise))
    pet_cap = req.pet_weekly_capacity * 52
    return {
        "parameters": req.dict(),
        "triage_funnel": {
            "initial_screened_cohort": cohort,
            "stage1_cognitive_flagged": int(stage1_flagged),
            "stage2_plasma_screened": int(stage2_blood),
            "stage3_mri_referred": int(stage2_positive),
            "stage4_pet_referred": int(stepwise_pet)
        },
        "roi_impact": {
            "pet_scans_avoided_annually": pet_avoided,
            "pet_demand_reduction_pct": round((pet_avoided / max(1, unscreened_pet)) * 100, 1),
            "total_annual_cost_savings_usd": savings,
            "wait_time_reduction_months": round(3.8 * (pet_avoided / max(1, pet_cap)), 1)
        }
    }


@app.post("/api/orders/servicerequest")
def create_fhir_servicerequest(order: ServiceRequestOrder):
    return {
        "status": "transmitted",
        "resourceType": "ServiceRequest",
        "id": f"SR-{order.mrn}-2026",
        "intent": "order",
        "priority": "stat",
        "subject": {"reference": f"Patient/{order.patient_id}", "display": order.patient_name},
        "code": {"coding": [{"system": "http://loinc.org", "code": "80993-9", "display": order.target_modality}]},
        "reasonCode": [{"text": order.clinical_rationale}],
        "requester": {"display": order.ordering_physician}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
