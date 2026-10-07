# 🧠 StepWise PRO: Precision Dementia Clinical Decision Support System

> **GE Healthcare Precision Care Challenge 2026** — *Grand Finale, Bangalore*  
> **Developed by Team Litchi:** Yatharth Gupta, Avyukt Sisodia, Ashish Kumar, Akhshat Sharma  
> **Platform Version:** 3.0.0 (Enterprise CDS Edition)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.142.2-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2.15-black.svg?logo=next.js&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB7A28.svg)](https://xgboost.readthedocs.io)
[![HL7 FHIR](https://img.shields.io/badge/HL7_FHIR-R4_Compliant-e83e8c.svg)](https://hl7.org/fhir/R4/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [The StepWise 4-Stage Clinical Escalation Paradigm](#-the-stepwise-4-stage-clinical-escalation-paradigm)
3. [System Architecture & Full-Stack Topology](#-system-architecture--full-stack-topology)
4. [End-to-End Data Strategy & Clinical Cohort Harmonization](#-end-to-end-data-strategy--clinical-cohort-harmonization)
5. [AI/ML Progression Modeling & Explainability](#-aiml-progression-modeling--explainability)
6. [Comprehensive Multi-Stage Evaluations & Visual Artifacts](#-comprehensive-multi-stage-evaluations--visual-artifacts)
   - [Stage 1: Primary Care Cognitive & EHR Prioritizer ($M_1$)](#stage-1-primary-care-cognitive--ehr-prioritizer-m_1)
   - [Stage 2: Plasma Biofluid & Genomic Enricher ($M_2$)](#stage-2-plasma-biofluid--genomic-enricher-m_2)
   - [Stage 3: 3D Volumetric MRI Morphometry Engine ($M_3$)](#stage-3-3d-volumetric-mri-morphometry-engine-m_3)
   - [Stage 4: Molecular PET & ARIA Safety Head ($M_4$)](#stage-4-molecular-pet--aria-safety-head-m_4)
   - [Benchmark Summary & Cross-Stage Comparison](#benchmark-summary--cross-stage-comparison)
7. [Deep Neural Vision Engine & Grad-CAM Heatmaps](#-deep-neural-vision-engine--grad-cam-heatmaps)
8. [Indian Clinical Calibration Layer (LASI-DAD HCAP)](#-indian-clinical-calibration-layer-lasi-dad-hcap)
9. [Enterprise Features & Healthcare Interoperability](#-enterprise-features--healthcare-interoperability)
10. [Repository Structure](#-repository-structure)
11. [Quickstart & Installation Guide](#-quickstart--installation-guide)
12. [API Reference & Endpoint Contracts](#-api-reference--endpoint-contracts)
13. [Team & Acknowledgments](#-team--acknowledgments)

---

## 🏥 Executive Summary & Problem Statement

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder affecting over **55 million individuals worldwide**, with over **8.8 million in India aged 60+** (7.4% prevalence; *LASI-DAD, Lee et al., 2023*).

### The Diagnostic Dilemma
In contemporary clinical workflows, millions of patients present to primary care clinics with non-specific, subjective cognitive complaints. Because primary care clinicians lack specialized biomarker tools, patients are routed through an **unstratified, one-size-fits-all diagnostic conveyor belt**:

```
Traditional Pathway (Bottlenecked & Expensive):
[Patient with Concern] ──► [MoCA/MMSE] ──► [Wait 6–18 Mo] ──► [Order MRI/PET for All] ──► [Late Diagnosis]
(Result: Severe scanner backlog, $8,000+ per patient, 70% negative scan rate, missed treatment windows)

StepWise Precision Escalation Pathway:
[Patient with Concern] ──► [Stage 1: Cognitive/EHR] ──► [Stage 2: Blood p-tau217/APOE]
                                  │ Gate 1                     │ Gate 2
                                  ▼                            ▼
                         [Primary Care Monitor]      [12-Month Watchful Recalibration]
                                                               │ (High Risk)
                                                               ▼
[Stage 4: Molecular PET + ARIA] ◄────── Gate 3 ─────── [Stage 3: Volumetric MRI]
```

### The Disease-Modifying Therapy (DMT) Revolution
Novel FDA and EMA-approved monoclonal antibodies (e.g., **Lecanemab / Leqembi**, **Donanemab / Kisunla**) are only clinically effective during the narrow **Mild Cognitive Impairment (MCI) to Mild AD** therapeutic window with confirmed amyloid pathology. If a patient progresses to moderate-to-severe dementia before detection, irreversible synaptic loss has occurred and therapy is contraindicated.

### The StepWise Solution
**StepWise PRO** is an enterprise-grade Clinical Decision Support System (CDSS) that converts unstratified dementia workups into an intelligent **4-stage progressive clinical escalation pipeline**. StepWise provides:
1. **Calibrated 24-Month Progression Risk:** Predicts the velocity of cognitive decline rather than outputting a static label.
2. **Precision Gating ($T_1 \rightarrow T_4$):** Recommends advanced diagnostics (blood biomarkers, volumetric MRI, amyloid/tau PET) only when mathematically justified.
3. **Safety Profiling for Anti-Amyloid DMTs:** Automated Amyloid-Related Imaging Abnormalities (**ARIA-E** edema and **ARIA-H** microhemorrhage) risk stratification based on APOE4 status, age, and baseline microbleeds.
4. **Health-Economic & Scanner Capacity ROI:** Dynamic simulation engine demonstrating scanner load reduction, early capture rates, and hospital cost savings.
5. **HL7 FHIR R4 & CDS Hooks Integration:** 1-click diagnostic order generation (`ServiceRequest` writeback).

---

## 🔄 The StepWise 4-Stage Clinical Escalation Paradigm

```mermaid
flowchart TD
    subgraph S1["Stage 1: Primary Care Screening ($15 - $50)"]
        A[Patient Intake] --> B[Demographics + Vitals + Comorbidities]
        B --> C[MoCA / MMSE / ADAS-Cog-13]
        C --> D[Calibrated XGBoost Model 1]
        D --> E{Progression Risk >= T1 (0.0737)?}
    end

    E -->|No| F[Primary Care Watchful Recalibration]
    E -->|Yes| G[Escalate to Stage 2: Phlebotomy]

    subgraph S2["Stage 2: Plasma Biofluid Biomarkers ($150 - $400)"]
        G --> H[Plasma %p-tau217 + Abeta42/40 + GFAP + NfL]
        H --> I[Genomics: APOE4 Allele Count]
        I --> J[Cumulative Stacking Model 2 (7-Assay Harmonized)]
        J --> K{Progression Risk >= T2 (0.1982)?}
    end

    K -->|No| L[12-Month Serial Plasma Biomarker Follow-up]
    K -->|Yes| M[Escalate to Stage 3: Structural Neuroimaging]

    subgraph S3["Stage 3: 3D Volumetric MRI Morphometry ($600 - $1,200)"]
        M --> N[FreeSurfer / PyTorch ResNet-18 Vision Head]
        N --> O[Hippocampal Volume + Ventricles + HVR + WMH]
        O --> P[Cumulative Stacking Model 3]
        P --> Q{Progression Risk >= T3 (0.3415)?}
    end

    Q -->|No| R[Vascular Intervention & Non-AD Differential Workup]
    Q -->|Yes| S[Escalate to Stage 4: Molecular Neuroimaging]

    subgraph S4["Stage 4: Molecular PET & DMT Safety Head ($3,000 - $5,000)"]
        S --> T[18F-Florbetapir / Florbetaben Amyloid PET]
        T --> U[Centiloid Scale SUVR + Braak Tau Staging]
        U --> V[ARIA-E / ARIA-H Risk Composite Engine]
        V --> W[DMT Authorization Badge: Lecanemab/Donanemab Ready]
    end
```

| Stage | Modality & Clinical Domain | Key Biomarkers & Features | Cost Range | Goal |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Primary Care Cognitive & EHR | Age, MoCA, MMSE, ADAS-Cog, Pulse Pressure, Diabetes | \$15 – \$50 | Screen large population; eliminate 80% primary care review backlog |
| **Stage 2** | Biofluid & Genomic Biomarkers | Plasma %p-tau217, $A\beta_{42/40}$, GFAP, NfL, APOE4 | \$150 – \$400 | Biological confirmation of amyloid pathology & neuroinflammation |
| **Stage 3** | 3D Volumetric MRI Morphometry | Hippocampal Volume ($\text{cm}^3$), Ventricles, HVR, WMH | \$600 – \$1,200 | Quantify structural neurodegeneration & vascular leukoaraiosis |
| **Stage 4** | Molecular PET & ARIA Safety | Amyloid Centiloid ($0-100\text{ CL}$), Tau SUVR, ARIA-E/H Index | \$3,000 – \$5,000 | Authorize anti-amyloid therapy & generate safe infusion dosage protocol |

---

## 🏛 System Architecture & Full-Stack Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          STEPWISE PRO SYSTEM ARCHITECTURE                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                         CLINICAL FRONTEND (WEB UI)                           │
    │  Next.js 14 (App Router) • React 18 • Tailwind CSS • Lucide Icons • Framer   │
    │  ┌───────────────────────┬─────────────────────────┬──────────────────────┐  │
    │  │ Triage Command Queue  │ 4-Stage Patient Journey │ TreeSHAP Waterfall   │  │
    │  ├───────────────────────┼─────────────────────────┼──────────────────────┤  │
    │  │ ARIA Safety Profiler  │ 1-Click Order Dispatch  │ Hospital ROI Sandbox │  │
    │  ├───────────────────────┴─────────────────────────┴──────────────────────┤  │
    │  │ Vision AI Studio (ResNet-18 Grad-CAM & Anatomical MRI/PET Synthesizer) │  │
    │  └────────────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────┬───────────────────────────────────────┘
                                           │ HTTPS / REST / JSON (Port 3000 -> 8000)
                                           ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                         FASTAPI CDS BACKEND ENGINE                           │
    │  ┌────────────────────────────────────────────────────────────────────────┐  │
    │  │ Asynchronous REST API Engine (`backend/app.py`)                        │  │
    │  │  • `/api/triage/predict` — Multi-stage XGBoost progression inference   │  │
    │  │  • `/api/vision/analyze-mri` — PyTorch ResNet-18 scan-to-volume engine │  │
    │  │  • `/api/vision/analyze-pet` — TorchScript PET Centiloid analyzer      │  │
    │  │  • `/api/orders/servicerequest` — FHIR R4 1-Click order dispatcher     │  │
    │  │  • `/api/simulation/roi` — Health-economic & scanner capacity engine   │  │
    │  ├────────────────────────────────────────────────────────────────────────┤  │
    │  │ AI / ML Core (`backend/models/`)                                       │  │
    │  │  • Calibrated XGBoost Classifiers (Stages 1–4, 50 to 96 features)      │  │
    │  │  • Polynomial-Time TreeSHAP Clinical Explainer                         │  │
    │  │  • Isotonic Out-of-Fold Probability Calibrator                         │  │
    │  │  • India LASI-DAD HCAP Demographic Education & Comorbidity Offset      │  │
    │  ├────────────────────────────────────────────────────────────────────────┤  │
    │  │ Vision Engine (`backend/vision_engine.py`)                             │  │
    │  │  • Multi-Task ResNet-18 (Hippocampus, Ventricles, HVR Ratio)           │  │
    │  │  • Grad-CAM Saliency Heatmap Generator with OpenCV / Matplotlib        │  │
    │  │  • Native DICOM / ZIP Ingestor (`pydicom`)                             │  │
    │  └────────────────────────────────────────────────────────────────────────┘  │
    └───────────────────────┬───────────────────────────────┬──────────────────────┘
                            │                               │
                            ▼                               ▼
    ┌──────────────────────────────────────┐  ┌────────────────────────────────────┐
    │         DATA & PERSISTENCE           │  │      FHIR R4 INTEROPERABILITY      │
    │  • SQLite Multi-Visit Clinical Store │  │  • FHIR R4 Bundle Converter        │
    │  • ADNI / OASIS Harmonized Dataset   │  │  • Patient, Observation, Condition │
    │  • Calibrated Scalers (.joblib)      │  │  • ServiceRequest Writeback Mock   │
    └──────────────────────────────────────┘  └────────────────────────────────────┘
```

---

## 📊 End-to-End Data Strategy & Clinical Cohort Harmonization

### 1. Ingestion & Temporal Splitting
* **9,453 longitudinal clinical visits** ingested across **2,458 unique patients** from ADNI 1/GO/2/3/4 and OASIS cohorts.
* **Temporal Patient-Level Split:** 75% development cohort (earliest enrollment dates) and 25% prospective temporal holdout (615 untouched patients), strictly grouped by Patient ID (`PTID`) to eliminate data leakage.

### 2. Multi-Assay Biofluid Z-Score Harmonization
Stage 2 harmonizes 7 independent clinical assay platforms through per-platform reference z-scoring:
1. `C2N_PRECIVITYAD2`: Plasma %p-tau217, $A\beta_{42/40}$ ratio, APS2 score.
2. `UPENN_PLASMA_FUJIREBIO`: Fujirebio Lumipulse p-tau217, Simoa GFAP, NfL.
3. `BLENNOW_LAB_TAU`: Plasma total-tau and p-tau181.
4. `ROCHE_ELECSYS`: Electrochemiluminescence $A\beta_{42/40}$, pTau181.
5. `JANSSEN_PLASMA`: Dilution-corrected p-tau217.
6. `LILLY_MSD600`: MSD600 electrochemiluminescence p-tau217.
7. `FNIHBC_TRAJECTORIES`: 17-biomarker longitudinal panel pivoted wide.

### 3. Native `NaN` Routing vs. Zero-Imputation
Traditional ML pipelines impute missing values with 0 or mean. In cognitive medicine:
$$\text{Setting Missing MMSE} = 0 \implies \text{Severe Coma / Brain Death}$$
StepWise utilizes XGBoost's native missing value routing. Unperformed tests are ingested as `np.nan`, and the tree routes the patient down learned optimal default split paths.

---

## 🤖 AI/ML Progression Modeling & Explainability

### 1. The 24-Month Rapid Progression Target
* **Flawed Approach (Static Classification):** Guessing the patient's current diagnosis provides zero triage value because symptoms are already known.
* **StepWise Progression Formulation:** Predicts calibrated risk $R_t \in [0, 1]$ that an MCI patient will convert to irreversible Dementia within **24 months**:
  $$\text{Progression}_{24\text{m}} = \mathbb{I}\left(\text{Diagnosis}(t + 24\text{m}) = \text{Dementia}\right)$$

### 2. Monotonic Constraints for Clinical Plausibility
To comply with FDA/CE-MDR clinical decision support guidance, mathematical monotonic constraints are enforced during gradient boosting:
* $\uparrow \text{Plasma } p\text{-tau217} \implies \text{Monotonic Increase in Risk } (+1)$
* $\downarrow \text{MoCA Total Score} \implies \text{Monotonic Increase in Risk } (-1)$
* $\downarrow \text{Hippocampal Volume} \implies \text{Monotonic Increase in Risk } (-1)$
* $\uparrow \text{Centiloid PET Tracer} \implies \text{Monotonic Increase in Risk } (+1)$

### 3. Polynomial-Time TreeSHAP Clinical Explanations
Every prediction decomposes into additive feature contributions in log-odds space:
$$\text{Margin}(x) = \phi_0 + \sum_{j=1}^{d_k} \phi_j(x)$$
The UI transforms these mathematical attributions into an automated natural-language clinical narrative for physicians.

---

## 📈 Comprehensive Multi-Stage Evaluations & Visual Artifacts

### Stage 1: Primary Care Cognitive & EHR Prioritizer ($M_1$)
* **Modality:** Demographics, Vitals, MoCA, MMSE, ADAS-Cog-13, Clinical Comorbidities (50 Features).
* **Cross-Validation ROC-AUC:** **$0.8077 \pm 0.0173$**
* **Holdout Calibration Brier Score:** **$0.0860$**
* **Clinical Triage Threshold ($T_1$):** **$0.0737$** (Sensitivity: 51.4%, Specificity: 83.3%)
* **Queue Lift:** Reviewing the **top 20% of the patient queue captures over 51.4% of all 24-month progressors**, eliminating 80% of unnecessary primary care referrals.

| Evaluation Metric | ROC Curve & PR Curve |
| :---: | :---: |
| <img src="evaluations/stage1/confusion_matrix.png" width="400" alt="Stage 1 Confusion Matrix" /> | <img src="evaluations/stage1/roc_curve.png" width="400" alt="Stage 1 ROC Curve" /> |

| Precision-Recall Curve | Top-K Prioritization Lift |
| :---: | :---: |
| <img src="evaluations/stage1/pr_curve.png" width="400" alt="Stage 1 PR Curve" /> | <img src="evaluations/stage1/topk_prioritization_lift.png" width="400" alt="Stage 1 Top-K Lift" /> |

| SHAP Global Feature Summary | Individual Case SHAP Waterfall |
| :---: | :---: |
| <img src="evaluations/stage1/shap_summary.png" width="400" alt="Stage 1 SHAP Summary" /> | <img src="evaluations/stage1/shap_waterfall_case.png" width="400" alt="Stage 1 SHAP Waterfall" /> |

---

### Stage 2: Plasma Biofluid & Genomic Enricher ($M_2$)
* **Modality:** Stage 1 Vector + Plasma %p-tau217, $A\beta_{42/40}$, GFAP, NfL, APOE4 Allele Count (77 Features).
* **Cross-Validation ROC-AUC:** **$0.8124 \pm 0.0158$**
* **Holdout Calibration Brier Score:** **$0.0861$**
* **Clinical Triage Threshold ($T_2$):** **$0.1982$**
* **Clinical Impact:** Enriches patient cohort with true biological amyloid pathology before ordering \$1,000+ MRI scans.

| Evaluation Metric | ROC Curve & PR Curve |
| :---: | :---: |
| <img src="evaluations/stage2/confusion_matrix.png" width="400" alt="Stage 2 Confusion Matrix" /> | <img src="evaluations/stage2/roc_curve.png" width="400" alt="Stage 2 ROC Curve" /> |

| Precision-Recall Curve | Top-K Prioritization Lift |
| :---: | :---: |
| <img src="evaluations/stage2/pr_curve.png" width="400" alt="Stage 2 PR Curve" /> | <img src="evaluations/stage2/topk_prioritization_lift.png" width="400" alt="Stage 2 Top-K Lift" /> |

| SHAP Global Feature Summary | Individual Case SHAP Waterfall |
| :---: | :---: |
| <img src="evaluations/stage2/shap_summary.png" width="400" alt="Stage 2 SHAP Summary" /> | <img src="evaluations/stage2/shap_waterfall_case.png" width="400" alt="Stage 2 SHAP Waterfall" /> |

---

### Stage 3: 3D Volumetric MRI Morphometry Engine ($M_3$)
* **Modality:** Stage 2 Vector + FreeSurfer / ResNet-18 Hippocampal Volumetry, Ventricles, Entorhinal Volume, White Matter Hyperintensities (85 Features).
* **Cross-Validation ROC-AUC:** **$0.8149 \pm 0.0142$**
* **Holdout ROC-AUC:** **$0.8225$**
* **Holdout Calibration Brier Score:** **$0.0862$**
* **Clinical Triage Threshold ($T_3$):** **$0.3415$**
* **Clinical Impact:** Confirms structural brain atrophy and identifies cerebrovascular leukoaraiosis (WMH) to rule out non-AD mimics and assess baseline ARIA hemorrhage risks.

| Evaluation Metric | ROC Curve & PR Curve |
| :---: | :---: |
| <img src="evaluations/stage3/confusion_matrix.png" width="400" alt="Stage 3 Confusion Matrix" /> | <img src="evaluations/stage3/roc_curve.png" width="400" alt="Stage 3 ROC Curve" /> |

| Precision-Recall Curve | Top-K Prioritization Lift |
| :---: | :---: |
| <img src="evaluations/stage3/pr_curve.png" width="400" alt="Stage 3 PR Curve" /> | <img src="evaluations/stage3/topk_prioritization_lift.png" width="400" alt="Stage 3 Top-K Lift" /> |

| SHAP Global Feature Summary | Individual Case SHAP Waterfall |
| :---: | :---: |
| <img src="evaluations/stage3/shap_summary.png" width="400" alt="Stage 3 SHAP Summary" /> | <img src="evaluations/stage3/shap_waterfall_case.png" width="400" alt="Stage 3 SHAP Waterfall" /> |

---

### Stage 4: Molecular PET & ARIA Safety Head ($M_4$)
* **Modality:** Stage 3 Vector + Amyloid Centiloids ($0-100\text{ CL}$), Braak Tau SUVR, Composite ARIA Risk Score (96 Features).
* **Cross-Validation ROC-AUC:** **$0.8185 \pm 0.0172$**
* **Holdout ROC-AUC:** **$\mathbf{0.8446}$**
* **Holdout Specificity:** **$\mathbf{86.82\%}$**
* **Negative Predictive Value (NPV):** **$\mathbf{97.76\%}$**
* **Calibration Brier Score:** **$\mathbf{0.0493}$** (Exceptional probability calibration)
* **Clinical Impact:** Final gate authorizing disease-modifying monoclonal antibody therapy (Lecanemab/Donanemab) with automated ARIA-E and ARIA-H risk stratification.

| Evaluation Metric | ROC Curve & PR Curve |
| :---: | :---: |
| <img src="evaluations/stage4/confusion_matrix.png" width="400" alt="Stage 4 Confusion Matrix" /> | <img src="evaluations/stage4/roc_curve.png" width="400" alt="Stage 4 ROC Curve" /> |

| Precision-Recall Curve | Top-K Prioritization Lift |
| :---: | :---: |
| <img src="evaluations/stage4/pr_curve.png" width="400" alt="Stage 4 PR Curve" /> | <img src="evaluations/stage4/topk_prioritization_lift.png" width="400" alt="Stage 4 Top-K Lift" /> |

| SHAP Global Feature Summary | Individual Case SHAP Waterfall |
| :---: | :---: |
| <img src="evaluations/stage4/shap_summary.png" width="400" alt="Stage 4 SHAP Summary" /> | <img src="evaluations/stage4/shap_waterfall_case.png" width="400" alt="Stage 4 SHAP Waterfall" /> |

---

### 📊 Benchmark Summary & Cross-Stage Comparison

| Stage | Features | 5-Fold CV ROC-AUC | Holdout ROC-AUC | Brier Loss | Specificity | NPV | Top 20% Queue Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Cognitive/EHR)** | 50 | $0.8077 \pm 0.0173$ | $0.8012$ | $0.0860$ | $83.30\%$ | $95.12\%$ | **$2.57\times$** |
| **Stage 2 (Biofluid/APOE)** | 77 | $0.8124 \pm 0.0158$ | $0.8105$ | $0.0861$ | $84.20\%$ | $96.04\%$ | **$2.68\times$** |
| **Stage 3 (Volumetric MRI)** | 85 | $0.8149 \pm 0.0142$ | $0.8225$ | $0.0862$ | $85.40\%$ | $96.88\%$ | **$2.74\times$** |
| **Stage 4 (PET + ARIA Head)** | 96 | **$0.8185 \pm 0.0172$** | **$\mathbf{0.8446}$** | **$\mathbf{0.0493}$** | **$\mathbf{86.82\%}$** | **$\mathbf{97.76\%}$** | **$\mathbf{2.89\times}$** |

---

## 🔬 Deep Neural Vision Engine & Grad-CAM Heatmaps

For Tier-2 and Tier-3 healthcare centers that do not have \$50,000 FreeSurfer / NeuroQuant neuroimaging analysis pipelines, StepWise includes an embedded **PyTorch Multi-Task Vision Head**:

```
RAW SAGITTAL / AXIAL DICOM / PNG ──► [PyTorch ResNet-18 Backbone] ──► Predicted Hippo Volume (cm³)
                                                     │              Predicted Ventricular Volume (cm³)
                                                     │              Predicted HVR Ratio
                                                     ▼
                                      [Grad-CAM Saliency Engine]
                                                     │
                                                     ▼ (Auto-Populates Stage 3 Vector)
                                      [Calibrated Stage 3 XGBoost Engine]
```

### Vision Engine Capabilities:
1. **Multi-Task Regression:** Extracts Hippocampal Volume ($\text{cm}^3$), Ventricle Volume ($\text{cm}^3$), and Hippocampal-to-Ventricle Ratio ($\text{HVR} = \frac{\text{Hippo}}{\text{Ventricles}}$) directly from 2D/3D slice arrays.
2. **Grad-CAM Saliency Heatmaps:** Visualizes exact convolutional feature activation maps in the medial temporal lobe, entorhinal cortex, and periventricular white matter.
3. **TorchScript PET JIT Engine:** Calibrates 18F-AV45 / PIB tracer scans into standardized Centiloid scale values ($0-100\text{ CL}$).
4. **DICOM & ZIP Ingestion:** Seamless drag-and-drop parsing of clinical `.dcm` files with metadata extraction.

---

## 🇮🇳 Indian Clinical Calibration Layer (LASI-DAD HCAP)

To eliminate false-positive dementia classifications in Indian clinical settings—where **35–40% of rural elderly have $<4$ years of formal schooling**—StepWise implements the **LASI-DAD (Longitudinal Aging Study in India — Diagnostic Assessment of Dementia)** normative adjustment layer:

### 1. Education Reserve Correction:
$$\text{MMSE}_{\text{adjusted}} = \text{MMSE}_{\text{raw}} + \max\left(0, (12 - \text{Education\_Years}) \times 0.35\right)$$
$$\text{MoCA}_{\text{adjusted}} = \text{MoCA}_{\text{raw}} + \max\left(0, (10 - \text{Education\_Years}) \times 0.40\right)$$

### 2. South Asian Cardiovascular Comorbidity Multipliers:
Adjusts vascular weightings based on high South Asian prevalence of early-onset Type-2 Diabetes Mellitus ($+18\%$ microvascular factor) and uncontrolled Hypertension ($+14\%$ white matter hyperintensity scaling).

---

## 💼 Enterprise Features & Healthcare Interoperability

### 1. Prioritized Triage Command Queue
* Dynamically ranks patient rosters by **24-Month Calibrated Progression Risk**.
* Immediate visual indicators for Gating recommendations, current clinical stage, and missing biomarker recommendations.

### 2. ARIA Safety Profiler & DMT Readiness Badge
* Calculates patient-specific **ARIA-E (Vasogenic Edema)** and **ARIA-H (Microhemorrhage)** risk percentages based on APOE4 homozygosity ($\epsilon 4/\epsilon 4$), age, and baseline microbleed count.
* Generates clear FDA/EMA-compliant DMT candidacy authorization badges for Lecanemab and Donanemab.

### 3. Health-Economic & Scanner Capacity ROI Sandbox
* Live interactive parameter simulation for hospital administrators.
* Models **Total Annual Diagnostic Cost Savings**, **Scanner Hours Saved**, and **Treatment Window Capture Rate** under StepWise precision gating vs. traditional unstratified scanning.

### 4. HL7 FHIR R4 & CDS Hooks Compliance
* Standardized FHIR resource mapping for `Patient`, `Observation`, `Condition`, `DiagnosticReport`, and `ServiceRequest`.
* **1-Click Order Writeback:** Generates validated FHIR JSON diagnostic test orders with automated CPT codes (e.g., CPT `81401` for APOE, `70553` for Brain MRI with contrast, `78814` for Amyloid PET).

---

## 📁 Repository Structure

```
GE_healthcare/
├── README.md                                   # Comprehensive System Documentation
├── 01_PROBLEM_AND_CLINICAL_KNOWLEDGE.md         # Clinical Neuropathology & Problem Statement
├── 02_DATA_STRATEGY_AND_COHORTS.md             # Cohort Ingestion & Biomarker Harmonization
├── 03_AI_ML_MODELING_AND_EXPLAINABILITY.md     # XGBoost, TreeSHAP & Calibration Math
├── 04_SYSTEM_ARCHITECTURE_AND_INTERACTION.md   # Architecture, Stack & Interaction Design
├── 05_PROJECT_EXECUTION_AND_BUILD_TRACKER.md   # Build Tracker & Milestone Verification
├── STEPWISE_OVERHAUL_PLAN.md                   # Enterprise Execution Blueprint
├── start_servers.sh                            # One-Click Unified Server Launcher
│
├── backend/                                    # FastAPI REST & ML Inference Backend
│   ├── app.py                                  # Core REST API Routes & CDS Dispatcher
│   ├── database.py                             # SQLite Patient Store & Reseed Seed Data
│   ├── vision_engine.py                        # PyTorch ResNet-18 & Grad-CAM Vision Engine
│   ├── models/                                 # Calibrated Staged ML Model Artifacts (.joblib)
│   │   ├── stage1_calibrated_model.joblib      # Stage 1 XGBoost + Isotonic Calibrator
│   │   ├── stage2_calibrated_model.joblib      # Stage 2 Biofluid XGBoost + Calibrator
│   │   ├── stage3_calibrated_model.joblib      # Stage 3 Volumetric MRI + Calibrator
│   │   └── stage4_calibrated_model.joblib      # Stage 4 Molecular PET + Calibrator
│   └── data/
│       └── stepwise.db                         # Multi-Visit Patient Dossier Database
│
├── frontend/                                   # Next.js 14 Clinical Web Application
│   ├── package.json                            # Next.js & React Dependencies
│   ├── tailwind.config.js                      # Custom Clinical Dark/Light Theme Design System
│   ├── tsconfig.json                           # TypeScript Configuration
│   └── src/                                    # Application Source Code
│       ├── app/                                # Next.js App Router Pages
│       │   ├── page.tsx                        # Main Patient Dossier & Triage Dashboard
│       │   └── layout.tsx                      # Root Layout & Typography
│       ├── components/                         # Modular Clinical UI Components
│       │   ├── PatientHeader.tsx               # Patient Bio, Stage Badge & DMT Readiness
│       │   ├── StageTracker.tsx                # 4-Stage Progressive Journey Tracker
│       │   ├── Stage1CognitiveTab.tsx          # Cognitive Testing & EHR Intake
│       │   ├── Stage2BloodBiomarkersTab.tsx    # Plasma Biofluid Panel (p-tau217, APOE4)
│       │   ├── Stage3MRITab.tsx                # Volumetric MRI & White Matter Hyperintensity
│       │   ├── Stage4PETTab.tsx                # Molecular PET Centiloids & ARIA Profiler
│       │   ├── SHAPWaterfallModal.tsx          # Interactive TreeSHAP Decomposition
│       │   ├── ROIAnalyticsModal.tsx           # Hospital Capacity & Economic Sandbox
│       │   ├── VisionStudioModal.tsx           # Imaging AI & Grad-CAM Heatmap Studio
│       │   └── FHIRDispatcherModal.tsx         # HL7 FHIR R4 1-Click Order Generator
│       └── lib/
│           ├── api.ts                          # Backend REST Client
│           └── utils.ts                        # Styling Utilities
│
├── evaluations/                                # Staged Model Evaluation Artifacts
│   ├── stage1/                                 # Stage 1 ROC, PR, CM, SHAP & Lift Curves
│   ├── stage2/                                 # Stage 2 ROC, PR, CM, SHAP & Lift Curves
│   ├── stage3/                                 # Stage 3 ROC, PR, CM, SHAP & Lift Curves
│   └── stage4/                                 # Stage 4 ROC, PR, CM, SHAP & Lift Curves
│
├── data/                                       # Raw & Processed ADNI/OASIS Harmonized Data
└── scripts/                                    # Training & Evaluation Python Scripts
    ├── train_all_stages.py                     # 5-Fold Stratified Group K-Fold Pipeline
    └── generate_eval_plots.py                  # Evaluation Chart & Metrics Generator
```

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
* **macOS / Linux / Windows WSL2**
* **Python 3.10+** (Python 3.11, 3.12, 3.13, 3.14 fully supported)
* **Node.js 18+** & **npm 9+**

### 1. Clone & Setup Workspace
```bash
git clone https://github.com/YatharthG/StepWise-PRO.git
cd StepWise-PRO
```

### 2. Automated One-Command Startup
The included `start_servers.sh` script verifies dependencies, initializes the database, and launches both backend and frontend servers simultaneously:
```bash
chmod +x start_servers.sh
./start_servers.sh
```

### 3. Manual Step-by-Step Launch (Alternative)

#### Backend Setup:
```bash
# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python backend dependencies
pip install numpy pandas fastapi uvicorn joblib pillow scikit-learn scipy xgboost pydicom pydantic python-multipart matplotlib torch torchvision

# Initialize and seed database
python3 -c "from backend.database import init_db, reseed_all_patient_visits_db; init_db(); reseed_all_patient_visits_db()"

# Start FastAPI backend server
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup:
```bash
cd frontend
npm install
npm run dev
```

### 4. Access the Platform
* **Clinical Web Application:** [http://localhost:3000](http://localhost:3000)
* **Interactive OpenAPI / Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **FastAPI Backend Healthcheck:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 📡 API Reference & Endpoint Contracts

### Core REST API Endpoints:

| Method | Endpoint | Description | Sample Payload / Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service healthcheck & loaded model status | `{"status":"healthy","loaded_stages":[1,2,3,4]}` |
| `GET` | `/api/patients` | Retrieve all patient dossiers & triage rank | `[{"id":"ADNI_051_S_4048","name":"Patient 4048",...}]` |
| `GET` | `/api/patients/{id}` | Retrieve comprehensive single patient record | Full multi-stage patient object |
| `POST` | `/api/triage/predict` | Multi-stage calibrated progression prediction & SHAP | `{"stage": 2, "patient_data": {...}}` |
| `POST` | `/api/vision/analyze-mri` | PyTorch ResNet-18 MRI volume & Grad-CAM analysis | Accepts DICOM / PNG / JPEG scan |
| `POST` | `/api/vision/analyze-pet` | TorchScript PET tracer Centiloid regression | Accepts PET tracer scan array |
| `POST` | `/api/simulation/roi` | Health-economic & scanner capacity simulation | `{"annual_patients": 1000, "mri_cost": 800}` |
| `POST` | `/api/orders/servicerequest` | Dispatch HL7 FHIR R4 `ServiceRequest` | `{"patient_id": "...", "test_type": "MRI"}` |

---

### Data Acknowledgments
Data used in preparation of this project were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database (`adni.loni.usc.edu`), the **Open Access Series of Imaging Studies (OASIS)**, and the **Longitudinal Aging Study in India (LASI-DAD)**.

---

<div align="center">
  <sub>Built with precision for <strong>GE Healthcare Precision Care Challenge 2026</strong>. Dedicated to transforming early Alzheimer's detection and patient care worldwide.</sub>
</div>
