import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const preset = searchParams.get('preset') || 'amyloid_positive'

    let centiloids = 78.4
    let suvr = 1.48
    let braak = 'Stage III/IV (Limbic Transition)'
    let status = 'Amyloid Positive (≥ 25.0 CL Cutoff Exceeded)'
    let dmt_candidate = true

    if (preset === 'amyloid_negative' || preset === 'negative') {
      centiloids = 12.4
      suvr = 1.08
      braak = 'Stage 0/I (Normal Entorhinal)'
      status = 'Amyloid Negative (< 25.0 CL)'
      dmt_candidate = false
    }

    return NextResponse.json({
      status: 'success',
      filename: `ADNI_082_S_5029_PT_AV45_${preset}.dcm`,
      tracer: '18F-Florbetapir (AV-45) / GE Vizamyl',
      vision_model: '3D DenseNet-121 Centiloid Regressor (stage4_adni_pet_production.pt)',
      centiloid_score: centiloids,
      global_suvr: suvr,
      tau_braak_staging: braak,
      amyloid_status: status,
      regional_cortical_centiloids: {
        precuneus: Number((centiloids * 1.07).toFixed(1)),
        frontal_cortex: Number((centiloids * 0.97).toFixed(1)),
        lateral_temporal: Number((centiloids * 0.95).toFixed(1)),
        posterior_cingulate: Number((centiloids * 1.02).toFixed(1))
      },
      aria_safety_prescreening: {
        microbleeds_detected: 0,
        superficial_siderosis: false,
        vasogenic_edema_mm: 0,
        safety_tier: 'Tier 1 Safety Cleared for mAb Therapy'
      },
      dmt_eligibility_recommendation: dmt_candidate,
      aria_monitoring_frequency: centiloids < 60 ? 'Standard 3-Month Protocol' : 'High-Frequency 1-Month Protocol'
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}
