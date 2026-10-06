import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const preset = searchParams.get('subject_preset') || 'mci_case'

    let hippo = 3.82
    let ventricles = 42.1
    let mta = 'MTA Grade 2'
    let dx = 'Mild Cognitive Impairment (MCI)'
    let conf = 87.3
    let percentile = '12th %ile (Moderate Atrophy)'

    if (preset === 'normal_case' || preset === 'normal') {
      hippo = 7.55
      ventricles = 11.5
      mta = 'MTA Grade 0'
      dx = 'Cognitively Normal (CN)'
      conf = 99.4
      percentile = '68th %ile (Normal)'
    } else if (preset === 'ad_case' || preset === 'ad') {
      hippo = 2.42
      ventricles = 58.1
      mta = 'MTA Grade 3'
      dx = 'Alzheimer’s Dementia (AD)'
      conf = 99.1
      percentile = '4th %ile (Severe Atrophy)'
    }

    return NextResponse.json({
      status: 'success',
      filename: `ADNI_082_S_5029_MR_T1_${preset}.dcm`,
      scan_type: 'T1-Weighted 3D Coronal MRI (GE SIGNA Premier 3.0T)',
      vision_model: 'PyTorch ResNet-50 (layer4 Saliency Head)',
      predicted_volumes: {
        hippocampus_cm3: hippo,
        hippocampus_percentile: percentile,
        ventricles_cm3: ventricles,
        intracranial_volume_cm3: 1450.2,
        hvr_ratio: Number((hippo / ventricles).toFixed(3)),
        mta_grade: mta,
        hippocampus_icv_ratio: Number(((hippo / 1450.2) * 1000.0).toFixed(2))
      },
      vision_classification: {
        predicted_class: dx,
        model_confidence: conf / 100.0,
        probabilities: {
          Cognitively_Normal: preset === 'normal_case' ? 0.99 : 0.04,
          Mild_Cognitive_Impairment: preset === 'mci_case' ? 0.87 : 0.08,
          Alzheimers_Dementia: preset === 'ad_case' ? 0.99 : 0.05
        }
      },
      auto_populated_stage3_vector: {
        hippocampus_icv_ratio: Number(((hippo / 1450.2) * 1000.0).toFixed(2)),
        wmh_volume_cm3: preset === 'normal_case' ? 3.1 : 7.4
      }
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}
