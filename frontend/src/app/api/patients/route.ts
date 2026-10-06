import { NextResponse } from 'next/server'
import initialPatients from '@/data/patients.json'

// In-memory persistent store during runtime
let patientsStore = [...initialPatients]

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const stageParam = searchParams.get('stage')
  const velocity = searchParams.get('velocity')
  const search = searchParams.get('search')

  let results = [...patientsStore]

  if (stageParam) {
    const stage = parseInt(stageParam)
    results = results.filter((p: any) => (p.currentStage || p.current_stage) === stage)
  }

  if (velocity) {
    results = results.filter((p: any) => p.velocity?.toLowerCase().includes(velocity.toLowerCase()))
  }

  if (search) {
    const s = search.toLowerCase()
    results = results.filter((p: any) => 
      p.name.toLowerCase().includes(s) || 
      p.id.toLowerCase().includes(s) || 
      (p.mrn && p.mrn.toLowerCase().includes(s))
    )
  }

  results.sort((a: any, b: any) => ((b.riskScore ?? b.risk_score) || 0) - ((a.riskScore ?? a.risk_score) || 0))

  return NextResponse.json({
    total: results.length,
    patients: results
  })
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const newId = body.id || `P-${Date.now().toString().slice(-4)}`
    const newMrn = body.mrn || `GE-HC-${Math.floor(1000 + Math.random() * 9000)}`

    const newPatient = {
      id: newId,
      mrn: newMrn,
      name: body.name || 'New Patient Case',
      age: body.age || 70,
      gender: body.gender || 'Female',
      dob: body.dob || '1955-01-01',
      apoe: body.apoe || 'ε3/ε3',
      referral: body.referral || 'Memory Clinic Intake',
      complaint: body.complaint || 'Subjective memory concerns',
      currentStage: body.currentStage || 1,
      riskScore: body.riskScore || 0.45,
      velocity: body.velocity || 'Moderate',
      vitals: body.vitals || { bp: '124/80', bmi: 24.2, hba1c: '5.6%' },
      stage1: body.stage1 || {
        mmse: body.mmse || 24,
        moca: body.moca || 22,
        cdrsb: 1.0,
        faq: 2.0,
        riskScorePct: Math.round((body.riskScore || 0.45) * 100)
      },
      stage2: body.stage2 || {
        ptau217: body.ptau217 || 0.22,
        ab42_40: body.ab42_40 || 0.089,
        nfl: 14.2,
        gfap: 187.0,
        riskScorePct: Math.round((body.riskScore || 0.45) * 100)
      },
      stage3: body.stage3 || {
        hippoVol: 3.82,
        mtaGrade: 'MTA Grade 2',
        hippoPercentile: '12th %ile (Moderate Atrophy)',
        riskScorePct: Math.round((body.riskScore || 0.45) * 100)
      },
      stage4: body.stage4 || {
        centiloids: 78.4,
        atnClassification: 'A+ T+ N+ (Amyloid/Tau Positive)',
        dmtEligible: true,
        ariaCleared: true,
        riskScorePct: Math.round((body.riskScore || 0.45) * 100)
      },
      visits: [
        {
          visit_code: 'M00',
          visit_date: new Date().toISOString().split('T')[0],
          mmse: body.mmse || 24,
          moca: body.moca || 22,
          ptau217: body.ptau217 || 0.22,
          ab42_40: body.ab42_40 || 0.089,
          hippocampus_cm3: 3.82,
          centiloids: 78.4,
          risk_score: body.riskScore || 0.45,
          stage_dx: 'Stage 1 Initial Intake'
        }
      ]
    }

    patientsStore.unshift(newPatient as any)

    return NextResponse.json({
      status: 'success',
      patient: newPatient
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}

export async function PUT(request: Request) {
  try {
    const body = await request.json()
    const id = body.id || body.mrn
    if (!id) {
      return NextResponse.json({ error: 'Missing patient id' }, { status: 400 })
    }
    const index = patientsStore.findIndex((p: any) => p.id === id || p.mrn === id)
    if (index === -1) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 })
    }
    patientsStore[index] = {
      ...patientsStore[index],
      ...body
    }
    return NextResponse.json({
      status: 'success',
      patient: patientsStore[index]
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}

export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    if (!id) {
      return NextResponse.json({ error: 'Missing patient id' }, { status: 400 })
    }
    patientsStore = patientsStore.filter((p: any) => p.id !== id && p.mrn !== id)
    return NextResponse.json({ status: 'success', remaining: patientsStore.length })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 })
  }
}
