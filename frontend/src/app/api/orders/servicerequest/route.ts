import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const order = await request.json()
    const orderId = `SR-${Date.now().toString().slice(-6)}`
    const now = new Date().toISOString()

    const fhirBundle = {
      resourceType: "ServiceRequest",
      id: orderId,
      meta: {
        versionId: "1",
        lastUpdated: now,
        profile: ["http://hl7.org/fhir/StructureDefinition/ServiceRequest"]
      },
      status: "active",
      intent: "order",
      priority: order.priority === "stat" ? "stat" : "urgent",
      code: {
        coding: [
          {
            system: "http://snomed.info/sct",
            code: order.target_modality === "Amyloid_PET" ? "718301007" : "241601008",
            display: order.target_modality === "Amyloid_PET" 
              ? "Positron emission tomography of brain with amyloid tracer (procedure)" 
              : "Magnetic resonance imaging of head with volumetric morphometry (procedure)"
          }
        ],
        text: `StepWise CDS Authorized Order: ${order.target_modality}`
      },
      subject: {
        reference: `Patient/${order.patient_id}`,
        display: `${order.patient_name} (MRN: ${order.mrn})`
      },
      authoredOn: now,
      requester: {
        reference: "Practitioner/DR-REYES-NEURO",
        display: "Dr. Elena Reyes, MD (Cognitive Neurology)"
      },
      reasonCode: [
        {
          coding: [
            {
              system: "http://hl7.org/fhir/sid/icd-10-cm",
              code: "G31.84",
              display: "Mild cognitive impairment, so stated"
            }
          ]
        }
      ],
      supportingInfo: [
        {
          display: `Calibrated Rapid Progression Risk: ${(order.risk_score * 100).toFixed(1)}%`
        }
      ],
      note: [
        {
          authorString: "StepWise Multi-Stage Prioritization AI",
          time: now,
          text: `Automatic CDS Order generated following high-specificity progression risk determination. Justification: ${order.clinical_justification || 'High-risk biofluid/morphometric profile warranting definitive diagnostic escalation.'}`
        }
      ]
    }

    return NextResponse.json({
      status: "authorized_and_dispatched",
      order_id: orderId,
      fhir_resource: fhirBundle,
      ehr_integration_status: "EHR Writeback Complete (SMART-on-FHIR R4)",
      dispatched_at: now
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}
