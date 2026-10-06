import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const sim = await request.json()
    const n_screened = sim.annual_screened_patients || 5000
    const mri_cap_annual = (sim.mri_weekly_capacity || 40) * 52
    const pet_cap_annual = (sim.pet_weekly_capacity || 15) * 52

    // Unoptimized conventional workflow: All patients get MRI, 50% get PET
    const conv_mri = n_screened
    const conv_pet = Math.round(n_screened * 0.48)
    const conv_cost = conv_mri * (sim.mri_cost_usd || 750) + conv_pet * (sim.pet_cost_usd || 3200)

    // StepWise Filtered Funnel:
    // Stage 1 filter: 42% proceed to blood
    const s2_patients = Math.round(n_screened * 0.42)
    // Stage 2 filter: 28% proceed to MRI
    const step_mri = Math.round(s2_patients * 0.65)
    // Stage 3 filter: 14% proceed to PET
    const step_pet = Math.round(step_mri * 0.45)

    const step_cost = 
      s2_patients * (sim.plasma_test_cost_usd || 250) +
      step_mri * (sim.mri_cost_usd || 750) +
      step_pet * (sim.pet_cost_usd || 3200)

    const savings = Math.max(0, conv_cost - step_cost)
    const mri_unlocked = Math.round(((conv_mri - step_mri) / conv_mri) * 100)
    const pet_unnecessary_reduction = Math.round(((conv_pet - step_pet) / conv_pet) * 100)
    const wait_time_months = Math.max(1.2, ((conv_mri / mri_cap_annual) - (step_mri / mri_cap_annual)) * 12)

    return NextResponse.json({
      cohort_size: n_screened,
      conventional_baseline: {
        total_cost_usd: conv_cost,
        mri_scans_ordered: conv_mri,
        pet_scans_ordered: conv_pet
      },
      stepwise_prioritized_pathway: {
        total_cost_usd: step_cost,
        plasma_assays_ordered: s2_patients,
        mri_scans_ordered: step_mri,
        pet_scans_ordered: step_pet
      },
      roi_impact: {
        total_annual_savings_usd: savings,
        mri_capacity_unlocked_pct: mri_unlocked,
        pet_unnecessary_scan_reduction_pct: pet_unnecessary_reduction,
        wait_time_reduction_months: Number(wait_time_months.toFixed(1))
      }
    })
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 400 })
  }
}
