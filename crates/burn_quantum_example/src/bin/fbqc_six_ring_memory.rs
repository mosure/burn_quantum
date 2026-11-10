//! Rotated six-ring toric FBQC with physical four-photon Shor encoding and
//! fixed-ZZ-failure standard fusions. A native Burn oracle checks a six-ring
//! resource; bounded host sampling/decoding retains every physical attempt.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
use burn_quantum::{DenseSimulator, Matrix, Observable};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::time::Instant;
fn run() -> ExampleResult<Report> {
    let engine = DenseSimulator::default();
    let resource = ResourceState::six_ring();
    let state = resource
        .circuit()?
        .run(&engine, None, None, 1, &execution_device()?)?;
    let mut xzz = vec![0.; 27 * 27];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                xzz[((1 - a) * 9 + b * 3 + c) * 27 + a * 9 + b * 3 + c] =
                    if (b + c) % 2 == 0 { 1. } else { -1. };
            }
        }
    }
    let expected = values(
        Observable::new(vec![0, 1, 5], Matrix::real(27, xzz)?)?.expectation(&engine, &state)?,
    )?[0];
    ensure(expected > 1. - 2e-5, "native six-ring stabilizer oracle")?;
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
        ],
        vec![FabricLink {
            from: NodeId(0),
            to: NodeId(1),
            ticks: 1,
            transmission: RailTransmission::IDEAL,
        }],
    )?;
    let mut metrics = vec![
        ("native_six_ring_xzz", expected),
        ("host_attempts_per_model", 64.),
    ];
    let labels = [
        [
            "side2_bare_corrected",
            "side2_shor_corrected",
            "side2_shor_logical_errors",
            "side2_shor_ambiguous",
            "side2_shor_budget",
            "side2_shor_failure_ci_low",
            "side2_shor_failure_ci_high",
            "side2_detected_inner_checks",
            "side2_photons_per_attempt",
            "side2_fusions_per_attempt",
            "side2_ideal_complete_bell_corrected",
            "side2_host_seconds",
        ],
        [
            "side4_bare_corrected",
            "side4_shor_corrected",
            "side4_shor_logical_errors",
            "side4_shor_ambiguous",
            "side4_shor_budget",
            "side4_shor_failure_ci_low",
            "side4_shor_failure_ci_high",
            "side4_detected_inner_checks",
            "side4_photons_per_attempt",
            "side4_fusions_per_attempt",
            "side4_ideal_complete_bell_corrected",
            "side4_host_seconds",
        ],
    ];
    for (side, labels) in [2, 4].into_iter().zip(labels) {
        let start = Instant::now();
        let network = SixRingTorus::new(side, NodeId(0), NodeId(1), 4096)?;
        let code = ToricFusionCode::six_ring(&network);
        let bare = PhysicalToricMemory::bare(code.clone());
        let shor = PhysicalToricMemory::shor(code, 4096)?;
        let bare_plan = bare.program().compile(&fabric, CompileOptions::default())?;
        let shor_plan = shor.program().compile(&fabric, CompileOptions::default())?;
        let pauli = PauliNoise::new(0.0005, 0.0005, 0.0005)?;
        let physical = NetworkNoise::new(0.003, pauli, true)?;
        let bare_result = bare.run(&bare_plan, physical, 0, 64, 281, MemoryBudgets::default())?;
        let encoded = shor.run(&shor_plan, physical, 0, 64, 281, MemoryBudgets::default())?;
        let ideal = shor.run(
            &shor_plan,
            NetworkNoise::new(0., PauliNoise::NONE, false)?,
            0,
            4,
            281,
            MemoryBudgets::default(),
        )?;
        ensure(ideal.memory.corrected == 4, "ideal encoded six-ring memory")?;
        let m = &encoded.memory;
        ensure(
            m.attempts == m.corrected + m.logical_errors + m.erased_logicals + m.decoder_budget,
            "encoded physical attempts lost from accounting",
        )?;
        let ci = m.unsuccessful_interval(1.95996398454)?;
        for (name, value) in labels.into_iter().zip([
            bare_result.memory.corrected as f64,
            m.corrected as f64,
            m.logical_errors as f64,
            m.erased_logicals as f64,
            m.decoder_budget as f64,
            ci.lower,
            ci.upper,
            encoded.detected_inner_events as f64,
            encoded.photons_per_attempt as f64,
            encoded.fusions_per_attempt as f64,
            ideal.memory.corrected as f64,
            start.elapsed().as_secs_f64(),
        ]) {
            metrics.push((name, value));
        }
    }
    Report::new("fbqc_six_ring_memory", 281, 0, &metrics)
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn physical_six_ring_network_and_native_oracle() {
        super::run().unwrap();
    }
}
