//! Periodic four-star FBQC memory: physical photon loss, Pauli faults, and bounded decoding.
//! The native selected device checks the four-photon resource; the lattice study is host-only.
//! The complete-Bell run is an ideal effective reference. The unboosted run retains actual
//! fixed-ZZ-failure parity erasures; neither finite study establishes a hardware threshold.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
use burn_quantum::{DenseSimulator, Matrix, Observable};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::time::Instant;
fn run() -> ExampleResult<Report> {
    let resource = ResourceState::four_star();
    let circuit = resource.circuit()?;
    let engine = DenseSimulator::default();
    let state = circuit.run(&engine, None, None, 1, &execution_device()?)?;
    state.validate(2e-5)?;
    let mut diagonal = vec![0.; 81 * 81];
    for i in 0..81 {
        let mut n = i;
        let mut sign = 1.;
        for _ in 0..4 {
            let digit = n % 3;
            n /= 3;
            sign *= match digit {
                0 => 1.,
                1 => -1.,
                _ => 0.,
            };
        }
        diagonal[i * 81 + i] = sign;
    }
    let zzzz = values(
        Observable::new(vec![0, 1, 2, 3], Matrix::real(81, diagonal)?)?
            .expectation(&engine, &state)?,
    )?[0];
    let mut xx = vec![0.; 81];
    for a in 0..2 {
        for b in 0..2 {
            xx[((1 - a) * 3 + 1 - b) * 9 + a * 3 + b] = 1.;
        }
    }
    let xx =
        values(Observable::new(vec![0, 1], Matrix::real(9, xx)?)?.expectation(&engine, &state)?)?
            [0];
    ensure(
        zzzz > 1. - 2e-5 && xx > 1. - 2e-5,
        "native four-star stabilizer oracle",
    )?;
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
        ("native_resource_zzzz", zzzz),
        ("native_resource_xx", xx),
        ("host_attempts_per_model", 256.),
    ];
    let names = [
        [
            "distance2_complete_bell_corrected",
            "distance2_pauli_logical_errors",
            "distance2_erased_logical",
            "distance2_budget",
            "distance2_failure_ci_low",
            "distance2_failure_ci_high",
            "distance2_unboosted_success",
            "distance2_frame_bytes",
            "distance2_host_seconds",
            "distance2_unboosted_erased_logicals",
            "distance2_unboosted_failure_ci_low",
            "distance2_unboosted_failure_ci_high",
            "distance2_setup_seconds",
            "distance2_sampling_seconds",
            "distance2_decoding_seconds",
        ],
        [
            "distance3_complete_bell_corrected",
            "distance3_pauli_logical_errors",
            "distance3_erased_logical",
            "distance3_budget",
            "distance3_failure_ci_low",
            "distance3_failure_ci_high",
            "distance3_unboosted_success",
            "distance3_frame_bytes",
            "distance3_host_seconds",
            "distance3_unboosted_erased_logicals",
            "distance3_unboosted_failure_ci_low",
            "distance3_unboosted_failure_ci_high",
            "distance3_setup_seconds",
            "distance3_sampling_seconds",
            "distance3_decoding_seconds",
        ],
        [
            "distance4_complete_bell_corrected",
            "distance4_pauli_logical_errors",
            "distance4_erased_logical",
            "distance4_budget",
            "distance4_failure_ci_low",
            "distance4_failure_ci_high",
            "distance4_unboosted_success",
            "distance4_frame_bytes",
            "distance4_host_seconds",
            "distance4_unboosted_erased_logicals",
            "distance4_unboosted_failure_ci_low",
            "distance4_unboosted_failure_ci_high",
            "distance4_setup_seconds",
            "distance4_sampling_seconds",
            "distance4_decoding_seconds",
        ],
    ];
    for (distance, names) in [2, 3, 4].into_iter().zip(names) {
        let begin = Instant::now();
        let network = FourStarTorus::new(distance, NodeId(0), NodeId(1), 4096)?;
        let plan = network
            .program()
            .compile(&fabric, CompileOptions::default())?;
        let ideal = run_four_star_memory(
            &network,
            &plan,
            NetworkNoise::new(0., PauliNoise::NONE, false)?,
            0,
            4,
            815,
            18,
            1 << 24,
        )?;
        ensure(
            ideal.corrected == ideal.attempts,
            "ideal four-star memory failed",
        )?;
        let pauli = PauliNoise::new(0.0005, 0.0005, 0.0005)?;
        let (report, timings) = run_four_star_memory_profiled(
            &network,
            &plan,
            NetworkNoise::new(0.005, pauli, false)?,
            0,
            256,
            815,
            18,
            1 << 24,
        )?;
        let unboosted = run_four_star_memory(
            &network,
            &plan,
            NetworkNoise::new(0.005, pauli, true)?,
            0,
            256,
            815,
            18,
            1 << 24,
        )?;
        ensure(
            report.attempts
                == report.corrected
                    + report.logical_errors
                    + report.erased_logicals
                    + report.decoder_budget,
            "missing physical attempts",
        )?;
        ensure(
            [
                timings.setup_seconds,
                timings.sampling_seconds,
                timings.decoding_seconds,
            ]
            .iter()
            .all(|t| t.is_finite() && *t >= 0.),
            "invalid host profiling diagnostics",
        )?;
        let interval = report.unsuccessful_interval(1.95996398454)?;
        let unboosted_interval = unboosted.unsuccessful_interval(1.95996398454)?;
        let initial = StabilizerSimulator::default().prepare(&plan)?;
        let sampler = ReferenceSampler::compile(&initial, &network.observables()?, 1 << 24)?;
        let session = sampler.start(0, 256, 815, pauli, 1 << 24)?;
        for (name, value) in names.into_iter().zip([
            report.corrected as f64,
            report.logical_errors as f64,
            report.erased_logicals as f64,
            report.decoder_budget as f64,
            interval.lower,
            interval.upper,
            unboosted.corrected as f64,
            session.frame_bytes() as f64,
            begin.elapsed().as_secs_f64(),
            unboosted.erased_logicals as f64,
            unboosted_interval.lower,
            unboosted_interval.upper,
            timings.setup_seconds,
            timings.sampling_seconds,
            timings.decoding_seconds,
        ]) {
            metrics.push((name, value));
        }
    }
    Report::new("fbqc_four_star_memory", 815, 0, &metrics)
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn native_resource_and_noisy_four_star_networks() {
        super::run().unwrap();
    }
}
