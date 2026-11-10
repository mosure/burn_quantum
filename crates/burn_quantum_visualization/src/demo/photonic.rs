//! Bell-resource chains with a shared photonic fabric and unpostselected loss accounting.
//!
//! The statevector timeline is ideal resource preparation. The independently sampled
//! instrument includes transport/storage/detection loss and destructive fusions.
//! Keeping those records separate avoids inventing a pure-state loss trajectory.
use super::{DemoConfig, Preview};
use crate::{
    model::*,
    simulation::{TraceLimits, simulate},
};
use burn::tensor::Device;
use burn_quantum::photonic::stabilizer::{
    Pauli, PauliProduct, StabilizerShot, StabilizerSimulator,
};
use burn_quantum::{Circuit, Error, Register, Result, Unitary, photonic::*};

const SHOTS: usize = 2_048;
const SEED: u64 = 41_027;

/// Exhaustive outcomes of one complete resource-chain attempt, before selection.
#[derive(Clone, Debug)]
struct ChainStatistics {
    /// At least one consumed photon was lost: fusion information is erased.
    erasure: f64,
    /// Every consumed photon survives, but at least one analyzer reports failure.
    ordinary_failure: f64,
    /// Every analyzer succeeds, but at least one retained output is vacuum.
    lost_output: f64,
    /// Every analyzer succeeds and both output photons survive.
    delivered: f64,
    herald: f64,
    output_survival: f64,
    /// Counts in the same order as the four exhaustive probabilities above.
    counts: [usize; 4],
    /// Fidelity after the herald-dependent Pauli frame, conditioned on delivery.
    corrected_fidelity: Option<f64>,
}

fn resource_circuit(resources: usize, physical: bool) -> Result<(Circuit, Vec<String>)> {
    let register = if physical {
        DualRail::register(2 * resources)?
    } else {
        Register::qubits(2 * resources)?
    };
    let mut circuit = Circuit::new(register, 0, 0);
    let mut labels = Vec::with_capacity(4 * resources);
    let h = if physical {
        DualRail::unitary(&Unitary::h())?
    } else {
        Unitary::h()
    };
    // This is precisely ResourceState::bell_pair's graph-state recipe. It works
    // in either code-space representation without interpreting vacuum as |1>.
    for resource in 0..resources {
        let a = 2 * resource;
        let b = a + 1;
        for q in [a, b] {
            circuit.unitary(vec![q], h.clone())?;
            labels.push(format!("Resource {} · H q{q}", resource + 1));
        }
        circuit.phase_flip(vec![(a, 1), (b, 1)])?;
        labels.push(format!("Resource {} · CZ q{a},q{b}", resource + 1));
        circuit.unitary(vec![b], h.clone())?;
        labels.push(format!(
            "Resource {} · H q{b}: Bell pair ready",
            resource + 1
        ));
    }
    Ok((circuit, labels))
}

fn compile_chain(resources: usize) -> Result<CompiledFusionNetwork> {
    let switch = NodeId(resources);
    let analyzer = NodeId(resources + 1);
    let storage = RailTransmission::uniform(0.999)?;
    let nodes = (0..resources)
        .map(|_| FabricNode {
            kind: NodeKind::Source,
            storage,
        })
        .chain([
            FabricNode {
                kind: NodeKind::Switch,
                storage,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage,
            },
        ])
        .collect();
    let mut links = (0..resources)
        .map(|source| {
            Ok(FabricLink {
                from: NodeId(source),
                to: switch,
                ticks: 2 + (source % 2) as u64,
                transmission: RailTransmission::uniform(0.985 - 0.005 * (source % 3) as f64)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    links.push(FabricLink {
        from: switch,
        to: analyzer,
        ticks: 2,
        transmission: RailTransmission::uniform(0.98)?,
    });
    let fabric = Fabric::new(nodes, links)?;
    let mut program = FusionProgram::new();
    let photons = (0..resources)
        .map(|source| program.add_resource(ResourceState::bell_pair(), NodeId(source)))
        .collect::<Result<Vec<_>>>()?;
    for r in 0..resources - 1 {
        program.fuse(photons[r][1], photons[r + 1][0], analyzer)?;
    }
    program.compile(
        &fabric,
        CompileOptions {
            detector_transmission: RailTransmission::uniform(0.97)?,
            ..Default::default()
        },
    )
}

fn readouts(plan: &CompiledFusionNetwork) -> Result<Vec<PauliProduct>> {
    let [a, b] = [plan.outputs()[0].0, plan.outputs()[1].0];
    [Pauli::X, Pauli::Y, Pauli::Z]
        .into_iter()
        .map(|p| PauliProduct::new(vec![(a, p), (b, p)]))
        .collect()
}

fn statistics(plan: &CompiledFusionNetwork, shots: &[StabilizerShot]) -> Result<ChainStatistics> {
    // Closed form only for this Bell chain, balanced independent attenuation and
    // the standard unboosted analyzer. Fusion success is NOT generally a fair coin
    // on arbitrary entangled inputs; the chain's independent Bell links justify it.
    let outputs = plan.outputs();
    let consumed_survival: f64 = plan
        .transmissions()
        .iter()
        .enumerate()
        .filter(|(q, _)| !outputs.iter().any(|p| p.0 == *q))
        .map(|(_, eta)| eta.values()[0])
        .product();
    let output_survival = outputs
        .iter()
        .map(|q| plan.transmissions()[q.0].values()[0])
        .product::<f64>();
    let ideal_herald = 0.5_f64.powi(plan.fusions().len() as i32);
    let herald = consumed_survival * ideal_herald;
    let mut counts = [0; 4];
    let mut fidelity_sum = 0.;
    for shot in shots {
        if shot
            .outcomes
            .iter()
            .any(|o| matches!(o, FusionOutcome::Loss))
        {
            counts[0] += 1;
        } else if shot
            .outcomes
            .iter()
            .any(|o| !matches!(o, FusionOutcome::Success { .. }))
        {
            counts[1] += 1;
        } else if shot.output_digits.contains(&DualRail::VACUUM) {
            counts[2] += 1;
        } else {
            counts[3] += 1;
            if shot.pauli_expectations.len() != 3 {
                return Err(Error::Shape(
                    "Bell-chain fidelity requires XX, YY, ZZ readouts".into(),
                ));
            }
            // Each successful Psi projection contributes X; every odd XX parity
            // contributes Z. Track these signs rather than changing raw outcomes.
            let x = if plan.fusions().len().is_multiple_of(2) {
                1.
            } else {
                -1.
            };
            let odd_xx = shot
                .outcomes
                .iter()
                .filter(|o| matches!(o, FusionOutcome::Success { xx: Parity::Odd }))
                .count();
            let z = if odd_xx.is_multiple_of(2) { 1. } else { -1. };
            let [xx, yy, zz] =
                std::array::from_fn::<_, 3, _>(|i| shot.pauli_expectations[i] as f64);
            fidelity_sum += (1. + z * xx - x * z * yy + x * zz) / 4.;
        }
    }
    Ok(ChainStatistics {
        erasure: 1. - consumed_survival,
        ordinary_failure: consumed_survival * (1. - ideal_herald),
        lost_output: herald * (1. - output_survival),
        delivered: herald * output_survival,
        herald,
        output_survival,
        counts,
        corrected_fidelity: (counts[3] > 0).then(|| fidelity_sum / counts[3] as f64),
    })
}

pub(super) fn preview(config: DemoConfig, limits: TraceLimits) -> Result<Preview> {
    if !(2..=6).contains(&config.size) || config.variant > 1 {
        return Err(Error::InvalidValue(
            "photonic chain requires 2–6 Bell resources and logical/physical mode".into(),
        ));
    }
    let (circuit, _) = resource_circuit(config.size, config.variant == 1)?;
    let steps = circuit.operations().len();
    limits.check(circuit.register(), steps + 1)?;
    let dimension = circuit.register().size();
    let local: usize = circuit
        .register()
        .dimensions()
        .iter()
        .map(|d| 16 * d * d + 8 * d + 32)
        .sum();
    Ok(Preview {
        wires: circuit.register().wires(),
        dimension,
        steps,
        snapshot_bytes: (24 * dimension + local) * (steps + 1),
    })
}

pub(super) async fn build(
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    preview(config, limits)?;
    let (circuit, labels) = resource_circuit(config.size, config.variant == 1)?;
    let plan = compile_chain(config.size)?;
    let mut fabric = FabricSnapshot::from_compiled(&plan);
    for (r, source) in fabric.nodes[..config.size].iter_mut().enumerate() {
        source.label = format!("Bell source {}", r + 1);
    }
    fabric.nodes[config.size].label = "Shared switch".into();
    fabric.nodes[config.size + 1].label = "Bell analyzer".into();
    let trace = simulate(&circuit, &[], &[], &labels, device, limits).await?;
    let simulator = StabilizerSimulator::default();
    let readouts = readouts(&plan)?;
    let mut shots = Vec::with_capacity(SHOTS);
    // Bounded chunks yield to browser input; each shot retains a stable global ID.
    for start in (0..SHOTS).step_by(128) {
        shots.extend(simulator.sample(&plan, start as u64, 128, SEED, &readouts)?);
        crate::simulation::yield_to_browser().await;
    }
    let stats = statistics(&plan, &shots)?;
    let representation = if config.variant == 0 {
        "Logical qubits: |0> and |1> represent one photon occupying either rail. Each pair is a Bell resource; its individual Bloch vectors sit at the center because the photons are entangled."
    } else {
        "Physical encoding: local levels 0 = |10>, 1 = |01>, 2 = |00> vacuum. The third level records photon loss; it is not a third logical level. The ideal preparation timeline has no vacuum population."
    };
    let study = StudyReport {
        title: "Deliver entanglement through a lossy photonic fabric".into(),
        summary: format!(
            "{} small Bell resources feed a shared switch and unboosted Bell analyzer. {} destructive fusions attempt to connect endpoint photons 0 and {}. Count every attempt: a high conditional Bell fidelity does not imply a high usable-pair yield.",
            config.size, config.size - 1, 2 * config.size - 1,
        ),
        metrics: vec![
            StudyMetric::new("Delivered pair · exact", 100. * stats.delivered, "%"),
            StudyMetric::new("Delivered pair · sampled", 100. * stats.counts[3] as f64 / SHOTS as f64, "%"),
            StudyMetric::new("Supplied photons / delivered pair", 2. * config.size as f64 / stats.delivered, "expected"),
            StudyMetric::new("All fusions herald success", 100. * stats.herald, "%"),
            StudyMetric::new("Both output photons survive", 100. * stats.output_survival, "%"),
            StudyMetric::new("Consumed-photon erasure", 100. * stats.erasure, "%"),
            StudyMetric::new("Ordinary fusion failure", 100. * stats.ordinary_failure, "%"),
            StudyMetric::new("Successful herald, lost output", 100. * stats.lost_output, "%"),
            StudyMetric::new("Corrected fidelity | delivered", stats.corrected_fidelity.unwrap_or(0.), ""),
            StudyMetric::new("Schedule duration", plan.finish_tick() as f64, "ticks"),
        ],
        notes: vec![
            representation.into(),
            "The circuit timeline prepares ideal resource states. The fabric animation shows actual compiled packet reservations, not measured surviving photons or observed analyzer outcomes. The independent loss-aware ensemble uses this same schedule, including output storage until completion.".into(),
            format!("{SHOTS} unconditional attempts; seed {SEED}. Host stabilizer trajectories provide polynomial-memory Clifford simulation, while Burn on the selected device computes the statevector timeline. Counts: {} consumed-photon erasures; {} ordinary failures; {} successful heralds with lost output; {} delivered pairs. Output loss is counted after successful heralding; ordinary failure otherwise takes precedence.", stats.counts[0], stats.counts[1], stats.counts[2], stats.counts[3]),
            "Exact yield = (1/2)^(resources−1) × product of every photon's compiled transmission. The standard analyzer is 50% successful for this independent Bell-chain construction; this probability is not universal for arbitrary fusion inputs. The three failure probabilities and delivered yield sum to one.".into(),
            "A delivered pair requires every fusion to succeed AND both outputs to survive. Fidelity is evaluated only on this surviving logical subspace, normalized by delivered-attempt count; the unconditional survival/yield mass is retained above. XX, YY and ZZ expectations are corrected with the recorded fusion Pauli frame before computing Bell fidelity.".into(),
            "Illustrative assumptions: ideal on-demand Bell resources; equal independent loss on both rails; source/switch/analyzer storage 99.9% per tick; source links 97.5–98.5%; analyzer link 98%; detector transmission 97%; ideal photon-number resolution. No dark counts, multiphoton emission, distinguishability error, active retries or fault-tolerant encoding. These values are not PsiQuantum calibration data.".into(),
            "Dual-rail photonic hardware encodes a qubit in two modes; the optional vacuum level is simulator bookkeeping. PsiQuantum describes this qubit encoding publicly: https://www.psiquantum.com/quantum-computing . This example is a generic fusion-network model, not an implementation of a provider-specific device.".into(),
            "Scale resources from 2 to 6. The explicit logical statevector has 2^(2R) amplitudes and the vacuum encoding has 3^(2R); default retained-trace limits allow up to R=4 in vacuum mode. Stabilizer transport/fusion sampling uses polynomial tableau storage. Growing an unencoded serial chain lowers yield exponentially: scalable fault tolerance requires multiplexing, encoding and decoding, not simply more photons.".into(),
        ],
        classification: None, sat: None,
        photonic: Some(PhotonicSummary {
            logical_qubits: 2 * config.size,
            includes_vacuum: config.variant == 1,
            attempts: SHOTS,
            delivered: stats.counts[3],
            erased: stats.counts[0] + stats.counts[2],
        }),
    };
    let document = VisualizationDocument {
        schema_version: 2,
        title: "Photonic Bell-resource network".into(),
        description: format!(
            "{representation}\n{} ideally supplied Bell resources, {} photons and {} destructive fusion attempts connect two surviving endpoints. Quantum frames show resource preparation; routing markers are scheduled packets. A separate {SHOTS}-shot host stabilizer ensemble applies the compiled transport, storage, detector loss and fusion instrument.\nExact delivered-pair yield {:.3}% per attempt; sampled {} / {SHOTS}. Corrected Bell fidelity {:.6} is conditioned on both successful heralds and surviving outputs. This illustrative balanced-loss model is not a calibrated hardware forecast.",
            config.size,
            2 * config.size,
            config.size - 1,
            100. * stats.delivered,
            stats.counts[3],
            stats.corrected_fidelity.unwrap_or(0.),
        ),
        circuit,
        parameters: vec![],
        inputs: vec![],
        operation_labels: labels,
        trace,
        fabric: Some(fabric),
        optimization: vec![],
        protocol: None,
        study: Some(study),
    };
    document.validate_with(limits)?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_density_instrument_preserves_unconditional_yield_and_logical_projection() {
        pollster::block_on(async {
            use crate::simulation::read_tensor;
            use burn::tensor::DType;
            let plan = compile_chain(2).unwrap();
            let exact = statistics(&plan, &[]).unwrap();
            let run = DenseFusionSimulator::default()
                .execute(&plan, 1, &Device::flex(), DType::F32)
                .unwrap();
            let mut mass = [0.; 4];
            for branch in run.branches() {
                let weight = read_tensor(branch.probability()).await.unwrap()[0];
                match branch.outcomes()[0] {
                    FusionOutcome::Loss => mass[0] += weight,
                    FusionOutcome::Failure { .. } => mass[1] += weight,
                    FusionOutcome::Success { xx } => {
                        let re = read_tensor(branch.unnormalized_state().unwrap().entries().real())
                            .await
                            .unwrap();
                        // Project both outputs into their logical subspace; keep
                        // its absolute probability BEFORE normalizing the 4x4 block.
                        let indices = [0, 1, 3, 4];
                        let code_mass: f64 = indices.iter().map(|&i| re[9 * i + i]).sum();
                        mass[2] += weight - code_mass;
                        mass[3] += code_mass;
                        let normalized_trace: f64 =
                            indices.iter().map(|&i| re[9 * i + i] / code_mass).sum();
                        assert!((normalized_trace - 1.).abs() < 2e-6);
                        let sign = if xx == Parity::Even { 1. } else { -1. };
                        // Success projects the pair to Psi+/- before X/Z frame correction.
                        let fidelity =
                            (re[10] + re[30] + sign * (re[12] + re[28])) / (2. * code_mass);
                        assert!((fidelity - 1.).abs() < 2e-6);
                    }
                    _ => panic!("unexpected analyzer outcome"),
                }
            }
            for (actual, expected) in mass.into_iter().zip([
                exact.erasure,
                exact.ordinary_failure,
                exact.lost_output,
                exact.delivered,
            ]) {
                assert!((actual - expected).abs() < 2e-6, "{actual} vs {expected}");
            }
            assert!((read_tensor(run.total_probability()).await.unwrap()[0] - 1.).abs() < 2e-6);
        });
    }

    #[test]
    fn balanced_loss_chain_probabilities_and_pauli_frames_match_trajectories() {
        for resources in [2, 3, 6] {
            let plan = compile_chain(resources).unwrap();
            let shots = StabilizerSimulator::default()
                .sample(&plan, 0, 8_192, SEED, &readouts(&plan).unwrap())
                .unwrap();
            let stats = statistics(&plan, &shots).unwrap();
            let exact = [
                stats.erasure,
                stats.ordinary_failure,
                stats.lost_output,
                stats.delivered,
            ];
            assert!((exact.iter().sum::<f64>() - 1.).abs() < 1e-12);
            for (p, count) in exact.into_iter().zip(stats.counts) {
                let measured = count as f64 / shots.len() as f64;
                let bound =
                    6. * (p * (1. - p) / shots.len() as f64).sqrt() + 1. / shots.len() as f64;
                assert!(
                    (p - measured).abs() < bound,
                    "{resources} resources: {p} vs {measured}"
                );
            }
            assert_eq!(stats.corrected_fidelity, Some(1.));
            assert!((stats.herald * stats.output_survival - stats.delivered).abs() < 1e-12);
            assert_eq!(plan.program().num_photons(), 2 * resources);
            assert_eq!(
                plan.outputs().iter().map(|q| q.0).collect::<Vec<_>>(),
                [0, 2 * resources - 1]
            );
        }
    }

    #[test]
    fn physical_and_logical_resource_states_have_identical_code_space_probabilities() {
        pollster::block_on(async {
            let limits = TraceLimits::default();
            let device = Device::flex();
            let (logical, labels) = resource_circuit(2, false).unwrap();
            let (physical, _) = resource_circuit(2, true).unwrap();
            let a = simulate(&logical, &[], &[], &labels, &device, limits)
                .await
                .unwrap();
            let b = simulate(&physical, &[], &[], &labels, &device, limits)
                .await
                .unwrap();
            for (a, b) in a.frames.iter().zip(&b.frames) {
                for (index, &probability) in a.probabilities.iter().enumerate() {
                    let ternary = (0..4).fold(0, |value, q| value * 3 + ((index >> (3 - q)) & 1));
                    assert!((probability - b.probabilities[ternary]).abs() < 2e-6);
                }
                assert!(b.subsystems.iter().all(|q| q.probabilities[2].abs() < 1e-7));
                assert!(a.subsystems.iter().all(|q| q.bloch.is_some()));
            }
            for state in &a.frames.last().unwrap().subsystems {
                assert!(state.bloch.unwrap().iter().all(|x| x.abs() < 2e-6));
            }
        });
    }

    #[test]
    fn preflight_bounds_the_explicit_vacuum_state_space() {
        let limits = TraceLimits::default();
        let config = |size, variant| DemoConfig {
            size,
            variant,
            depth: 1,
        };
        assert_eq!(preview(config(6, 0), limits).unwrap().dimension, 4_096);
        assert_eq!(preview(config(4, 1), limits).unwrap().dimension, 6_561);
        assert!(preview(config(5, 1), limits).is_err());
        assert!(preview(config(6, 1), limits).is_err());
        assert!(preview(config(7, 0), limits).is_err());
    }
}
