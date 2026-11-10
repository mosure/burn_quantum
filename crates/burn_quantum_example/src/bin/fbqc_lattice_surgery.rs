//! Growing planar surgery, actual six-ring lowering, and explicit negative optical-yield scaling.
use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_quantum::{
    photonic::{
        qec::{fusion_surgery::*, *},
        stabilizer::*,
    },
    *,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use serde::Serialize;
#[derive(Serialize)]
struct DistanceRun {
    distance: usize,
    protected_x_distance: usize,
    protected_z_distance: usize,
    repeated_rounds: usize,
    source_rings: usize,
    source_photons: usize,
    check_preparations: usize,
    destructive_fusions: usize,
    log_ideal_herald_probability: f64,
    conditioned_channel_witness_error: usize,
}
#[derive(Serialize)]
struct BoundaryRun {
    width: usize,
    depth: usize,
    merge_window: Option<[usize; 2]>,
    source_rings: usize,
    source_photons: usize,
    destructive_fusions: usize,
    terminal_measurements: usize,
    end_code_photons: Vec<usize>,
    end_x_distances: Vec<usize>,
    end_z_distances: Vec<usize>,
    logical_choi_generators: Vec<String>,
    logical_lift_negative: Vec<bool>,
    signed_relation_record_counts: Vec<usize>,
    transmitted_logicals: usize,
}
fn boundary_runs() -> ExampleResult<Vec<BoundaryRun>> {
    use burn_quantum::photonic::NodeId;
    let mut result = vec![];
    for (width, depth, window) in [
        (3, 3, None),
        (5, 3, None),
        (7, 3, None),
        (3, 10, Some([0, 0])),
        (3, 10, Some([1, 4])),
        (3, 11, Some([1, 4])),
    ] {
        let network = match window {
            None => SixRingBoundaryNetwork::memory(width, depth, NodeId(0), NodeId(1), 4096),
            Some([0, 0]) => {
                SixRingBoundaryNetwork::separated(width, depth, NodeId(0), NodeId(1), 4096)
            }
            Some(window) => SixRingBoundaryNetwork::parity_merge(
                width,
                depth,
                window,
                NodeId(0),
                NodeId(1),
                4096,
            ),
        }?;
        let proof = network.certify(Default::default())?;
        ensure(
            proof.transmitted_logicals == if window == Some([0, 0]) { 2 } else { 1 },
            "six-ring boundary encoded channel rank changed",
        )?;
        let logical_choi_generators = proof
            .logical_observables
            .iter()
            .map(|p| {
                let mut label = vec!['I'; proof.codes.len()];
                for &(i, p) in p.terms() {
                    label[i] = match p {
                        Pauli::X => 'X',
                        Pauli::Y => 'Y',
                        Pauli::Z => 'Z',
                    };
                }
                label.into_iter().collect()
            })
            .collect();
        result.push(BoundaryRun {
            width,
            depth,
            merge_window: network.merge_window(),
            source_rings: network.resource_centers().len(),
            source_photons: network.patch().program().num_photons(),
            destructive_fusions: network.patch().program().fusions().len(),
            terminal_measurements: network.patch().boundary_measurements().len(),
            end_code_photons: proof.codes.iter().map(|c| c.photons.len()).collect(),
            end_x_distances: proof.codes.iter().map(|c| c.x_distance.distance).collect(),
            end_z_distances: proof.codes.iter().map(|c| c.z_distance.distance).collect(),
            logical_choi_generators,
            logical_lift_negative: proof.logical_phase_negative,
            signed_relation_record_counts: proof
                .logical_relations
                .iter()
                .map(|(_, r)| r.terms.len())
                .collect(),
            transmitted_logicals: proof.transmitted_logicals,
        });
    }
    Ok(result)
}
#[derive(Serialize)]
struct SurfaceRun {
    distance_label: usize,
    resource_layers: usize,
    merge_window: Option<[usize; 2]>,
    source_rings: usize,
    source_photons: usize,
    destructive_fusions: usize,
    terminal_measurements: usize,
    declared_h_frame_photons: usize,
    end_xz_distances: Vec<[usize; 2]>,
    logical_choi_generators: Vec<String>,
    logical_lift_negative: Vec<bool>,
    transmitted_logicals: usize,
}
fn surface_runs() -> ExampleResult<Vec<SurfaceRun>> {
    use burn_quantum::photonic::NodeId;
    let mut result = vec![];
    for (d, merge) in [(3, false), (5, false), (7, false), (3, true)] {
        let network = if merge {
            SixRingSurfaceNetwork::parity_merge(d, 1, d, 1, NodeId(0), NodeId(1), 4096)
        } else {
            SixRingSurfaceNetwork::memory(d, d, NodeId(0), NodeId(1), 4096)
        }?;
        let proof = network.certify(BoundaryCertificateLimits {
            max_relation_entries: 1 << 25,
            max_distance_candidates: 1 << 27,
            max_distance_weight: 7,
        })?;
        ensure(
            proof.transmitted_logicals == 1,
            "colored surface logical channel rank changed",
        )?;
        ensure(
            proof
                .codes
                .iter()
                .all(|c| c.x_distance.distance == d && c.z_distance.distance == d),
            "colored surface end-code distance failed",
        )?;
        let logical_choi_generators = proof
            .logical_observables
            .iter()
            .map(|p| {
                let mut label = vec!['I'; proof.codes.len()];
                for &(i, p) in p.terms() {
                    label[i] = match p {
                        Pauli::X => 'X',
                        Pauli::Y => 'Y',
                        Pauli::Z => 'Z',
                    };
                }
                label.into_iter().collect()
            })
            .collect();
        result.push(SurfaceRun {
            distance_label: d,
            resource_layers: network.rounds(),
            merge_window: network.merge_window(),
            source_rings: network.resource_centers().len(),
            source_photons: network.patch().program().num_photons(),
            destructive_fusions: network.patch().program().fusions().len(),
            terminal_measurements: network.patch().boundary_measurements().len(),
            declared_h_frame_photons: network.hadamard_frame().len(),
            end_xz_distances: proof
                .codes
                .iter()
                .map(|c| [c.x_distance.distance, c.z_distance.distance])
                .collect(),
            logical_choi_generators,
            logical_lift_negative: proof.logical_phase_negative,
            transmitted_logicals: proof.transmitted_logicals,
        });
    }
    Ok(result)
}
#[derive(Serialize)]
struct Run {
    #[serde(flatten)]
    report: Report,
    distance_runs: Vec<DistanceRun>,
    boundary_networks: Vec<BoundaryRun>,
    color_preserving_surface_networks: Vec<SurfaceRun>,
    sampling_contract: &'static str,
}
fn p(terms: &[(usize, Pauli)]) -> Result<PauliProduct> {
    PauliProduct::new(terms.to_vec())
}
fn native_resource_oracle() -> ExampleResult<f64> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut ring = Circuit::new(Register::qubits(6)?, 0, 0);
    for q in 0..6 {
        ring.unitary(vec![q], Unitary::h())?;
    }
    for q in 0..6 {
        ring.unitary(vec![q, (q + 1) % 6], Unitary::z().controlled(2, 1)?)?;
    }
    let initial = engine.execute(&ring, None, None, 1, &device)?;
    let mut expected = vec![0f32; 16];
    for a in 0..2 {
        for b in 0..2 {
            expected[12 * a + 2 * (a ^ b) + b] = 0.5;
        }
    }
    let ideal = StateVector::from_amplitudes(
        Register::qubits(4)?,
        ComplexTensor::new(
            Tensor::from_data(TensorData::new(expected, [1, 16]), &device),
            Tensor::zeros([1, 16], (&device, DType::F32)),
        )?,
        engine.limits,
    )?;
    let mut error = 0f64;
    for trim in 0..4 {
        let indices = Tensor::<1, Int>::from_data(
            TensorData::new(
                (0..16).map(|i| (4 * i + trim) as i64).collect::<Vec<_>>(),
                [16],
            ),
            &device,
        );
        let branch = StateVector::from_amplitudes(
            Register::qubits(4)?,
            ComplexTensor::new(
                initial.amplitudes().real().select(1, indices.clone()) * 2.,
                initial.amplitudes().imag().select(1, indices) * 2.,
            )?,
            engine.limits,
        )?;
        let mut corrections = Circuit::new(Register::qubits(4)?, 0, 0);
        if trim & 2 != 0 {
            corrections.unitary(vec![3], Unitary::z())?;
        }
        if trim & 1 != 0 {
            corrections.unitary(vec![0], Unitary::z())?;
        }
        corrections
            .unitary(vec![0], Unitary::h())?
            .unitary(vec![2], Unitary::h())?;
        let output = corrections.run_from(&engine, branch, None, None)?;
        output.validate(3e-6)?;
        error = error.max((values(output.fidelity(&ideal)?)?[0] - 1.).abs());
    }
    ensure(error < 3e-6, "native six-ring Choi oracle failed")?;
    Ok(error)
}
fn encoded_inputs(surgery: &PlanarSurgery) -> Result<Tableau> {
    let n = surgery.code().qubits();
    let mut state = Tableau::zero(2 * n + 9, Default::default())?;
    for block in 0..2 {
        state.h(2 * n + block)?;
        state.cnot(2 * n + block, block * n + surgery.code().input_site())?;
        surgery.code().encode(&mut state, block * n)?;
    }
    Ok(state)
}
fn run() -> ExampleResult<Run> {
    let error = native_resource_oracle()?;
    let boundary_networks = boundary_runs()?;
    let color_preserving_surface_networks = surface_runs()?;
    let mut distance_runs = vec![];
    let mut aborted = 0;
    let mut physical_rings = 0;
    let mut physical_fusions = 0;
    for d in [3, 5, 7] {
        let surgery = PlanarSurgery::new(d, MergeBasis::ZZ, Default::default())?;
        let plan = SixRingSurgery::compile(&surgery, Default::default())?;
        let n = surgery.code().qubits();
        let workspace = std::array::from_fn(|i| 2 * n + 2 + i);
        let initial = encoded_inputs(&surgery)?;
        let mut state = initial.clone();
        let output = plan.execute(
            &mut state,
            [0, n],
            workspace,
            SixRingCheckInstrument::IDEAL,
            SixRingSampling::ConditionedReference,
            177 + d as u64,
        )?;
        let parity = match output.logical? {
            SurgeryAttempt::Complete(record) => record.parity,
            SurgeryAttempt::Aborted(_) => return Err("conditioned ideal surgery aborted".into()),
        };
        let za = surgery.code().logical_z(0)?;
        let zb = surgery.code().logical_z(n)?;
        let xa = surgery.code().logical_x(0)?;
        let xb = surgery.code().logical_x(n)?;
        let mut witnesses = surgery
            .split_checks()
            .iter()
            .cloned()
            .map(|x| (x, 1))
            .collect::<Vec<_>>();
        for (logical, reference) in [(&za, 2 * n), (&zb, 2 * n + 1)] {
            witnesses.push((
                logical.multiply_commuting(&p(&[(reference, Pauli::Z)])?)?.1,
                1,
            ));
        }
        witnesses.push((
            xa.multiply_commuting(&xb)?
                .1
                .multiply_commuting(&p(&[(2 * n, Pauli::X), (2 * n + 1, Pauli::X)])?)?
                .1,
            1,
        ));
        witnesses.push((za.multiply_commuting(&zb)?.1, if parity { -1 } else { 1 }));
        let wrong = witnesses
            .iter()
            .map(|(p, expected)| Ok(usize::from(state.expectation(p)? != *expected)))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum();
        ensure(
            wrong == 0,
            "photonic surgery failed complete encoded Choi witnesses",
        )?;
        let cert = surgery.distance_certificate();
        ensure(
            cert.x_distance >= d && cert.z_distance >= d && cert.temporal_repetitions == d,
            "surgery distance certificate failed",
        )?;
        ensure(
            output.cost.rings_emitted == plan.scheduled_rings(),
            "lowering omitted resources",
        )?;
        distance_runs.push(DistanceRun {
            distance: d,
            protected_x_distance: cert.x_distance,
            protected_z_distance: cert.z_distance,
            repeated_rounds: cert.temporal_repetitions,
            source_rings: output.cost.rings_emitted,
            source_photons: output.cost.resource_photons,
            check_preparations: output.cost.syndrome_photons,
            destructive_fusions: output.cost.fusions,
            log_ideal_herald_probability: plan.ideal_log_herald_probability(),
            conditioned_channel_witness_error: wrong,
        });
        if d == 3 {
            for seed in 0..128 {
                let mut state = initial.clone();
                let shot = plan.execute(
                    &mut state,
                    [0, n],
                    workspace,
                    SixRingCheckInstrument::IDEAL,
                    SixRingSampling::Physical,
                    seed,
                )?;
                physical_rings += shot.cost.ring_attempts;
                physical_fusions += shot.cost.fusions;
                if matches!(shot.logical?, SurgeryAttempt::Aborted(_)) {
                    aborted += 1;
                }
            }
        }
    }
    ensure(
        aborted == 128,
        "unexpected all-success physical surgery path; inspect the transcript",
    )?;
    ensure(
        distance_runs.windows(2).all(|p| {
            p[1].source_rings > p[0].source_rings
                && p[1].log_ideal_herald_probability < p[0].log_ideal_herald_probability
        }),
        "negative herald scaling was hidden",
    )?;
    Ok(Run {
        report: Report::new(
            "fbqc_lattice_surgery",
            177,
            0,
            &[
                ("native_choi_fidelity_error", error),
                ("distances", 3.),
                ("physical_attempts", 128.),
                ("physical_aborted_attempts", aborted as f64),
                ("physical_ring_attempts", physical_rings as f64),
                ("physical_fusions", physical_fusions as f64),
            ],
        )?,
        distance_runs,
        boundary_networks,
        color_preserving_surface_networks,
        sampling_contract: "Boundary networks derive signed ideal six-ring Choi relations and finite end-code distances, not noisy spacetime distance or threshold. Planar distance/channel witnesses use explicitly conditioned ideal gate-teleportation six-ring branches; 128 separate unconditioned attempts retain rejection. Exponential standard-fusion yield is reported, not claimed fault-tolerant. Initial encoded inputs are supplied.",
    })
}
fn main() -> ExampleResult<()> {
    println!("{}", serde_json::to_string_pretty(&run()?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn physical_lowering_preserves_logical_channel_and_reports_bad_scaling() {
        super::run().unwrap();
    }
}
