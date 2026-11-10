//! Bounded T injection and the [[15,1,3]] Reed–Muller distillation protocol.
//!
//! Raw resources are independent Z-noisy T|+> states. Clifford operations are
//! ideal; four syndrome readouts may flip. This is not a circuit-level threshold
//! model. See Bravyi–Haah, arXiv:1209.2426, Sec. V and Fig. 2.
use crate::{
    Circuit, ComplexTensor, DenseSimulator, DensityMatrix, Error, Matrix, Register, ResourceLimits,
    Result, Unitary,
};
use burn::tensor::{DType, Device};
use num_complex::Complex64;
use serde::Serialize;

/// An independently prepared `(1-p)|A><A| + p Z|A><A|Z`, |A> = T|+>.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MagicState {
    phase_error: f64,
    availability: f64,
}
impl MagicState {
    /// Availability is a heralded source yield, separate from conditional fidelity.
    pub fn new(phase_error: f64, availability: f64) -> Result<Self> {
        probability(phase_error)?;
        probability(availability)?;
        Ok(Self {
            phase_error,
            availability,
        })
    }
    /// Conditional phase-error probability.
    pub fn phase_error(&self) -> f64 {
        self.phase_error
    }
    /// Probability that a source attempt supplies the resource.
    pub fn availability(&self) -> f64 {
        self.availability
    }
    /// Conditional resource density on the selected device.
    pub fn density(
        &self,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<DensityMatrix> {
        limits.check_state(1, 4)?;
        limits.check_matrix(1, 2)?;
        let c = (1. - 2. * self.phase_error) / (2. * 2f64.sqrt());
        let entries = Matrix::new(
            2,
            vec![
                Complex64::new(0.5, 0.),
                Complex64::new(c, -c),
                Complex64::new(c, c),
                Complex64::new(0.5, 0.),
            ],
        )?
        .to_tensor(device, dtype)?;
        DensityMatrix::from_entries(Register::qubits(1)?, entries, limits)
    }
    /// Append the resource, CNOT(data,ancilla), measure ancilla Z, apply S on
    /// outcome one, and trace it out. Every branch is subnormalized; the missing
    /// branch preserves the input. All data and entangled-reference gradients
    /// remain native. The input must be a physical density matrix of qubits.
    pub fn inject(
        &self,
        input: &DensityMatrix,
        target: usize,
        limits: ResourceLimits,
    ) -> Result<MagicInjection> {
        if input.register().dimensions().iter().any(|&d| d != 2)
            || target >= input.register().wires()
        {
            return Err(Error::InvalidDimensions(
                "T injection requires a valid qubit target".into(),
            ));
        }
        let n = input.register().size();
        let b = input.batch_size();
        let n2 = n
            .checked_mul(n)
            .and_then(|v| v.checked_mul(4))
            .ok_or(Error::Overflow)?;
        limits.check_state(b, n2)?;
        let anc = self.density(&input.entries().device(), input.entries().dtype(), limits)?;
        let a = input.entries().real().reshape([b, n, 1, n, 1]);
        let z = input.entries().imag().reshape([b, n, 1, n, 1]);
        let u = anc.entries().real().reshape([1, 1, 2, 1, 2]);
        let v = anc.entries().imag().reshape([1, 1, 2, 1, 2]);
        let appended = DensityMatrix::from_entries(
            Register::qubits(input.register().wires() + 1)?,
            ComplexTensor::new(
                (a.clone() * u.clone() - z.clone() * v.clone()).reshape([b, 2 * n, 2 * n]),
                (a * v + z * u).reshape([b, 2 * n, 2 * n]),
            )?,
            limits,
        )?;
        let engine = DenseSimulator::new(limits);
        let ancilla = input.register().wires();
        let joint = appended.apply(&engine, &[target, ancilla], &Unitary::cnot())?;
        let keep: Vec<_> = (0..ancilla).collect();
        let mut branches = vec![MagicInjectionBranch {
            record: MagicRecord::Missing,
            state: scaled(input, 1. - self.availability, limits)?,
        }];
        for bit in 0..2 {
            let projector = Matrix::real(
                2,
                if bit == 0 {
                    vec![1., 0., 0., 0.]
                } else {
                    vec![0., 0., 0., 1.]
                },
            )?;
            let mut state = joint
                .left_right(&engine, &[ancilla], &projector)?
                .reduced(&keep, limits)?;
            if bit == 1 {
                state = state.apply(&engine, &[target], &Unitary::s())?;
            }
            branches.push(MagicInjectionBranch {
                record: if bit == 0 {
                    MagicRecord::Zero
                } else {
                    MagicRecord::One
                },
                state: scaled(&state, self.availability, limits)?,
            });
        }
        Ok(MagicInjection {
            branches,
            raw_source_attempts: 1,
        })
    }
}
/// Complete herald and ancilla measurement records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MagicRecord {
    /// Resource unavailable: no gate executed.
    Missing,
    /// Z ancilla outcome zero.
    Zero,
    /// Z ancilla outcome one, followed by S correction.
    One,
}
/// A complete instrument branch; trace is its absolute probability per batch.
#[derive(Clone, Debug)]
pub struct MagicInjectionBranch {
    /// Classical record.
    pub record: MagicRecord,
    /// Subnormalized remaining data and references.
    pub state: DensityMatrix,
}
/// Result of one source attempt and conditional native injection.
#[derive(Clone, Debug)]
pub struct MagicInjection {
    /// Missing, zero and one branches, without postselection.
    pub branches: Vec<MagicInjectionBranch>,
    /// Always one, including missing resources.
    pub raw_source_attempts: usize,
}
impl MagicInjection {
    /// Sum only the two completed-gate branches, retaining their absolute weight.
    pub fn completed(&self, limits: ResourceLimits) -> Result<DensityMatrix> {
        if self.branches.len() != 3
            || self.branches[0].record != MagicRecord::Missing
            || self.branches[1].record != MagicRecord::Zero
            || self.branches[2].record != MagicRecord::One
            || self.branches[1].state.register() != self.branches[2].state.register()
            || self.branches[1].state.entries().dims() != self.branches[2].state.entries().dims()
        {
            return Err(Error::InvalidValue(
                "incomplete magic injection record".into(),
            ));
        }
        let a = &self.branches[1].state;
        let b = &self.branches[2].state;
        crate::complex::compatible(&a.entries().real(), &b.entries().real())?;
        DensityMatrix::from_entries(
            a.register().clone(),
            ComplexTensor::new(
                a.entries().real() + b.entries().real(),
                a.entries().imag() + b.entries().imag(),
            )?,
            limits,
        )
    }
}
fn scaled(state: &DensityMatrix, p: f64, limits: ResourceLimits) -> Result<DensityMatrix> {
    DensityMatrix::from_entries(
        state.register().clone(),
        ComplexTensor::new(state.entries().real() * p, state.entries().imag() * p)?,
        limits,
    )
}
fn probability(p: f64) -> Result<()> {
    if p.is_finite() && (0. ..=1.).contains(&p) {
        Ok(())
    } else {
        Err(Error::InvalidValue(
            "magic-state probability outside [0,1]".into(),
        ))
    }
}

/// Independent raw phase faults, heralded availability, and classical readout flips.
#[derive(Clone, Debug, Serialize)]
pub struct MagicDistillationConfig {
    /// Probability of Z after each physical T injection.
    pub phase_errors: [f64; 15],
    /// All fifteen sources are attempted once, before starting the code circuit.
    pub source_availability: [f64; 15],
    /// Independent flip of each of the four measured X-stabilizer bits.
    pub syndrome_readout_errors: [f64; 4],
    /// Exhaustive enumeration must permit all 32768 physical fault masks.
    pub max_fault_patterns: usize,
}
impl Default for MagicDistillationConfig {
    fn default() -> Self {
        Self {
            phase_errors: [0.; 15],
            source_availability: [1.; 15],
            syndrome_readout_errors: [0.; 4],
            max_fault_patterns: 1 << 15,
        }
    }
}
/// Explicit cost of the encoded-block implementation, not an optimized factory.
#[derive(Clone, Debug, Serialize)]
pub struct MagicDistillationCost {
    /// Number of independently attempted raw resource sources, even on abort.
    pub raw_source_attempts: usize,
    /// Includes resources emitted but discarded when another source is missing.
    pub expected_raw_states_emitted: f64,
    /// All fifteen initialized code carriers, including the initial logical |+>.
    /// These are separate from raw magic-resource carriers and syndrome ancillas.
    pub expected_code_carriers_prepared: f64,
    /// Additional |0> code ancillas (14) and syndrome ancillas (4), if sources ready.
    pub expected_clifford_ancillas_prepared: f64,
    /// Fifteen injection and four syndrome measurements when sources are ready.
    pub expected_measurements: f64,
    /// Encoder + 15 injection + 32 syndrome CNOTs, decoder only after acceptance.
    pub expected_cnot_gates: f64,
    /// Five preparation + eight syndrome H gates, four decode H gates if accepted.
    pub expected_hadamards: f64,
    /// Fifteen S† for T† and mean 7.5 outcome-dependent S injection corrections.
    pub expected_phase_cliffords: f64,
    /// Raw source attempts per accepted output under independent repeated trials.
    pub raw_attempts_per_output: Option<f64>,
}
/// Exact host enumeration report, with an explicitly costed native output resource.
#[derive(Clone, Debug, Serialize)]
pub struct MagicDistillation {
    /// All sources available, prior to quantum syndrome rejection.
    pub sources_ready: f64,
    /// Absolute probability of a measured all-zero syndrome and all sources ready.
    pub acceptance: f64,
    /// Absolute probability that faulty readout conceals a nonzero true syndrome.
    pub false_herald_probability: f64,
    /// Conditional output Z-error, absent if acceptance is zero.
    pub output_phase_error: Option<f64>,
    /// True syndrome distribution conditional on sources being ready.
    pub true_syndrome_probabilities: [f64; 16],
    /// Observed syndrome distribution, including raw-source yield (sum = sources_ready).
    pub observed_syndrome_probabilities: [f64; 16],
    /// Number of physical phase-fault patterns explicitly evaluated.
    pub evaluated_fault_patterns: usize,
    /// Elementary counts of this declared implementation.
    pub cost: MagicDistillationCost,
}
impl MagicDistillation {
    /// A resource with its absolute yield and conditional error, ready for injection.
    pub fn resource(&self) -> Result<MagicState> {
        MagicState::new(self.output_phase_error.unwrap_or(0.), self.acceptance)
    }
}
/// The standard `[[15,1,3]]` code: columns are the nonzero four-bit strings;
/// four even X rows select each column bit, and logical X is the all-one row.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReedMuller15;
impl ReedMuller15 {
    /// Bit q represents physical qubit q. All rows have weight eight.
    pub fn x_checks() -> [u16; 4] {
        std::array::from_fn(|bit| {
            (0..15).fold(0, |mask, q| mask | ((((q + 1) >> bit) & 1) as u16) << q)
        })
    }
    /// X on every site. Its overlap with a phase fault gives decoded logical Z.
    pub fn logical_x_mask() -> u16 {
        0x7fff
    }
    /// Exact X-check syndrome and decoded logical phase error for a Z fault mask.
    pub fn phase_fault(mask: u16) -> Result<(u8, bool)> {
        if mask >= 1 << 15 {
            return Err(Error::InvalidValue("15-qubit phase mask".into()));
        }
        let syndrome = Self::x_checks().iter().enumerate().fold(0, |s, (i, r)| {
            s | (((mask & r).count_ones() & 1) as u8) << i
        });
        Ok((syndrome, mask.count_ones() % 2 == 1))
    }
    /// Deterministic Clifford encoder. Input wire zero is the logical qubit;
    /// wires 1..=4 start in |+>, all others in |0>. No hidden state projection.
    pub fn encoding_cnots() -> Vec<(usize, usize)> {
        let mut columns = vec![Self::logical_x_mask()];
        columns.extend(Self::x_checks());
        for q in 0..15 {
            let unit = 1u16 << q;
            let mut candidate = columns.clone();
            candidate.push(unit);
            if rank(&candidate) > columns.len() {
                columns.push(unit);
            }
        }
        let mut rows: Vec<u16> = (0..15)
            .map(|q| {
                columns
                    .iter()
                    .enumerate()
                    .fold(0, |r, (j, c)| r | ((c >> q) & 1) << j)
            })
            .collect();
        let mut operations = vec![];
        for pivot in 0..15 {
            let found = (pivot..15).find(|&r| rows[r] & (1 << pivot) != 0).unwrap();
            if found != pivot {
                for (c, t) in [(pivot, found), (found, pivot), (pivot, found)] {
                    rows[t] ^= rows[c];
                    operations.push((c, t));
                }
            }
            for row in 0..15 {
                if row != pivot && rows[row] & (1 << pivot) != 0 {
                    rows[row] ^= rows[pivot];
                    operations.push((pivot, row));
                }
            }
        }
        operations.reverse();
        operations
    }
    /// Executable ideal-block oracle: prepare encoded |+>, apply transversal T†,
    /// the requested Z pattern, and optionally decode. T† is S† times T and can
    /// therefore be supplied by fifteen noisy raw T-state injections. The four
    /// decoded wires 1..=4 contain the physical X syndrome in the Z basis.
    /// This pure-state circuit is for validation; exhaustive mixtures use `distill`.
    pub fn fault_circuit(mask: u16, decode: bool) -> Result<Circuit> {
        Self::phase_fault(mask)?;
        let mut circuit = Circuit::new(Register::qubits(15)?, 0, 0);
        for q in 0..5 {
            circuit.unitary(vec![q], Unitary::h())?;
        }
        let cnots = Self::encoding_cnots();
        for &(c, t) in &cnots {
            circuit.unitary(vec![c, t], Unitary::cnot())?;
        }
        for q in 0..15 {
            circuit.unitary(vec![q], Unitary::t().adjoint())?;
            if mask & (1 << q) != 0 {
                circuit.unitary(vec![q], Unitary::z())?;
            }
        }
        if decode {
            for &(c, t) in cnots.iter().rev() {
                circuit.unitary(vec![c, t], Unitary::cnot())?;
            }
            for q in 1..5 {
                circuit.unitary(vec![q], Unitary::h())?;
            }
        }
        Ok(circuit)
    }
    /// Exhaust all 2^15 independent phase-fault masks. Acceptance uses the actual
    /// four stabilizer parities and each readout-error pattern; no asymptotic
    /// success/error polynomial is used. Classical enumeration is host F64.
    pub fn distill(config: &MagicDistillationConfig) -> Result<MagicDistillation> {
        for &p in config
            .phase_errors
            .iter()
            .chain(&config.source_availability)
            .chain(&config.syndrome_readout_errors)
        {
            probability(p)?;
        }
        if config.max_fault_patterns < 1 << 15 {
            return Err(Error::ResourceLimit {
                resource: "magic phase fault patterns",
                required: 1 << 15,
                limit: config.max_fault_patterns,
            });
        }
        let mut joint = [[0.; 2]; 16];
        for mask in 0u16..1 << 15 {
            let p = config
                .phase_errors
                .iter()
                .enumerate()
                .map(|(q, &p)| if mask & (1 << q) == 0 { 1. - p } else { p })
                .product::<f64>();
            let (syndrome, logical) = Self::phase_fault(mask)?;
            joint[syndrome as usize][usize::from(logical)] += p;
        }
        let available = config.source_availability.iter().product::<f64>();
        let mut observed = [[0.; 2]; 16];
        let mut false_herald = 0.;
        for (syndrome, weight) in joint.iter().enumerate() {
            for (record, out) in observed.iter_mut().enumerate() {
                let readout = config
                    .syndrome_readout_errors
                    .iter()
                    .enumerate()
                    .map(|(bit, &p)| {
                        if (syndrome ^ record) & (1 << bit) == 0 {
                            1. - p
                        } else {
                            p
                        }
                    })
                    .product::<f64>();
                for logical in 0..2 {
                    out[logical] += available * weight[logical] * readout;
                }
                if record == 0 && syndrome != 0 {
                    false_herald += available * (weight[0] + weight[1]) * readout;
                }
            }
        }
        let accepted = observed[0][0] + observed[0][1];
        let encoder = Self::encoding_cnots().len() as f64;
        Ok(MagicDistillation {
            sources_ready: available,
            acceptance: accepted,
            false_herald_probability: false_herald,
            output_phase_error: (accepted > 0.).then(|| observed[0][1] / accepted),
            true_syndrome_probabilities: joint.map(|v| v[0] + v[1]),
            observed_syndrome_probabilities: observed.map(|v| v[0] + v[1]),
            evaluated_fault_patterns: 1 << 15,
            cost: MagicDistillationCost {
                raw_source_attempts: 15,
                expected_raw_states_emitted: config.source_availability.iter().sum(),
                expected_code_carriers_prepared: available * 15.,
                expected_clifford_ancillas_prepared: available * 18.,
                expected_measurements: available * 19.,
                expected_cnot_gates: available * (encoder + 47.) + accepted * encoder,
                expected_hadamards: available * 13. + accepted * 4.,
                expected_phase_cliffords: available * 22.5,
                raw_attempts_per_output: (accepted > 0.).then(|| 15. / accepted),
            },
        })
    }
}
fn rank(rows: &[u16]) -> usize {
    let mut basis = [0u16; 15];
    let mut rank = 0;
    for &row in rows {
        let mut value = row;
        for bit in (0..15).rev() {
            if value & (1 << bit) != 0 {
                if basis[bit] != 0 {
                    value ^= basis[bit];
                } else {
                    basis[bit] = value;
                    rank += 1;
                    break;
                }
            }
        }
    }
    rank
}
