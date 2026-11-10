//! Verified retries on supplied parity-encoded data, never on destroyed bare inputs.
use super::{HeraldedGatePlan, TeleportationPorts};
use crate::photonic::{CompileOptions, DualRail, Fabric, FusionOutcome, RailTransmission};
use crate::{
    ComplexTensor, DenseSimulator, DensityMatrix, Error, Matrix, Register, Result, StateVector,
    Unitary,
};
use burn::tensor::{Int, Tensor, TensorData};
use num_complex::Complex64 as C;
use serde::Serialize;
/// A terminal retry history, including its absolute probability and retained reference.
#[derive(Clone, Debug)]
pub struct ParityRetryBranch {
    /// Standard fusion records in actual attempted order; only known Z failures retry.
    pub outcomes: Vec<FusionOutcome>,
    /// Successful decoding's known Z outcomes, in surviving-code-site order.
    pub decode_bits: Vec<bool>,
    /// True iff a resolved teleportation and final parity decoding completed.
    pub success: bool,
    /// Absolute branch probability, with no postselection.
    pub probability: Tensor<1>,
    /// On success: logical output followed by original references. On failure: only
    /// references (None if absent), after discarding all unusable encoded data.
    pub state: Option<DensityMatrix>,
    /// End of this bounded branch including classical feedback and final decoding.
    pub finish_tick: u64,
}
/// Complete finite parity-code retry instrument with explicit physical resource costs.
#[derive(Clone, Debug)]
pub struct ParityRetryRun {
    branches: Vec<ParityRetryBranch>,
    resource_photons: Tensor<1>,
    fusions: Tensor<1>,
}
impl ParityRetryRun {
    /// All terminal success, exhausted-failure and loss histories.
    pub fn branches(&self) -> &[ParityRetryBranch] {
        &self.branches
    }
    /// Expected supplied two-photon gate resources; encoding photons are separate inputs.
    pub fn expected_resource_photons(&self) -> Tensor<1> {
        self.resource_photons.clone()
    }
    /// Expected destructive fusions per encoded input attempt.
    pub fn expected_fusions(&self) -> Tensor<1> {
        self.fusions.clone()
    }
    /// Complete accepted weight; output loss can still make accepted logical data unusable.
    pub fn herald_probability(&self) -> Tensor<1> {
        let mut sum = self.fusions.clone() * 0.;
        for branch in &self.branches {
            if branch.success {
                sum = sum + branch.probability.clone();
            }
        }
        sum
    }
}
/// Bounded retry protocol supplied with a parity encoding of one unknown logical input.
///
/// |0_L> is the uniform even-parity superposition and |1_L> the odd one. A known
/// Z measurement removes one photon and preserves the logical state up to X. A standard
/// fusion's `Failure{bit}` therefore permits another disposable port to be tried, after
/// its known correction. `Loss` always aborts. Success teleports one physical port;
/// Z measurements of the remaining encoding followed by U X^parity U† decode the output.
///
/// This is a verified input-encoding assumption, not a deterministic optical encoder.
/// Resource states and final local Z/correction operations are supplied ideally. All
/// routed/storage/detector attenuation in the underlying gate plan is retained.
#[derive(Clone, Debug, Serialize)]
pub struct ParityRetryPlan {
    gate: HeraldedGatePlan,
    photons: usize,
    feedback: u64,
    cycle: u64,
}
impl ParityRetryPlan {
    /// Compile two or three supplied encoding photons and fresh resources on each retry.
    /// Sequential rounds reuse routes only after previous consumption and feedback.
    pub fn compile(
        gate: Unitary,
        photons: usize,
        fabric: &Fabric,
        ports: TeleportationPorts,
        feedback_ticks: u64,
        options: CompileOptions,
    ) -> Result<Self> {
        if !(2..=3).contains(&photons) {
            return Err(Error::Unsupported(
                "exact parity retry supports two or three encoded photons".into(),
            ));
        }
        let required = photons.checked_mul(3).ok_or(Error::Overflow)?;
        if required > options.max_photons {
            return Err(Error::ResourceLimit {
                resource: "parity retry photons",
                required,
                limit: options.max_photons,
            });
        }
        if photons > options.max_fusions {
            return Err(Error::ResourceLimit {
                resource: "parity retry fusions",
                required: photons,
                limit: options.max_fusions,
            });
        }
        let gate = HeraldedGatePlan::compile(gate, fabric, ports, options)?;
        let cycle = gate
            .finish_tick()
            .checked_add(feedback_ticks)
            .and_then(|x| x.checked_add(1))
            .ok_or(Error::Overflow)?;
        if cycle.checked_mul(photons as u64).ok_or(Error::Overflow)? > options.max_ticks {
            return Err(Error::Unsupported("parity retry exceeds horizon".into()));
        }
        Ok(Self {
            gate,
            photons,
            feedback: feedback_ticks,
            cycle,
        })
    }
    /// Number of supplied encoded data photons (not freshly generated retry resources).
    pub fn encoding_photons(&self) -> usize {
        self.photons
    }
    /// Execute on a supplied parity encoding of the first logical input qubit.
    /// The native isometry explicitly constructs that supplied encoding; all remaining
    /// input qubits are untouched entangled references. Branches prove the whole channel.
    pub fn execute_with_supplied_encoding(
        &self,
        engine: &DenseSimulator,
        input: StateVector,
    ) -> Result<ParityRetryRun> {
        if input.register().dimensions().iter().any(|&d| d != 2) {
            return Err(Error::Unsupported(
                "parity retry input requires qubits".into(),
            ));
        }
        let references = input.register().wires() - 1;
        let width = self
            .photons
            .checked_add(references)
            .ok_or(Error::Overflow)?;
        let size = 1usize
            .checked_shl(u32::try_from(width).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        engine.limits.check_state(input.batch_size(), size)?;
        let indices: Vec<i64> = (0..size)
            .map(|i| {
                let code = i >> references;
                let parity = code.count_ones() as usize % 2;
                ((parity << references) | (i & ((1 << references) - 1))) as i64
            })
            .collect();
        let dev = input.device();
        let dtype = input.amplitudes().dtype();
        let batch = input.batch_size();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(indices, [size]), &dev);
        let scale = (1usize << (self.photons - 1)) as f64;
        let encoded = StateVector::from_amplitudes(
            Register::qubits(self.photons + references)?,
            ComplexTensor::new(
                input.amplitudes().real().select(1, idx.clone()) / scale.sqrt(),
                input.amplitudes().imag().select(1, idx) / scale.sqrt(),
            )?,
            engine.limits,
        )?;
        let encoded = super::physical::execution::encode_state(engine, encoded)?;
        let mut frontier = vec![(
            DensityMatrix::from_pure(&encoded, engine.limits)?,
            Vec::<FusionOutcome>::new(),
        )];
        let mut branches = vec![];
        let mut fusions = Tensor::zeros([batch], (&dev, dtype));
        let storage = self
            .gate
            .fabric()
            .node(self.gate.ports().input)?
            .storage
            .over_ticks(self.cycle);
        let output_storage = self
            .gate
            .fabric()
            .node(self.gate.ports().resource)?
            .storage
            .over_ticks(self.feedback + 1);
        for attempt in 0..self.photons {
            let remaining = self.photons - attempt;
            let mut next = vec![];
            for (mut state, history) in frontier {
                fusions = fusions + state.trace();
                for site in 1..remaining {
                    if storage != RailTransmission::IDEAL {
                        state = state.channel(engine, &[site], &storage.channel()?)?;
                    }
                }
                for outcome in self.gate.execute(engine, &state, 0)? {
                    let record = outcome.outcomes()[0];
                    let mut history = history.clone();
                    history.push(record);
                    let state = outcome.unnormalized_state().unwrap();
                    match record {
                        FusionOutcome::Failure { bit } if remaining > 1 => {
                            let keep: Vec<_> = (1..state.register().wires()).collect();
                            let mut state = state.reduced(&keep, engine.limits)?;
                            if bit {
                                state = state.apply(
                                    engine,
                                    &[0],
                                    &DualRail::unitary(&Unitary::x())?,
                                )?;
                            }
                            next.push((state, history));
                        }
                        FusionOutcome::Success { xx } => {
                            // The primitive applies its zero-latency Bell frame. Undo it
                            // here so both Bell and decode corrections follow real storage
                            // during this protocol's declared feedback latency.
                            let pauli = if xx == crate::photonic::Parity::Even {
                                Unitary::x()
                            } else {
                                Unitary::y()
                            };
                            let bell_correction =
                                DualRail::unitary(&conjugate(self.gate.gate(), &pauli)?)?;
                            let uncorrected = state.apply(engine, &[0], &bell_correction)?;
                            for bits in 0..1usize << (remaining - 1) {
                                let mut decoded = uncorrected.clone();
                                let mut decode_bits = vec![];
                                for site in 1..remaining {
                                    let bit = (bits >> (site - 1)) & 1;
                                    decode_bits.push(bit != 0);
                                    let mut diag = vec![0.; 9];
                                    diag[bit * 3 + bit] = 1.;
                                    decoded = decoded.left_right(
                                        engine,
                                        &[site],
                                        &Matrix::real(3, diag)?,
                                    )?;
                                }
                                let mut keep = vec![0];
                                keep.extend(remaining..remaining + references);
                                decoded = decoded.reduced(&keep, engine.limits)?;
                                if output_storage != RailTransmission::IDEAL {
                                    decoded = decoded.channel(
                                        engine,
                                        &[0],
                                        &output_storage.channel()?,
                                    )?;
                                }
                                decoded = decoded.apply(engine, &[0], &bell_correction)?;
                                if bits.count_ones() % 2 == 1 {
                                    decoded = decoded.apply(
                                        engine,
                                        &[0],
                                        &DualRail::unitary(&conjugate(
                                            self.gate.gate(),
                                            &Unitary::x(),
                                        )?)?,
                                    )?;
                                }
                                branches.push(ParityRetryBranch {
                                    outcomes: history.clone(),
                                    decode_bits,
                                    success: true,
                                    probability: decoded.trace(),
                                    state: Some(decoded),
                                    finish_tick: (attempt as u64 + 1) * self.cycle,
                                });
                            }
                            // Vacuum among the remaining code sites prevents a valid Z decoding.
                            // Retain that weight explicitly, never manufacture a known measurement bit.
                            // State of failed decoding references is obtained by the complementary projector.
                            let mut erasure = vec![0.; 3usize.pow(remaining as u32).pow(2)];
                            let n = 3usize.pow(remaining as u32);
                            for i in 0..n {
                                if DualRail::register(remaining)?.decode(i)?[1..].contains(&2) {
                                    erasure[i * n + i] = 1.;
                                }
                            }
                            let rejected = state.left_right(
                                engine,
                                &(0..remaining).collect::<Vec<_>>(),
                                &Matrix::real(n, erasure)?,
                            )?;
                            let retained = if references == 0 {
                                None
                            } else {
                                Some(rejected.reduced(
                                    &(remaining..remaining + references).collect::<Vec<_>>(),
                                    engine.limits,
                                )?)
                            };
                            branches.push(ParityRetryBranch {
                                outcomes: history,
                                decode_bits: vec![],
                                success: false,
                                probability: rejected.trace(),
                                state: retained,
                                finish_tick: (attempt as u64 + 1) * self.cycle,
                            });
                        }
                        _ => {
                            let retained = if references == 0 {
                                None
                            } else {
                                Some(state.reduced(
                                    &(remaining..remaining + references).collect::<Vec<_>>(),
                                    engine.limits,
                                )?)
                            };
                            branches.push(ParityRetryBranch {
                                outcomes: history,
                                decode_bits: vec![],
                                success: false,
                                probability: outcome.probability(),
                                state: retained,
                                finish_tick: (attempt as u64 + 1) * self.cycle,
                            });
                        }
                    }
                }
            }
            frontier = next;
        }
        Ok(ParityRetryRun {
            branches,
            resource_photons: fusions.clone() * 2.,
            fusions,
        })
    }
}
fn conjugate(u: &Unitary, p: &Unitary) -> Result<Unitary> {
    let m = u.matrix().entries();
    let p = p.matrix().entries();
    let mut out = vec![C::new(0., 0.); 4];
    for r in 0..2 {
        for c in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    out[r * 2 + c] += m[r * 2 + a] * p[a * 2 + b] * m[c * 2 + b].conj();
                }
            }
        }
    }
    Unitary::new(Matrix::new(2, out)?)
}
