//! Amplitude amplification of an arbitrary coherent preparation and a reversible
//! Boolean predicate. The generated circuits use the ordinary Burn register, gates
//! and bindings. Wire zero is most significant and the last wire is a clean flag.
//!
//! [General amplitude amplification](https://arxiv.org/abs/quant-ph/0005055),
//! [fixed-point search](https://arxiv.org/abs/1409.3305) and
//! [exponential search](https://arxiv.org/abs/quant-ph/9605034).
//!
//! Amplify two labels in a qubit–qutrit preparation. The spare flag is wire 2.
//! ```
//! use burn::tensor::Device;
//! use burn_quantum::{Circuit, DenseSimulator, Register, Unitary};
//! use burn_quantum::algorithms::amplification::{AmplitudeAmplifier, AmplificationSchedule};
//!
//! let mut a = Circuit::new(Register::new(vec![2, 3])?, 0, 0);
//! a.unitary(vec![0], Unitary::h())?.unitary(vec![1], Unitary::fourier(3)?)?;
//! let amplifier = AmplitudeAmplifier::from_marked(a, &[1, 5])?;
//! let plan = AmplificationSchedule::fixed_point(0.2, 0.01)?;
//! let circuit = amplifier.compile(plan.phases(), 256)?;
//! let state = circuit.run(&DenseSimulator::default(), None, None, 1, &Device::flex())?;
//! let p = state.probabilities().into_data().try_to_vec::<f32>().unwrap();
//! assert!(p[2] + p[10] >= 0.99 - 1e-5);
//! # Ok::<(), burn_quantum::Error>(())
//! ```
mod parity;
mod schedule;
mod search;

pub use parity::OddParityExperiment;
pub use schedule::{AmplificationSchedule, ReflectionPhases};
pub use search::{ExponentialSearchResult, SearchConfig, exponential_search};

use crate::{Circuit, Error, Register, Result, Samples, StateVector, Unitary};
use std::collections::BTreeSet;

/// A coherent preparation A on data wires and a reversible bit oracle on data plus one flag.
/// Each round implements `A S_zero(source) A† O† P_flag(good) O` in matrix order.
/// With pi phases this differs from the conventional Grover iterate by a global minus sign.
#[derive(Clone, Debug)]
pub struct AmplitudeAmplifier {
    preparation: Circuit,
    oracle: Circuit,
}

impl AmplitudeAmplifier {
    /// Use a measurement-free preparation, which may have parameters and batched inputs.
    /// The oracle must compute the predicate into the last qubit when started
    /// at flag zero. Its adjoint uncomputes any reversible temporary work. The caller
    /// supplies the Boolean semantics at every binding used. Both circuits share
    /// a parameter and input namespace. Equal indices refer to the same value.
    pub fn new(preparation: Circuit, oracle: Circuit) -> Result<Self> {
        if oracle.register() != &Self::oracle_register(preparation.register())? {
            return Err(Error::Shape(
                "oracle requires the preparation register followed by one flag qubit".into(),
            ));
        }
        Ok(Self {
            preparation,
            oracle,
        })
    }

    /// Append the clean qubit for an oracle acting on arbitrary mixed-radix data.
    pub fn oracle_register(data: &Register) -> Result<Register> {
        let mut dimensions = data.dimensions().to_vec();
        dimensions.push(2);
        Register::new(dimensions)
    }

    /// Convenience oracle for a list of distinct, big-endian flat data basis indices.
    /// Empty and fully marked sets are valid. For structured problems, pass a predicate
    /// circuit to [Self::new] to avoid enumerating marked labels.
    pub fn from_marked(preparation: Circuit, marked: &[usize]) -> Result<Self> {
        let data = preparation.register();
        let mut oracle = Circuit::new(Self::oracle_register(data)?, 0, 0);
        let mut seen = BTreeSet::new();
        for &index in marked {
            let digits = data.decode(index)?;
            if !seen.insert(index) {
                return Err(Error::InvalidValue("duplicate marked basis index".into()));
            }
            oracle.controlled_x(digits.into_iter().enumerate().collect(), data.wires())?;
        }
        Self::new(preparation, oracle)
    }

    /// Unamplified data preparation with its original bindings.
    pub fn preparation(&self) -> &Circuit {
        &self.preparation
    }

    /// Actual reversible marking computation. The flag is the least significant bit.
    pub fn oracle(&self) -> &Circuit {
        &self.oracle
    }

    /// Number of ordinary circuit instructions including preparation, both oracle
    /// directions, both preparation directions and all selective-phase gates.
    pub fn operation_count(&self, rounds: usize) -> Result<usize> {
        let prepare = self.preparation.operations().len();
        let oracle = self.oracle.operations().len();
        prepare
            .checked_add(oracle)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(rounds))
            .and_then(|n| n.checked_add(prepare))
            .ok_or(Error::Overflow)
    }

    /// Search without supplying a success probability. The executor binds and runs
    /// one fresh, single-row circuit per trial on the caller's chosen device.
    /// Sampling uses synchronous terminal readback. The verifier receives a data
    /// basis index with the clean flag removed and must check the same predicate
    /// as the bit oracle. The operation budget applies per trial.
    pub fn search(
        &self,
        config: SearchConfig,
        max_operations: usize,
        mut execute: impl FnMut(&Circuit) -> Result<StateVector>,
        verify: impl FnMut(&usize) -> Result<bool>,
    ) -> Result<ExponentialSearchResult<usize>> {
        exponential_search(
            None,
            config,
            |rounds, seed| {
                let circuit = self.compile(
                    AmplificationSchedule::standard(rounds).phases(),
                    max_operations,
                )?;
                let state = execute(&circuit)?;
                if state.register() != circuit.register() || state.batch_size() != 1 {
                    return Err(Error::Shape(
                        "search trial requires one state on the data and flag register".into(),
                    ));
                }
                let outcome = Samples::from_state(&state, 1, seed, 1)?.outcomes[0][0];
                if outcome % 2 != 0 {
                    return Err(Error::InvalidValue(
                        "sampled oracle flag was not uncomputed".into(),
                    ));
                }
                Ok(outcome / 2)
            },
            verify,
        )
    }

    /// Compile a bounded reusable circuit from a schedule or caller-supplied phase
    /// pairs in execution order. No full-register matrix is constructed. Flat output
    /// index `2*x + flag` contains data label x and the restored flag.
    pub fn compile(
        &self,
        phases: impl ExactSizeIterator<Item = ReflectionPhases>,
        max_operations: usize,
    ) -> Result<Circuit> {
        let required = self.operation_count(phases.len())?;
        if required > max_operations {
            return Err(Error::ResourceLimit {
                resource: "amplification circuit operations",
                required,
                limit: max_operations,
            });
        }
        let flag = self.preparation.register().wires();
        let mut circuit = Circuit::new(
            self.oracle.register().clone(),
            self.preparation
                .num_parameters()
                .max(self.oracle.num_parameters()),
            self.preparation.num_inputs().max(self.oracle.num_inputs()),
        );
        let append = |output: &mut Circuit, block: &Circuit| -> Result<()> {
            for op in block.operations() {
                output.push(op.clone())?;
            }
            Ok(())
        };
        append(&mut circuit, &self.preparation)?;
        if phases.len() == 0 {
            return Ok(circuit);
        }
        let undo_preparation = self.preparation.adjoint();
        let undo_oracle = self.oracle.adjoint();
        let zero: Vec<_> = (0..flag).map(|wire| (wire, 0)).collect();
        for pair in phases {
            append(&mut circuit, &self.oracle)?;
            circuit.unitary(vec![flag], Unitary::phase(pair.good)?)?;
            append(&mut circuit, &undo_oracle)?;
            append(&mut circuit, &undo_preparation)?;
            circuit.controlled_x(zero.clone(), flag)?;
            circuit.unitary(vec![flag], Unitary::phase(pair.source)?)?;
            circuit.controlled_x(zero.clone(), flag)?;
            append(&mut circuit, &self.preparation)?;
        }
        Ok(circuit)
    }
}
