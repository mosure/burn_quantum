//! Reversible oracle compilation, diffusion, and fixed-iteration execution.
use super::Cnf;
use crate::{
    Circuit, DenseSimulator, Error, Register, ResourceLimits, Result, StateVector, Unitary,
};
use burn::tensor::Device;

/// Reversible CNF oracle and Grover diffusion on the dense Burn simulator.
///
/// Variables occupy the first n wires; each normalized clause adds one clean
/// ancilla. Storage is exponential in n + clause count. A zero-wire constant
/// formula uses one dummy qubit. Limits are checked before constructing circuits.
#[derive(Clone, Debug)]
pub struct GroverSat {
    formula: Cnf,
    engine: DenseSimulator,
    preparation: Circuit,
    oracle: Circuit,
    diffusion: Circuit,
}
impl GroverSat {
    /// Compile clause logic, without evaluating candidate assignments or counting solutions.
    pub fn new(formula: Cnf, limits: ResourceLimits) -> Result<Self> {
        let n = formula.num_variables();
        let total = n
            .checked_add(formula.clauses().len())
            .ok_or(Error::Overflow)?
            .max(1);
        let register = Register::qubits(total)?;
        limits.check_state(1, register.size())?;
        limits.check_plan(register.size())?;
        if register.size() > i32::MAX as usize {
            return Err(Error::Unsupported(
                "dense gather indices exceed signed 32-bit range".into(),
            ));
        }
        if n > 0 {
            limits.check_matrix(1, 2)?;
        }
        let mut preparation = Circuit::new(register.clone(), 0, 0);
        let mut oracle = Circuit::new(register.clone(), 0, 0);
        let mut diffusion = Circuit::new(register, 0, 0);
        for wire in 0..n {
            preparation.unitary(vec![wire], Unitary::h())?;
        }
        let falsifying = |clause: &[i32]| {
            clause
                .iter()
                .map(|&x| (x.unsigned_abs() as usize - 1, usize::from(x < 0)))
                .collect::<Vec<_>>()
        };
        for (i, clause) in formula.clauses().iter().enumerate() {
            // ancilla = 1 XOR all_literals_false = clause_truth.
            oracle.controlled_x(vec![], n + i)?;
            oracle.controlled_x(falsifying(clause), n + i)?;
        }
        oracle.phase_flip((n..n + formula.clauses().len()).map(|w| (w, 1)).collect())?;
        for (i, clause) in formula.clauses().iter().enumerate().rev() {
            oracle.controlled_x(falsifying(clause), n + i)?;
            oracle.controlled_x(vec![], n + i)?;
        }
        for wire in 0..n {
            diffusion.unitary(vec![wire], Unitary::h())?;
        }
        diffusion.phase_flip((0..n).map(|w| (w, 0)).collect())?;
        for wire in 0..n {
            diffusion.unitary(vec![wire], Unitary::h())?;
        }
        // Makes diffusion exactly 2|s><s| - I, not its negative.
        diffusion.phase_flip(vec![])?;
        Ok(Self {
            formula,
            engine: DenseSimulator::new(limits),
            preparation,
            oracle,
            diffusion,
        })
    }
    /// Normalized input formula.
    pub fn formula(&self) -> &Cnf {
        &self.formula
    }
    /// Total simulated qubits, including clean clause ancillas and any dummy wire.
    pub fn num_qubits(&self) -> usize {
        self.oracle.register().wires()
    }
    /// Serializable oracle: |x>|0...0> maps to (-1)^CNF(x)|x>|0...0>.
    /// The ancilla-zero precondition is required for this phase-oracle contract.
    pub fn oracle(&self) -> &Circuit {
        &self.oracle
    }
    /// Reflection about the uniform variable state, acting trivially on clause ancillas.
    pub fn diffusion(&self) -> &Circuit {
        &self.diffusion
    }
    /// Prepare uniform variables and zero-valued clause ancillas.
    pub fn initial_state(&self, device: &Device) -> Result<StateVector> {
        self.preparation.run(&self.engine, None, None, 1, device)
    }
    /// Execute an explicit number of Grover iterations for analysis or a known-count problem.
    /// Iterations stream through a single state; no depth-sized circuit is allocated.
    pub fn amplify(&self, iterations: usize, device: &Device) -> Result<StateVector> {
        self.iterate(self.initial_state(device)?, iterations)
    }
    pub(super) fn iterate(&self, mut state: StateVector, iterations: usize) -> Result<StateVector> {
        for _ in 0..iterations {
            state = self.oracle.run_from(&self.engine, state, None, None)?;
            state = self.diffusion.run_from(&self.engine, state, None, None)?;
        }
        Ok(state)
    }
}
