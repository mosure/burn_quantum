//! Bounded, reproducible exact-3SAT instances for simulator demonstrations.
//!
//! These are illustrative Boolean configuration constraints, not industrial SAT
//! benchmarks. Every clause has three distinct variables. The oracle is compiled
//! from the clauses; exhaustive classical enumeration is a separately identified
//! reference used to check the simulation and explain the small problem.
use super::{Cnf, GroverSat};
use crate::{Circuit, Error, Register, ResourceLimits, Result, Unitary};
use serde::{Deserialize, Serialize};

/// A three-literal CNF, its reversible Grover circuit, and a small classical reference.
#[derive(Clone, Debug)]
pub struct ThreeSatBenchmark {
    formula: Cnf,
    circuit: Circuit,
    labels: Vec<String>,
    satisfying: Vec<usize>,
    oracle_operations: usize,
    diffusion_operations: usize,
}

/// Actual readout and explicit resource/reference costs for the 3SAT benchmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreeSatReadout {
    /// Boolean problem variables; variable one is the most significant problem bit.
    pub variables: usize,
    /// Number of exact-three-literal clauses and clean work qubits.
    pub clauses: usize,
    /// Problem variables plus clause ancillas, all simulated as qubits.
    pub total_qubits: usize,
    /// Dense amplitudes in one simulation state.
    pub basis_states: usize,
    /// Elementary structured operations in preparation and the one Grover round.
    pub circuit_operations: usize,
    /// Operations in one reversible clause oracle, including uncomputation.
    pub oracle_operations: usize,
    /// Operations in the uniform-state reflection.
    pub diffusion_operations: usize,
    /// Actual number of oracle applications; this example deliberately uses one.
    pub oracle_queries: usize,
    /// All satisfying problem indices from an exhaustive classical reference.
    pub satisfying_assignments: Vec<usize>,
    /// Uniform random assignment's success probability.
    pub uniform_success_probability: f64,
    /// Independent amplitude-amplification prediction for one Grover round.
    pub analytic_success_probability: f64,
    /// Actual final probability of any satisfying assignment, marginalizing all ancillas.
    pub success_probability: f64,
    /// Actual final probability that every clause ancilla is zero.
    pub clean_ancilla_probability: f64,
    /// Actual variable-only marginal in big-endian assignment order.
    pub assignment_probabilities: Vec<f64>,
    /// Highest-probability assignment; a simulator readout, not a sampled shot.
    pub peak_assignment: Vec<bool>,
    /// Probability of the highest-probability problem assignment.
    pub peak_probability: f64,
    /// Independently evaluated number of clauses satisfied by that peak assignment.
    pub clauses_satisfied_by_peak: usize,
    /// The peak assignment only when it was classically verified against every clause.
    pub verified_witness: Option<Vec<bool>>,
    /// True only when exhaustive classical enumeration found no satisfying assignment.
    /// Quantum search failure alone never proves UNSAT.
    pub reference_proves_unsatisfiable: bool,
}

impl ThreeSatBenchmark {
    /// Construct an exact-3SAT fixture with 3–6 variables and one amplification round.
    ///
    /// The satisfiable family has 5–8 clauses and uses every variable. The
    /// contradictory family contains all eight signed clauses over x1, x2, x3;
    /// additional declared variables remain free. Enumeration visits only 2^n
    /// classical assignments and is never used to synthesize the quantum oracle.
    pub fn new(variables: usize, contradictory: bool, limits: ResourceLimits) -> Result<Self> {
        if !(3..=6).contains(&variables) {
            return Err(Error::InvalidValue(
                "3SAT benchmark needs 3–6 variables".into(),
            ));
        }
        let clauses = if contradictory {
            (0..8)
                .map(|bits| {
                    (0..3)
                        .map(|i| {
                            let variable = i + 1;
                            if bits & (1 << i) == 0 {
                                variable
                            } else {
                                -variable
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            let clauses: &[[i32; 3]] = match variables {
                3 => &[[-1, 2, 3], [1, -2, -3], [1, -2, 3], [1, 2, -3], [1, 2, 3]],
                4 => &[
                    [-1, -2, 4],
                    [1, -2, -4],
                    [1, -2, 4],
                    [1, 2, -4],
                    [2, -3, 4],
                    [2, 3, 4],
                ],
                5 => &[
                    [-1, -2, -4],
                    [-1, -2, 3],
                    [-1, 4, 5],
                    [1, -4, -5],
                    [1, -2, 5],
                    [2, -4, 5],
                    [2, 3, 4],
                ],
                _ => &[
                    [-3, -4, -5],
                    [-3, 4, 6],
                    [-2, 3, 6],
                    [-1, -4, -6],
                    [-1, 4, 6],
                    [1, -4, -5],
                    [1, 4, 5],
                    [4, -5, -6],
                ],
            };
            clauses.iter().map(|clause| clause.to_vec()).collect()
        };
        let formula = Cnf::new(variables, clauses)?;
        let oracle = GroverSat::new(formula.clone(), limits)?;
        let mut circuit = Circuit::new(Register::qubits(oracle.num_qubits())?, 0, 0);
        let mut labels = Vec::new();
        for wire in 0..variables {
            circuit.unitary(vec![wire], Unitary::h())?;
            labels.push(format!("Uniform Boolean variable x{} (q{wire})", wire + 1));
        }
        let clause_count = formula.clauses().len();
        for (index, operation) in oracle.oracle().operations().iter().enumerate() {
            circuit.push(operation.clone())?;
            labels.push(if index < 2 * clause_count {
                format!(
                    "Compute clause {} into clean ancilla q{}",
                    index / 2 + 1,
                    variables + index / 2
                )
            } else if index == 2 * clause_count {
                "Phase −1 exactly when every clause is satisfied".into()
            } else {
                let clause = clause_count - (index - 2 * clause_count - 1) / 2;
                format!("Uncompute clause {clause}; restore its work qubit to zero")
            });
        }
        for (index, operation) in oracle.diffusion().operations().iter().enumerate() {
            circuit.push(operation.clone())?;
            labels.push(format!(
                "Diffuse Boolean-variable amplitudes · operation {}",
                index + 1
            ));
        }
        let satisfying = (0..1 << variables)
            .filter(|&index| formula.evaluate(&assignment(index, variables)).unwrap())
            .collect();
        Ok(Self {
            formula,
            circuit,
            labels,
            satisfying,
            oracle_operations: oracle.oracle().operations().len(),
            diffusion_operations: oracle.diffusion().operations().len(),
        })
    }

    /// Validated normalized exact-3SAT clauses in DIMACS variable notation.
    pub fn formula(&self) -> &Cnf {
        &self.formula
    }
    /// Full preparation, oracle and diffusion circuit; no truth-table oracle is used.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// One semantic label per elementary circuit operation.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
    /// Exhaustively checked reference satisfying indices, independent of simulation.
    pub fn satisfying_assignments(&self) -> &[usize] {
        &self.satisfying
    }
    /// Evaluate a physically valid full-state probability vector after the final operation.
    pub fn readout(&self, probabilities: &[f64]) -> Result<ThreeSatReadout> {
        if probabilities.len() != self.circuit.register().size()
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.)
            || (probabilities.iter().sum::<f64>() - 1.).abs() > 5e-4
        {
            return Err(Error::InvalidValue(
                "invalid 3SAT probability readout".into(),
            ));
        }
        let variables = self.formula.num_variables();
        let clauses = self.formula.clauses().len();
        let mut marginal = vec![0.; 1 << variables];
        let mut clean = 0.;
        for (index, &probability) in probabilities.iter().enumerate() {
            marginal[index >> clauses] += probability;
            if index & ((1 << clauses) - 1) == 0 {
                clean += probability;
            }
        }
        let baseline = self.satisfying.len() as f64 / marginal.len() as f64;
        let success = self.satisfying.iter().map(|&i| marginal[i]).sum();
        let peak = marginal
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let peak_assignment = assignment(peak, variables);
        let verified = self.formula.evaluate(&peak_assignment)?;
        let clauses_satisfied_by_peak = self
            .formula
            .clauses()
            .iter()
            .filter(|clause| {
                clause.iter().any(|literal| {
                    peak_assignment[literal.unsigned_abs() as usize - 1] == (*literal > 0)
                })
            })
            .count();
        Ok(ThreeSatReadout {
            variables,
            clauses,
            total_qubits: self.circuit.register().wires(),
            basis_states: self.circuit.register().size(),
            circuit_operations: self.circuit.operations().len(),
            oracle_operations: self.oracle_operations,
            diffusion_operations: self.diffusion_operations,
            oracle_queries: 1,
            satisfying_assignments: self.satisfying.clone(),
            uniform_success_probability: baseline,
            analytic_success_probability: (3. * baseline.sqrt().asin()).sin().powi(2),
            success_probability: success,
            clean_ancilla_probability: clean,
            peak_probability: marginal[peak],
            assignment_probabilities: marginal,
            verified_witness: verified.then(|| peak_assignment.clone()),
            peak_assignment,
            clauses_satisfied_by_peak,
            reference_proves_unsatisfiable: self.satisfying.is_empty(),
        })
    }
}

fn assignment(index: usize, variables: usize) -> Vec<bool> {
    (0..variables)
        .map(|wire| index & (1 << (variables - wire - 1)) != 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseSimulator;
    use burn::tensor::{DType, Device};

    #[test]
    fn fixtures_are_exact_three_sat_with_independent_counts_and_resource_limits() {
        for (n, count) in [(3, 3), (4, 4), (5, 8), (6, 16)] {
            let benchmark = ThreeSatBenchmark::new(n, false, ResourceLimits::default()).unwrap();
            assert_eq!(benchmark.satisfying_assignments().len(), count);
            assert_eq!(benchmark.formula().clauses().len(), n + 2);
            assert!(
                benchmark
                    .formula()
                    .clauses()
                    .iter()
                    .all(|clause| clause.len() == 3)
            );
            for variable in 1..=n {
                assert!(
                    benchmark
                        .formula()
                        .clauses()
                        .iter()
                        .flatten()
                        .any(|v| v.unsigned_abs() as usize == variable)
                );
            }
            assert_eq!(
                benchmark.labels().len(),
                benchmark.circuit().operations().len()
            );
            assert_eq!(
                ThreeSatBenchmark::new(n, true, ResourceLimits::default())
                    .unwrap()
                    .satisfying_assignments()
                    .len(),
                0
            );
        }
        assert!(ThreeSatBenchmark::new(7, false, ResourceLimits::default()).is_err());
        assert!(
            ThreeSatBenchmark::new(
                5,
                false,
                ResourceLimits {
                    max_state_elements: 1024,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn actual_grover_readout_matches_amplitude_law_and_cleans_every_ancilla() {
        let device = Device::flex();
        for n in 3..=6 {
            for contradictory in [false, true] {
                let benchmark =
                    ThreeSatBenchmark::new(n, contradictory, ResourceLimits::default()).unwrap();
                let state = benchmark
                    .circuit()
                    .run(&DenseSimulator::default(), None, None, 1, &device)
                    .unwrap();
                state.validate(5e-5).unwrap();
                let probabilities = state
                    .probabilities()
                    .cast(DType::F64)
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap();
                let readout = benchmark.readout(&probabilities).unwrap();
                assert!(
                    (readout.success_probability - readout.analytic_success_probability).abs()
                        < 5e-5
                );
                assert!(readout.clean_ancilla_probability > 1. - 5e-5);
                assert_eq!(readout.reference_proves_unsatisfiable, contradictory);
                if contradictory {
                    assert!(readout.verified_witness.is_none());
                    assert_eq!(readout.success_probability, 0.);
                } else {
                    assert!(
                        benchmark
                            .formula()
                            .evaluate(&readout.verified_witness.unwrap())
                            .unwrap()
                    );
                    assert!(readout.success_probability > 2. * readout.uniform_success_probability);
                }
            }
        }
    }
}
