use super::{AmplificationSchedule, AmplitudeAmplifier};
use crate::{
    Angle, Circuit, Error, Register, Result, Rotation, algorithms::AlgorithmCircuit, gate::Axis,
};

/// Coherent model of independent binary events. A reversible parity check marks
/// every odd-weight data label. Their probabilities are generally unequal.
#[derive(Clone, Debug)]
pub struct OddParityExperiment {
    amplifier: AmplitudeAmplifier,
    initial_success: f64,
}

impl OddParityExperiment {
    /// Prepare a product of Ry rotations with per-wire event probability `rate`.
    /// Each data bit controls one XOR into the flag, so the predicate uses n gates
    /// even though it marks half of all 2^n labels.
    pub fn new(qubits: usize, rate: f64) -> Result<Self> {
        if !(0. ..=1.).contains(&rate) {
            return Err(Error::InvalidValue(
                "event probability must be in [0, 1]".into(),
            ));
        }
        let data = Register::qubits(qubits)?;
        let mut oracle = Circuit::new(AmplitudeAmplifier::oracle_register(&data)?, 0, 0);
        let mut preparation = Circuit::new(data, 0, 0);
        let angle = 2. * rate.sqrt().asin();
        let mut odd = 0.;
        for wire in 0..qubits {
            preparation.rotate(wire, Rotation::qubit(Axis::Y, Angle::Constant(angle)))?;
            oracle.controlled_x(vec![(wire, 1)], qubits)?;
            // Equivalent to (1 - (1 - 2r)^n) / 2 without cancellation for rare events.
            odd = odd * (1. - rate) + (1. - odd) * rate;
        }
        Ok(Self {
            amplifier: AmplitudeAmplifier::new(preparation, oracle)?,
            initial_success: odd,
        })
    }

    /// Reusable preparation and bit oracle for any amplification strategy.
    pub fn amplifier(&self) -> &AmplitudeAmplifier {
        &self.amplifier
    }

    /// Independent classical probability of an odd number of Bernoulli events.
    pub fn initial_success(&self) -> f64 {
        self.initial_success
    }

    /// Measured odd-parity data mass and clean-flag probability, respectively.
    /// The first value marginalizes over both flag levels. The second checks cleanup.
    /// Divide by the total Born weight after checking it is within 5e-4 of unity.
    pub fn readout(&self, probabilities: &[f64]) -> Result<(f64, f64)> {
        if probabilities.len() != self.amplifier.oracle().register().size() {
            return Err(Error::Shape(
                "expected one joint data-and-flag distribution".into(),
            ));
        }
        let total: f64 = probabilities.iter().sum();
        if probabilities.iter().any(|p| !p.is_finite() || *p < 0.) || (total - 1.).abs() > 5e-4 {
            return Err(Error::InvalidValue(
                "invalid joint probability distribution".into(),
            ));
        }
        let mut odd = 0.;
        let mut clean = 0.;
        for (x, flag) in probabilities.chunks_exact(2).enumerate() {
            if x.count_ones() % 2 == 1 {
                odd += flag[0] + flag[1];
            }
            clean += flag[0];
        }
        Ok((odd / total, clean / total))
    }

    /// Build a circuit with one educational timeline label per physical instruction.
    pub fn labeled_circuit(
        &self,
        schedule: AmplificationSchedule,
        max_operations: usize,
    ) -> Result<AlgorithmCircuit> {
        let circuit = self.amplifier.compile(schedule.phases(), max_operations)?;
        let n = self.amplifier.preparation().register().wires();
        let mut labels: Vec<_> = (0..n)
            .map(|q| format!("Prepare event amplitude q{q}"))
            .collect();
        for (i, phase) in schedule.phases().enumerate() {
            let round = i + 1;
            let prefix = format!("Round {round}");
            labels.extend((0..n).map(|q| format!("{prefix} · accumulate parity q{q}")));
            labels.push(format!(
                "{prefix} · phase odd patterns by {:.3} rad",
                phase.good
            ));
            labels.extend(
                (0..n)
                    .rev()
                    .map(|q| format!("{prefix} · uncompute parity q{q}")),
            );
            labels.extend(
                (0..n)
                    .rev()
                    .map(|q| format!("{prefix} · undo preparation q{q}")),
            );
            labels.push(format!("{prefix} · flag the all-zero input"));
            labels.push(format!(
                "{prefix} · phase the source by {:.3} rad",
                phase.source
            ));
            labels.push(format!("{prefix} · clear source flag"));
            labels.extend((0..n).map(|q| format!("{prefix} · restore preparation q{q}")));
        }
        Ok(AlgorithmCircuit {
            circuit,
            labels,
            parameters: vec![],
            protocol: None,
        })
    }
}
