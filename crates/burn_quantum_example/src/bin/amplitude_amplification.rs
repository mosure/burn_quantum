//! Amplify odd-parity faults in a biased coherent event model. Compare a known
//! success mass, a promised lower bound, and a search which only verifies samples.
use burn_quantum::{
    DenseSimulator,
    algorithms::amplification::{AmplificationSchedule, OddParityExperiment, SearchConfig},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use serde_json::{Value, json};

const MAX_OPERATIONS: usize = 8192;
const DEFAULT_SEED: u64 = 7;

fn run(n: usize, rate: f64, minimum: f64, failure: f64, seed: u64) -> ExampleResult<Value> {
    ensure((3..=10).contains(&n), "choose 3 to 10 data qubits")?;
    let experiment = OddParityExperiment::new(n, rate)?;
    let initial = experiment.initial_success();
    ensure(
        minimum <= initial,
        "the supplied lower bound exceeds this model's initial good mass",
    )?;
    let known = AmplificationSchedule::known_success(initial)?;
    let fixed = AmplificationSchedule::fixed_point(minimum, failure)?;
    let amplifier = experiment.amplifier();
    let engine = DenseSimulator::default();
    let device = execution_device()?;
    let execute = |c: &burn_quantum::Circuit| c.run(&engine, None, None, 1, &device);
    let measure = |schedule: AmplificationSchedule| -> ExampleResult<(f64, f64)> {
        let circuit = amplifier.compile(schedule.phases(), MAX_OPERATIONS)?;
        let state = execute(&circuit)?;
        state.validate(5e-5)?;
        let (odd, clean) = experiment.readout(&values(state.probabilities())?)?;
        ensure(
            (odd - schedule.predicted_success(initial)?).abs() < 5e-5,
            "circuit and analytic success differ",
        )?;
        ensure(
            (clean - 1.).abs() < 5e-5,
            "oracle flag did not return to zero",
        )?;
        Ok((odd, clean))
    };
    let (baseline, _) = measure(AmplificationSchedule::standard(0))?;
    let (standard_success, standard_clean) = measure(known)?;
    let (fixed_success, fixed_clean) = measure(fixed)?;
    ensure(
        fixed_success + 5e-5 >= 1. - failure,
        "fixed-point failure exceeds its design budget",
    )?;

    let config = SearchConfig {
        seed,
        ..Default::default()
    };
    // This selector sees only independent terminal samples and the classical verifier.
    let search = amplifier.search(config, MAX_OPERATIONS, execute, |x| {
        Ok(x.count_ones() % 2 == 1)
    })?;
    let execution = Report::new(
        "amplitude_amplification",
        seed,
        fixed.rounds(),
        &[
            ("data_qubits", n as f64),
            ("initial_success", baseline),
            ("standard_success", standard_success),
            ("standard_reference", known.predicted_success(initial)?),
            ("fixed_point_success", fixed_success),
            ("fixed_point_reference", fixed.predicted_success(initial)?),
            ("fixed_point_guarantee", 1. - failure),
            ("standard_clean_flag", standard_clean),
            ("fixed_point_clean_flag", fixed_clean),
        ],
    )?;
    let mut report = serde_json::to_value(execution)?;
    report["model"] = json!({
        "per_wire_event_probability": rate,
        "marked_data_labels": (1usize << n) / 2,
        "analytic_initial_success": initial,
        "minimum_promised_success": minimum
    });
    report["strategies"] = json!({
        "known_mass": {
            "rounds": known.rounds(),
            "bit_oracle_calls": 2 * known.rounds()
        },
        "fixed_point": {
            "rounds": fixed.rounds(),
            "bit_oracle_calls": 2 * fixed.rounds(),
            "maximum_failure_probability": failure
        },
        "exponential_search": {
            "trials": search.trials,
            "phase_oracle_calls": search.oracle_calls,
            "bit_oracle_calls": 2 * search.oracle_calls,
            "verified_data_index": search.witness,
            "status": if search.witness.is_some() { "found" } else { "inconclusive" }
        }
    });
    Ok(report)
}

fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let usage = "amplitude_amplification [data_qubits=5] [event_probability=0.04] [minimum_good_mass=0.08] [maximum_failure=0.01] [seed=7]";
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{usage}");
        return Ok(());
    }
    ensure(args.len() <= 5, usage)?;
    let result = run(
        args.first().map(|s| s.parse()).transpose()?.unwrap_or(5),
        args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0.04),
        args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0.08),
        args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(0.01),
        args.get(4)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(DEFAULT_SEED),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn default_strategies_amplify_and_find_a_verified_syndrome() {
        let report = super::run(5, 0.04, 0.08, 0.01, super::DEFAULT_SEED).unwrap();
        let adaptive = &report["strategies"]["exponential_search"];
        assert_eq!(adaptive["status"], "found");
        assert_eq!(
            adaptive["verified_data_index"]
                .as_u64()
                .unwrap()
                .count_ones()
                % 2,
            1
        );
    }
}
