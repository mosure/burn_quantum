//! Noisy repeated-check planar memory, with a fixed-shot union-find decoder study.
//! Checks have ideal quantum backaction, independent noisy records and an ideal
//! final closure. This is not a circuit-level optical noise model.
use burn_quantum::photonic::{qec::threshold::*, qec::*, runtime::shot_seed};
use burn_quantum_example::{ExampleResult, Report};
use serde::Serialize;

#[derive(Serialize)]
struct Cell {
    distance: usize,
    rounds: usize,
    noise: PhenomenologicalNoise,
    seed: u64,
    start: u64,
    counts: ThresholdCounts,
    failure_interval: BinomialInterval,
    decoder_budget: DecoderBudget,
}
fn run(qualify: bool, confirmation: bool) -> ExampleResult<serde_json::Value> {
    let distances = if qualify {
        vec![3, 5, 7, 9, 11]
    } else {
        vec![3, 5]
    };
    let rates = if qualify {
        vec![0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    } else {
        vec![0.01, 0.05]
    };
    let shots = if confirmation {
        100_000
    } else if qualify {
        20_000
    } else {
        128
    };
    let base_seed = if confirmation { 711214 } else { 711210 };
    let family = if confirmation {
        40
    } else if qualify {
        96
    } else {
        distances.len() * rates.len()
    };
    let budget = DecoderBudget {
        max_edge_visits: 10_000_000,
        max_growth_rounds: 128,
    };
    let rows = std::thread::scope(|scope| -> burn_quantum::Result<Vec<Cell>> {
        let workers: Vec<_> = distances
            .iter()
            .map(|&distance| {
                let rates = &rates;
                scope.spawn(move || -> burn_quantum::Result<Vec<Cell>> {
                    let code = PlanarCode::new(distance, 1024)?;
                    let model =
                        PhenomenologicalMemory::new(&code, distance, GraphLimits::default())?;
                    let mut rows = vec![];
                    for (index, &p) in rates.iter().enumerate() {
                        let noise = PhenomenologicalNoise::symmetric(p)?;
                        let seed = shot_seed(base_seed, distance as u64, index as u64);
                        let counts = model.run(noise, 0, shots, seed, budget)?;
                        let failure_interval = BinomialInterval::simultaneous(
                            counts.unsuccessful()?,
                            shots,
                            0.01,
                            family,
                        )?;
                        eprintln!(
                            "planar phenomenological d={distance} p={p}: failures={}/{} limits={}",
                            counts.logical_failures, shots, counts.decoder_limits
                        );
                        rows.push(Cell {
                            distance,
                            rounds: distance,
                            noise,
                            seed,
                            start: 0,
                            counts,
                            failure_interval,
                            decoder_budget: budget,
                        });
                    }
                    Ok(rows)
                })
            })
            .collect();
        let mut rows = vec![];
        for worker in workers {
            rows.extend(worker.join().map_err(|_| {
                burn_quantum::Error::InvalidValue("planar threshold worker panicked".into())
            })??);
        }
        Ok(rows)
    })?;
    let qualification_valid = rows.iter().all(|r| {
        r.counts.attempts == shots && r.counts.decoder_limits == 0 && r.counts.inconsistent == 0
    });
    let mut trends = vec![];
    for &p in &rates {
        let points: Vec<_> = rows
            .iter()
            .filter(|r| r.noise.data_x == p)
            .map(|r| (r.distance, r.failure_interval))
            .collect();
        trends.push((
            p,
            if qualification_valid {
                compare_sizes(&points)?
            } else {
                FiniteSizeTrend::Unresolved
            },
        ));
    }
    let report = Report::new(
        "planar_threshold",
        base_seed,
        0,
        &[
            ("attempts", (shots * rows.len()) as f64),
            ("cells", rows.len() as f64),
            ("shots_per_cell", shots as f64),
            ("family_alpha", 0.01),
            ("planned_intervals", family as f64),
        ],
    )?;
    let mut json = serde_json::to_value(report)?;
    json["qualification"] = serde_json::json!(qualify);
    json["confirmation"] = serde_json::json!(confirmation);
    json["qualification_valid"] = serde_json::json!(qualification_valid);
    json["cells"] = serde_json::to_value(rows)?;
    json["finite_size_bracket"] = serde_json::to_value(crossing_bracket(&trends)?)?;
    json["finite_size_trends"] = serde_json::to_value(trends)?;
    json["scope"] = serde_json::json!(
        "Host union-find phenomenological memory: independent X and Z faults and readout flips each at p, Y probability p^2; d noisy rounds and ideal initial/final checks. Not circuit-level, lattice-surgery or hardware threshold."
    );
    Ok(json)
}
fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (qualify, confirmation) = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => (false, false),
        ["--qualify"] => (true, false),
        ["--qualify-confirmation"] => (true, true),
        _ => return Err("usage: planar_threshold [--qualify | --qualify-confirmation]".into()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&run(qualify, confirmation)?)?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn noisy_measurement_threshold_smoke() {
        super::run(false, false).unwrap();
    }
}
