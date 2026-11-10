//! Fixed-shot six-ring erasure qualification. All graph decoding runs on the host.
//! `--qualify six-ring` or `--qualify optical` runs the predeclared 20,000-shot grid.
use burn_quantum::photonic::{NodeId, qec::threshold::*, qec::*, runtime::shot_seed};
use burn_quantum_example::{ExampleResult, Report};
use rand::Rng;
use serde::Serialize;

#[derive(Serialize)]
struct Cell {
    model: &'static str,
    size: usize,
    noise: f64,
    seed: u64,
    start: u64,
    counts: ThresholdCounts,
    failure_interval: BinomialInterval,
    conditional_expected_failures: f64,
    records: usize,
    decoder_budget: DecoderBudget,
    optical_law: Option<FusionErasureDistribution>,
}

fn run(qualify: bool, optical: bool) -> ExampleResult<serde_json::Value> {
    let sides = if qualify {
        vec![12, 16, 20, 24]
    } else {
        vec![4, 6]
    };
    let rates = if optical {
        if qualify {
            vec![0.005, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        } else {
            vec![0.01, 0.04]
        }
    } else if qualify {
        vec![0.10, 0.11, 0.115, 0.12, 0.125, 0.13, 0.14]
    } else {
        vec![0.08, 0.16]
    };
    let shots = if qualify { 20_000 } else { 256 };
    let family = if qualify {
        96
    } else {
        sides.len() * rates.len()
    };
    let base_seed = if optical { 711211 } else { 711209 };
    let model = if optical {
        "six_ring_shor_boosted_optical_loss"
    } else {
        "six_ring_iid_outcome_erasure"
    };
    let rows = std::thread::scope(|scope| -> burn_quantum::Result<Vec<Cell>> {
        let workers: Vec<_> = sides
            .iter()
            .map(|&side| {
                let rates = &rates;
                scope.spawn(move || -> burn_quantum::Result<Vec<Cell>> {
                    let network = SixRingTorus::new(side, NodeId(0), NodeId(1), 6 * side.pow(3))?;
                    let engine = SixRingErasureModel::new(&network, GraphLimits::default())?;
                    let budget = DecoderBudget {
                        max_edge_visits: 8 * engine.records() as u64,
                        max_growth_rounds: 128,
                    };
                    let mut rows = Vec::new();
                    for (index, &noise) in rates.iter().enumerate() {
                        let seed = shot_seed(base_seed, side as u64, index as u64);
                        let optical_law = optical
                            .then(|| {
                                OpticalFusionLoss::new(noise, 1, true).map(|law| law.distribution())
                            })
                            .transpose()?;
                        let counts = if let Some(law) = optical_law {
                            engine.run_with_masks(0, shots, seed, budget, |_, rng, mask| {
                                for pair in mask.chunks_exact_mut(2) {
                                    pair.copy_from_slice(&law.sample(rng.random())?);
                                }
                                Ok(())
                            })?
                        } else {
                            engine.run(noise, 0, shots, seed, budget)?
                        };
                        let failure_interval = BinomialInterval::simultaneous(
                            counts.unsuccessful()?,
                            shots,
                            0.01,
                            family,
                        )?;
                        eprintln!(
                            "{model} size={side} p={noise} logical={}/{} ambiguous={} limits={}",
                            counts.logical_failures,
                            shots,
                            counts.ambiguous_erasures,
                            counts.decoder_limits
                        );
                        rows.push(Cell {
                            model,
                            size: side,
                            noise,
                            seed,
                            start: 0,
                            conditional_expected_failures: counts.conditional_logical_failures(),
                            counts,
                            failure_interval,
                            records: engine.records(),
                            decoder_budget: budget,
                            optical_law,
                        });
                    }
                    Ok(rows)
                })
            })
            .collect();
        let mut rows = Vec::new();
        for worker in workers {
            rows.extend(worker.join().map_err(|_| {
                burn_quantum::Error::InvalidValue("threshold worker panicked".into())
            })??);
        }
        Ok(rows)
    })?;
    let qualification_valid = rows.iter().all(|r| {
        r.counts.attempts == shots && r.counts.decoder_limits == 0 && r.counts.inconsistent == 0
    });
    let mut trends = Vec::new();
    for &rate in &rates {
        let points: Vec<_> = rows
            .iter()
            .filter(|r| r.noise == rate)
            .map(|r| (r.size, r.failure_interval))
            .collect();
        trends.push((
            rate,
            if qualification_valid {
                compare_sizes(&points)?
            } else {
                FiniteSizeTrend::Unresolved
            },
        ));
    }
    let bracket = crossing_bracket(&trends)?;
    let report = Report::new(
        "fbqc_threshold",
        base_seed,
        0,
        &[
            ("cells", rows.len() as f64),
            ("attempts", (shots * rows.len()) as f64),
            ("shots_per_cell", shots as f64),
            ("family_alpha", 0.01),
            ("planned_intervals", family as f64),
            (
                "decoder_limits",
                rows.iter().map(|r| r.counts.decoder_limits as f64).sum(),
            ),
        ],
    )?;
    let mut json = serde_json::to_value(report)?;
    json["host_model"] = serde_json::json!(model);
    json["qualification"] = serde_json::json!(qualify);
    json["qualification_valid"] = serde_json::json!(qualification_valid);
    json["cells"] = serde_json::to_value(rows)?;
    json["finite_size_trends"] = serde_json::to_value(trends)?;
    json["finite_size_bracket"] = serde_json::to_value(bracket)?;
    json["scope"] = serde_json::json!(
        "Host exact erasure homology; fixed-size suppression/crossing evidence, not an asymptotic theorem or hardware qualification. No unheralded errors. Native quantum-channel parity is tested by fbqc_lattice_surgery."
    );
    Ok(json)
}
fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (qualify, optical) = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => (false, false),
        ["--optical"] => (false, true),
        ["--qualify", "six-ring"] => (true, false),
        ["--qualify", "optical"] => (true, true),
        _ => return Err("usage: fbqc_threshold [--optical | --qualify {six-ring|optical}]".into()),
    };
    println!("{}", serde_json::to_string_pretty(&run(qualify, optical)?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn fixed_shot_threshold_protocol_and_correlated_optical_model() {
        for optical in [false, true] {
            let report = super::run(false, optical).unwrap();
            assert_eq!(report["metrics"]["attempts"], 1024.);
        }
    }
}
