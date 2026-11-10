//! Actual six-ring surface memory and merge record-noise qualification.
//! Quantum resources, fusions and terminal bases determine every detector edge.
//! The declared noise is independent outcome flips with perfect observed closure.
use burn_quantum::photonic::{
    NodeId,
    qec::{fusion_surgery::SixRingSurfaceNetwork, threshold::*},
    runtime::shot_seed,
};
use burn_quantum_example::{ExampleResult, Report};
use serde::Serialize;
use std::{collections::BTreeMap, time::Instant};

#[derive(Serialize)]
struct Cell {
    instrument: &'static str,
    distance: usize,
    rounds: usize,
    probability: f64,
    seed: u64,
    start: u64,
    resource_states: usize,
    photons: usize,
    fusion_records: usize,
    terminal_records: usize,
    ideal_closure_records: usize,
    detectors: usize,
    logical_distances: Vec<usize>,
    decoder_budget: DecoderBudget,
    study: GraphNoiseStudy,
    failure_interval: BinomialInterval,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Study {
    Smoke,
    Qualification,
    MemoryConfirmation,
}
fn run(mode: Study) -> ExampleResult<serde_json::Value> {
    let qualify = mode != Study::Smoke;
    let confirmation = mode == Study::MemoryConfirmation;
    let distances = if qualify {
        vec![3, 5, 7, 9]
    } else {
        vec![3, 5]
    };
    let rates = if confirmation {
        vec![0.003, 0.004, 0.0075, 0.0125]
    } else if qualify {
        vec![0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    } else {
        vec![0.01, 0.03]
    };
    let shots = if confirmation {
        100_000
    } else if qualify {
        50_000
    } else {
        64
    };
    let instruments = if confirmation {
        vec!["memory"]
    } else {
        vec!["memory", "merge"]
    };
    let family = instruments.len() * distances.len() * rates.len();
    let base_seed = if confirmation { 711216 } else { 711215 };
    let started = Instant::now();
    let cells = std::thread::scope(|scope| -> burn_quantum::Result<Vec<Cell>> {
        let jobs: Vec<_> = distances
            .iter()
            .flat_map(|&d| [(d, false), (d, true)])
            .filter(|&(_, merge)| !confirmation || !merge)
            .collect();
        let workers: Vec<_> = jobs.into_iter().map(|(d, merge)| {
            let rates = &rates;
            scope.spawn(move || -> burn_quantum::Result<Vec<Cell>> {
                let net = if merge {
                    SixRingSurfaceNetwork::parity_merge(d, 1, d, 1, NodeId(0), NodeId(1), 16384)?
                } else {
                    SixRingSurfaceNetwork::memory(d, d, NodeId(0), NodeId(1), 16384)?
                };
                let model = CubicalRecordModel::new(&net, 1 << 20)?;
                let graph = if merge {
                    model.surgery_graph(GraphLimits::default())?
                } else {
                    model.memory_graph(GraphLimits::default())?
                };
                let logical_distances = (0..graph.logicals())
                    .map(|i| graph.minimum_logical_distance(i, 1 << 30).map(|v| v.distance))
                    .collect::<burn_quantum::Result<Vec<_>>>()?;
                if logical_distances.iter().any(|&v| v != d) {
                    return Err(burn_quantum::Error::InvalidValue("surface record-distance prerequisite".into()));
                }
                let budget = DecoderBudget { max_edge_visits: 256 * graph.edges().len() as u64, max_growth_rounds: 128 };
                let instrument = if merge { "merge" } else { "memory" };
                let fusion_records = 2 * net.patch().program().num_fusions();
                let mut rows = Vec::new();
                for (index, &p) in rates.iter().enumerate() {
                    let seed = shot_seed(base_seed, d as u64, 2 * index as u64 + u64::from(merge));
                    let begin = Instant::now();
                    let study = graph.run_independent_faults(p, 0, shots, seed, budget)?;
                    let failure_interval = BinomialInterval::simultaneous(study.counts.unsuccessful()?, shots, 0.01, family)?;
                    eprintln!("surface {instrument} d={d} p={p}: failures={}/{} limits={} inconsistent={}", study.counts.logical_failures, shots, study.counts.decoder_limits, study.counts.inconsistent);
                    rows.push(Cell {
                        instrument, distance: d, rounds: net.rounds(), probability: p, seed, start: 0,
                        resource_states: net.resource_centers().len(), photons: net.patch().program().num_photons(),
                        fusion_records, terminal_records: model.raw_records() - fusion_records,
                        ideal_closure_records: model.closure().len(), detectors: graph.detectors(),
                        logical_distances: logical_distances.clone(), decoder_budget: budget,
                        study, failure_interval, seconds: qualify.then(|| begin.elapsed().as_secs_f64()),
                    });
                }
                Ok(rows)
            })
        }).collect();
        let mut rows = Vec::new();
        for worker in workers {
            rows.extend(worker.join().map_err(|_| {
                burn_quantum::Error::InvalidValue("surface threshold worker panicked".into())
            })??);
        }
        Ok(rows)
    })?;
    let valid = cells.iter().all(|c| {
        c.study.counts.attempts == shots
            && c.study.counts.decoder_limits == 0
            && c.study.counts.inconsistent == 0
    });
    let mut trends = BTreeMap::new();
    let mut brackets = BTreeMap::new();
    for instrument in instruments {
        let values = rates
            .iter()
            .map(|&p| {
                let points = cells
                    .iter()
                    .filter(|c| c.instrument == instrument && c.probability == p)
                    .map(|c| (c.distance, c.failure_interval))
                    .collect::<Vec<_>>();
                Ok((
                    p,
                    if valid {
                        compare_sizes(&points)?
                    } else {
                        FiniteSizeTrend::Unresolved
                    },
                ))
            })
            .collect::<burn_quantum::Result<Vec<_>>>()?;
        brackets.insert(
            instrument,
            if valid {
                crossing_bracket(&values)?
            } else {
                None
            },
        );
        trends.insert(instrument, values);
    }
    let mut metrics = vec![
        ("attempts", (shots * cells.len()) as f64),
        ("cells", cells.len() as f64),
        ("shots_per_cell", shots as f64),
        ("family_alpha", 0.01),
        ("planned_intervals", family as f64),
    ];
    if qualify {
        metrics.push(("seconds", started.elapsed().as_secs_f64()));
    }
    let mut report = Report::new("surface_threshold", base_seed, 0, &metrics)?;
    report.dtype = "binary/f64";
    let mut output = serde_json::to_value(report)?;
    output["qualification"] = serde_json::json!(qualify);
    output["qualification_valid"] = serde_json::json!(valid);
    if confirmation {
        output["stage"] = serde_json::json!("independent_memory_confirmation_without_pooling");
    }
    output["cells"] = serde_json::to_value(cells)?;
    output["finite_size_trends"] = serde_json::to_value(trends)?;
    output["finite_size_brackets"] = serde_json::to_value(brackets)?;
    output["model"] = serde_json::json!({
        "name": "actual_six_ring_colored_boundary_record_flips",
        "geometry": "d by d by d memory; two d by d patches, one pre-round, d fusion-seam rounds, one post-round",
        "noise": "Independent sign flip with probability p on every physical XX/ZZ fusion and terminal Pauli outcome. Ideal sources and quantum measurements. No missing fusions, loss, correlated source errors or noisy closing checks.",
        "closure": "Explicit nondestructive end-code check measurements with actual quantum signs, assumed perfectly reported. Appended closure bits are not noisy coordinates.",
        "logical_event": "any residual nontrivial encoded Choi correlation after syndrome-only decoding",
        "decoder": "synchronous unweighted union-find growth and forest peeling, host execution",
        "stopping_rule": "fixed shots per cell, no pooling, every failed decoder attempt retained",
        "statistics": "99% simultaneous Clopper-Pearson across all planned cells; every adjacent distance must have disjoint intervals",
        "scope": "finite-distance reference-model qualification, not a hardware threshold or an asymptotic proof"
    });
    Ok(output)
}
fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mode = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => Study::Smoke,
        ["--qualify"] => Study::Qualification,
        ["--qualify-confirmation"] => Study::MemoryConfirmation,
        _ => return Err("usage: surface_threshold [--qualify | --qualify-confirmation]".into()),
    };
    println!("{}", serde_json::to_string_pretty(&run(mode)?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn actual_surface_record_noise_smoke() {
        let output = super::run(super::Study::Smoke).unwrap();
        assert_eq!(output["qualification_valid"], true);
        assert_eq!(output["metrics"]["attempts"], 512.);
        assert_eq!(output["cells"].as_array().unwrap().len(), 8);
    }
}
