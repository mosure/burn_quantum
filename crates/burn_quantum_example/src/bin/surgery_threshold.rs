//! Complete merge/split instrument qualification under an explicit local
//! phenomenological model. All sampling/decoding is host-only. `--qualify` runs
//! the predeclared 80-cell, 1.6-million-shot experiment; default/tests are smoke.
use burn_quantum::photonic::{qec::threshold::*, qec::*, runtime::shot_seed};
use burn_quantum_example::{ExampleResult, Report};
use serde::Serialize;
use std::{collections::BTreeMap, time::Instant};

#[derive(Serialize)]
struct Cell {
    basis: &'static str,
    distance: usize,
    noisy_rounds: usize,
    probability: f64,
    seed: u64,
    start: u64,
    study: GraphNoiseStudy,
    failure_interval: BinomialInterval,
    decoder_budget: DecoderBudget,
    fault_coordinates: usize,
    data_fault_coordinates: usize,
    readout_fault_coordinates: usize,
    records_including_final: usize,
    detector_count: usize,
    logical_distances: [usize; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<f64>,
}
fn run(qualify: bool) -> ExampleResult<serde_json::Value> {
    let distances = if qualify {
        vec![3, 5, 7, 9, 11]
    } else {
        vec![3, 5]
    };
    let rates = if qualify {
        vec![0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    } else {
        vec![0.01, 0.03]
    };
    let shots = if qualify { 20_000 } else { 64 };
    let family = if qualify {
        80
    } else {
        2 * distances.len() * rates.len()
    };
    let base_seed = 711212;
    let started = Instant::now();
    let cells = std::thread::scope(|scope| -> burn_quantum::Result<Vec<Cell>> {
        let jobs: Vec<_> = distances
            .iter()
            .flat_map(|&d| [(d, MergeBasis::ZZ), (d, MergeBasis::XX)])
            .collect();
        let workers:Vec<_>=jobs.into_iter().map(|(d,basis)|{let rates=&rates;scope.spawn(move||->burn_quantum::Result<Vec<Cell>>{
            let model=SurgeryDetectorModel::new(PlanarSurgery::new(d,basis,SurgeryLimits::default())?,SurgeryDetectorLimits::default())?;
            let graph=model.graph();let data=model.faults().iter().filter(|f|matches!(f,SurgeryFault::Data{..})).count();
            let budget=DecoderBudget{max_edge_visits:256*graph.edges().len() as u64,max_growth_rounds:128};
            let basis_name=if basis==MergeBasis::ZZ{"ZZ"}else{"XX"};let basis_index=u64::from(basis==MergeBasis::XX);
            let mut rows=Vec::new();
            for(index,&p)in rates.iter().enumerate(){
                let seed=shot_seed(base_seed,d as u64,2*index as u64+basis_index);
                let begin=Instant::now();let study=graph.run_independent_faults(p,0,shots,seed,budget)?;
                let failure_interval=BinomialInterval::simultaneous(study.counts.unsuccessful()?,shots,0.01,family)?;
                eprintln!("surgery {basis_name} d={d} p={p}: failures={}/{} limits={} inconsistent={}",study.counts.logical_failures,shots,study.counts.decoder_limits,study.counts.inconsistent);
                rows.push(Cell{basis:basis_name,distance:d,noisy_rounds:2*d,probability:p,seed,start:0,study,failure_interval,decoder_budget:budget,fault_coordinates:graph.edges().len(),data_fault_coordinates:data,readout_fault_coordinates:graph.edges().len()-data,records_including_final:model.records().len(),detector_count:graph.detectors(),logical_distances:model.logical_distances(),seconds:qualify.then(||begin.elapsed().as_secs_f64())});
            }
            Ok(rows)
        })}).collect();
        let mut rows = Vec::new();
        for worker in workers {
            rows.extend(worker.join().map_err(|_| {
                burn_quantum::Error::InvalidValue("surgery threshold worker panicked".into())
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
    for basis in ["ZZ", "XX"] {
        let mut values = Vec::new();
        for &p in &rates {
            let points: Vec<_> = cells
                .iter()
                .filter(|c| c.basis == basis && c.probability == p)
                .map(|c| (c.distance, c.failure_interval))
                .collect();
            values.push((p, compare_sizes(&points)?));
        }
        brackets.insert(
            basis,
            if valid {
                crossing_bracket(&values)?
            } else {
                None
            },
        );
        trends.insert(basis, values);
    }
    // Identity follows the workspace's explicit backend selection. Sampling and
    // decoding remain host work, as the model metadata states on every run.
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
    let mut report = Report::new("surgery_threshold", base_seed, 0, &metrics)?;
    report.dtype = "binary/f64";
    let mut output = serde_json::to_value(report)?;
    output["qualification"] = serde_json::json!(qualify);
    output["qualification_valid"] = serde_json::json!(valid);
    output["cells"] = serde_json::to_value(cells)?;
    output["finite_size_trends"] = serde_json::to_value(trends)?;
    output["finite_size_brackets"] = serde_json::to_value(brackets)?;
    output["model"] = serde_json::json!({"name":"complete_doubled_seam_planar_surgery_phenomenological","basis":["ZZ","XX"],"noise":"Independent X and Z before every data round, and readout sign flips, each at p. Y has probability p^2. No hook, encoder, optical or final-closure errors.","rounds":"d merge + d split + ideal final checks; unknown first seam and split-boundary signs retained","logical_event":"any residual flip among four output-Choi correlations including reported joint parity","stopping_rule":"fixed shots per cell; every decoder failure retained","statistics":"simultaneous Clopper-Pearson alpha/family; adjacent-distance strict interval separation; separate family from other threshold runners","scope":"finite-distance instrument suppression/crossing, not an asymptotic proof or photonic hardware threshold"});
    Ok(output)
}
fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let qualify = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => false,
        ["--qualify"] => true,
        _ => return Err("usage: surgery_threshold [--qualify]".into()),
    };
    println!("{}", serde_json::to_string_pretty(&run(qualify)?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn bounded_complete_instrument_threshold_smoke() {
        let output = super::run(false).unwrap();
        assert_eq!(output["dtype"], "binary/f64");
        assert_eq!(output["qualification_valid"], true);
        assert_eq!(output["metrics"]["attempts"], 512.);
        assert_eq!(output["cells"].as_array().unwrap().len(), 8);
    }
}
