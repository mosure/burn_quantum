//! Deterministic threaded or TCP shot sharding and checked resumable result ledgers.
//! Independent trajectories share immutable recipes, not ownership of one quantum state.
pub mod events;
pub mod jobs;
mod ledger;
pub mod network;
use crate::{Error, Result};
pub use ledger::{JobIdentity, JobLedger, ShotRecord};
use std::sync::atomic::{AtomicBool, Ordering};

/// Version-one seed derivation. Independent of worker count and scheduling order.
/// Component/event IDs must identify physical randomness, not execution thread IDs.
pub fn shot_seed(seed: u64, shot: u64, component: u64) -> u64 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    mix(seed ^ mix(shot.wrapping_add(0x9e3779b97f4a7c15)) ^ mix(component))
}

/// Execute missing shot IDs in bounded waves. Completed waves survive cancellation/errors.
///
/// The callback receives `(shot_id, derived_seed)` and must be deterministic for that
/// identity. No simulation fallback or retry-until-success filtering is performed.
/// Errors leave the current wave uncommitted; resumption repeats its same shot IDs.
/// Cancellation is cooperative between waves; it cannot interrupt a running callback.
pub fn execute_shards<T, F>(
    ledger: &mut JobLedger<T>,
    workers: usize,
    max_wave: usize,
    cancelled: &AtomicBool,
    evaluate: F,
) -> Result<()>
where
    T: Clone + PartialEq + Send,
    F: Fn(u64, u64) -> Result<T> + Sync,
{
    if workers == 0 || workers > 256 || max_wave == 0 || max_wave > 1_048_576 {
        return Err(Error::InvalidValue("invalid worker/wave limits".into()));
    }
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        let missing = ledger.missing(max_wave);
        if missing.is_empty() {
            return Ok(());
        }
        let seed = ledger.identity().seed;
        let wave: Result<Vec<ShotRecord<T>>> = std::thread::scope(|scope| {
            let chunks = missing.chunks(missing.len().div_ceil(workers));
            let handles: Vec<_> = chunks
                .map(|chunk| {
                    let evaluate = &evaluate;
                    scope.spawn(move || -> Result<Vec<ShotRecord<T>>> {
                        chunk
                            .iter()
                            .map(|&id| {
                                Ok(ShotRecord {
                                    id,
                                    value: evaluate(id, shot_seed(seed, id, 0))?,
                                })
                            })
                            .collect()
                    })
                })
                .collect();
            let mut output = Vec::with_capacity(missing.len());
            let joined: Vec<_> = handles.into_iter().map(|handle| handle.join()).collect();
            for result in joined {
                output.extend(
                    result.map_err(|_| Error::InvalidValue("shot worker panicked".into()))??,
                );
            }
            Ok(output)
        });
        ledger.merge(ledger.identity().clone(), wave?)?;
    }
}
