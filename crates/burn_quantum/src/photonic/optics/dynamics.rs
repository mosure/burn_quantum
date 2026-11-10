//! Stateful nonparalyzable detector dead time with reproducible event identities.
use super::Detector;
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// One attempted detection window, including blinded detectors and zero clicks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectorEvent {
    /// Detection-window start tick.
    pub tick: u64,
    /// Saturated measured counts in detector order.
    pub counts: Vec<usize>,
    /// True where the detector was blind because of an earlier click.
    pub blind: Vec<bool>,
}
/// Checked stateful bank. Any click disables its detector for `dead_ticks`.
/// Blind events do not extend dead time (nonparalyzable model).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "BankData", into = "BankData")]
pub struct DetectorBank(BankData);
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BankData {
    version: u32,
    detectors: Vec<Detector>,
    dead_ticks: Vec<u64>,
    available: Vec<u64>,
    last_tick: Option<u64>,
    events: usize,
    max_events: usize,
}
impl DetectorBank {
    /// Create up to 256 detectors with independent calibrated responses/dead times.
    pub fn new(detectors: Vec<Detector>, dead_ticks: Vec<u64>, max_events: usize) -> Result<Self> {
        let n = detectors.len();
        Self::try_from(BankData {
            version: 1,
            detectors,
            dead_ticks,
            available: vec![0; n],
            last_tick: None,
            events: 0,
            max_events,
        })
    }
    /// Earliest live tick for each detector; half-open blind intervals end here.
    pub fn available_at(&self) -> &[u64] {
        &self.0.available
    }
    /// Number of attempted windows, including complete blindness/no clicks.
    pub fn events(&self) -> usize {
        self.0.events
    }
    /// Observe a strictly increasing window tick. Errors are atomic.
    /// The seed/shot/window identities must be stable across retry/repartitioning.
    pub fn observe(
        &mut self,
        tick: u64,
        photons: &[usize],
        seed: u64,
        shot: u64,
        window: u64,
    ) -> Result<DetectorEvent> {
        if photons.len() != self.0.detectors.len()
            || photons.iter().any(|&n| n > 64)
            || self.0.last_tick.is_some_and(|last| tick <= last)
        {
            return Err(Error::InvalidValue(
                "invalid detector counts or noncausal window".into(),
            ));
        }
        let next = self.0.events.checked_add(1).ok_or(Error::Overflow)?;
        if next > self.0.max_events {
            return Err(Error::ResourceLimit {
                resource: "detector windows",
                required: next,
                limit: self.0.max_events,
            });
        }
        let mut available = self.0.available.clone();
        let mut counts = vec![];
        let mut blind = vec![];
        let domain = crate::photonic::runtime::shot_seed(seed, shot, 0x4445_5445_4354_4f52);
        for (i, (&n, detector)) in photons.iter().zip(&self.0.detectors).enumerate() {
            let dead = tick < available[i];
            blind.push(dead);
            let count = if dead {
                0
            } else {
                let mut rng = ChaCha8Rng::seed_from_u64(crate::photonic::runtime::shot_seed(
                    domain, window, i as u64,
                ));
                let draw = rng.random::<f64>();
                let mut total = 0.;
                let mut answer = detector.resolution();
                for (k, p) in detector.response(n)?.into_iter().enumerate() {
                    total += p;
                    if draw < total {
                        answer = k;
                        break;
                    }
                }
                answer
            };
            if count > 0 {
                available[i] = tick
                    .checked_add(self.0.dead_ticks[i])
                    .ok_or(Error::Overflow)?;
            }
            counts.push(count);
        }
        self.0.available = available;
        self.0.last_tick = Some(tick);
        self.0.events = next;
        Ok(DetectorEvent {
            tick,
            counts,
            blind,
        })
    }
}
impl TryFrom<BankData> for DetectorBank {
    type Error = Error;
    fn try_from(v: BankData) -> Result<Self> {
        let n = v.detectors.len();
        if v.version != 1
            || n == 0
            || n > 256
            || v.dead_ticks.len() != n
            || v.available.len() != n
            || v.events > v.max_events
            || (v.events == 0) != v.last_tick.is_none()
            || v.available
                .iter()
                .zip(&v.dead_ticks)
                .any(|(&ready, &dead)| match v.last_tick {
                    Some(last) => ready > last.saturating_add(dead),
                    None => ready != 0,
                })
        {
            return Err(Error::InvalidValue(
                "invalid detector-bank checkpoint".into(),
            ));
        }
        Ok(Self(v))
    }
}
impl From<DetectorBank> for BankData {
    fn from(v: DetectorBank) -> Self {
        v.0
    }
}
