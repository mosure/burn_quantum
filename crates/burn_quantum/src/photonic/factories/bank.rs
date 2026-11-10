//! Finite latent-mode source correlations with exact joint-yield accounting.
use super::{FactoryReport, ResourceFactory};
use crate::photonic::runtime::shot_seed;
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
/// Conditional source configuration shared by one resource-generation invocation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactoryMode {
    /// Prior probability of this shared classical mode.
    pub probability: f64,
    /// Independent bounded sources conditional on the selected mode.
    pub sources: Vec<ResourceFactory>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BankData {
    version: u32,
    calibration: String,
    modes: Vec<FactoryMode>,
}
/// Correlated availability model: sample one mode, then conditional source attempts.
///
/// This models common outages/brightness changes, not entanglement or coherent
/// resource defects. Quality channels remain explicit quantum-model inputs.
/// Each invocation draws a fresh mode; persistent drift across invocations
/// requires a separate temporal model. Sources run concurrently within a mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "BankData", into = "BankData")]
pub struct CorrelatedFactoryBank(BankData);
/// One generation invocation, retaining the latent mode and every source attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactoryBankReport {
    /// Shared sampled mode, useful for synthetic-calibration studies.
    pub mode: usize,
    /// Source outcomes in configured order, including exhausted failures.
    pub sources: Vec<FactoryReport>,
}
impl CorrelatedFactoryBank {
    /// Validate a complete finite mixture and stable calibration identity.
    pub fn new(calibration: String, modes: Vec<FactoryMode>) -> Result<Self> {
        Self::try_from(BankData {
            version: 1,
            calibration,
            modes,
        })
    }
    /// Exact probability all requested sources become available within their own budgets.
    pub fn joint_yield(&self) -> f64 {
        self.0
            .modes
            .iter()
            .map(|m| {
                m.probability
                    * m.sources
                        .iter()
                        .map(ResourceFactory::yield_probability)
                        .product::<f64>()
            })
            .sum()
    }
    /// Immutable calibration identity; callers include it in compiled/job provenance.
    pub fn calibration(&self) -> &str {
        &self.0.calibration
    }
    /// Sample one shared latent mode and independent conditional source histories.
    /// `invocation` distinguishes fresh-resource retries within the same global shot.
    /// Every mode's full source budgets must fit before `deadline`, including
    /// zero-probability modes, so validation never depends on a stochastic draw.
    pub fn generate(
        &self,
        seed: u64,
        shot: u64,
        invocation: u64,
        start: u64,
        deadline: u64,
    ) -> Result<FactoryBankReport> {
        for mode in &self.0.modes {
            for source in &mode.sources {
                let budget_end = start
                    .checked_add(source.worst_case_ticks())
                    .ok_or(Error::Overflow)?;
                if budget_end > deadline {
                    return Err(Error::InvalidValue(
                        "factory bank worst-case completion exceeds deadline".into(),
                    ));
                }
            }
        }
        let domain = shot_seed(seed, shot, 0x4641_4354_4241_4e4b);
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(domain, invocation, 0));
        let draw = rng.random::<f64>();
        let mut cumulative = 0.;
        let mut selected = self.0.modes.len() - 1;
        for (index, mode) in self.0.modes.iter().enumerate() {
            cumulative += mode.probability;
            if draw < cumulative {
                selected = index;
                break;
            }
        }
        let sources = self.0.modes[selected]
            .sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                source.generate(
                    shot_seed(domain, invocation, 1),
                    shot,
                    index as u64,
                    start,
                    deadline,
                )
            })
            .collect::<Result<_>>()?;
        Ok(FactoryBankReport {
            mode: selected,
            sources,
        })
    }
}
impl TryFrom<BankData> for CorrelatedFactoryBank {
    type Error = Error;
    fn try_from(v: BankData) -> Result<Self> {
        let sources = v.modes.first().map_or(0, |m| m.sources.len());
        if v.version != 1
            || v.calibration.is_empty()
            || v.calibration.len() > 4096
            || v.modes.is_empty()
            || v.modes.len() > 256
            || sources == 0
            || sources > 4096
            || v.modes.len() * sources > 16384
            || v.modes.iter().any(|m| {
                m.sources.len() != sources
                    || !m.probability.is_finite()
                    || !(0.0..=1.).contains(&m.probability)
            })
            || (v.modes.iter().map(|m| m.probability).sum::<f64>() - 1.).abs() > 1e-12
        {
            return Err(Error::InvalidValue(
                "invalid correlated factory mixture".into(),
            ));
        }
        Ok(Self(v))
    }
}
impl From<CorrelatedFactoryBank> for BankData {
    fn from(v: CorrelatedFactoryBank) -> Self {
        v.0
    }
}
