//! Calibrated resource preparation and optical instrument execution.
use super::*;
use crate::KrausChannel;
use crate::photonic::{factories::ResourceFactory, optics::OpticalBellInstrument};
use serde::Serialize;

/// Heralded generation and conditional quantum quality of a gate's supplied Choi state.
/// The noise channel acts on ordered measured halves then output halves, after the ideal
/// gate is encoded. It can include correlated defects, vacuum, coherent noise and mixtures.
#[derive(Clone, Debug, Serialize)]
pub struct ResourceFactoryModel {
    calibration: String,
    factory: ResourceFactory,
    noise: Option<KrausChannel>,
}
impl ResourceFactoryModel {
    /// A calibrated source. `noise=None` declares an ideal emitted resource, not unit yield.
    pub fn new(
        calibration: String,
        factory: ResourceFactory,
        noise: Option<KrausChannel>,
    ) -> Result<Self> {
        if calibration.is_empty() || calibration.len() > 4096 {
            return Err(Error::InvalidValue(
                "invalid resource calibration identity".into(),
            ));
        }
        Ok(Self {
            calibration,
            factory,
            noise,
        })
    }
    /// Bounded availability/cost model. Exhaustion emits no state and terminates the attempt.
    pub fn factory(&self) -> &ResourceFactory {
        &self.factory
    }
    /// Calibration identity retained in compiled provenance.
    pub fn calibration(&self) -> &str {
        &self.calibration
    }
    /// Conditional emitted-resource channel, in measured-then-output site order.
    pub fn noise(&self) -> Option<&KrausChannel> {
        self.noise.as_ref()
    }
    fn validate(&self, k: usize) -> Result<()> {
        if self.factory.photons_per_attempt() < 2 * k {
            return Err(Error::InvalidValue(
                "resource factory photon cost is below supplied resource size".into(),
            ));
        }
        if self
            .noise
            .as_ref()
            .is_some_and(|c| c.operators()[0].dimension() != 3usize.pow((2 * k) as u32))
        {
            return Err(Error::Shape(
                "resource noise must act on all measured/output dual-rail sites".into(),
            ));
        }
        Ok(())
    }
}
/// Complete optical/source calibration used for scheduling and exact noisy execution.
/// Factory attempts run in a fixed worst-case window; early success is released at its
/// end. This conservative policy avoids stochastic schedule changes. Conditional noise
/// describes the resource at release, including any internally managed source storage.
#[derive(Clone, Debug, Serialize)]
pub struct PhysicalCalibration {
    local: ResourceFactoryModel,
    entangling: ResourceFactoryModel,
    analyzer: OpticalBellInstrument,
    max_kraus_terms: usize,
}
impl PhysicalCalibration {
    /// Sources for two- and four-photon gate resources, a complete destructive Bell POVM,
    /// and a per-gate bound on contracted Kraus terms (before tensor execution).
    pub fn new(
        local: ResourceFactoryModel,
        entangling: ResourceFactoryModel,
        analyzer: OpticalBellInstrument,
        max_kraus_terms: usize,
    ) -> Result<Self> {
        local.validate(1)?;
        entangling.validate(2)?;
        if max_kraus_terms == 0 || max_kraus_terms > 1_000_000 {
            return Err(Error::InvalidValue(
                "invalid calibrated Kraus work budget".into(),
            ));
        }
        Ok(Self {
            local,
            entangling,
            analyzer,
            max_kraus_terms,
        })
    }
    /// Exact analyzer and its physical ancillary-photon count.
    pub fn analyzer(&self) -> &OpticalBellInstrument {
        &self.analyzer
    }
    /// Source model selected by gate arity (one or two logical sites).
    pub fn resource(&self, arity: usize) -> Result<&ResourceFactoryModel> {
        match arity {
            1 => Ok(&self.local),
            2 => Ok(&self.entangling),
            _ => Err(Error::Unsupported(
                "calibrated resources support one or two logical sites".into(),
            )),
        }
    }
    pub(super) fn max_terms(&self) -> usize {
        self.max_kraus_terms
    }
}
