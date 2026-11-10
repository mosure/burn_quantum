//! Friendly resource/fusion programs with stable photon identities and checked persistence.
mod compiler;
mod timing;
use super::{Fabric, NodeId, ResourceState};
use crate::{Error, Result};
pub use compiler::{CompileOptions, CompiledFusionNetwork, RoutedPhoton, ScheduledFusion};
use serde::{Deserialize, Serialize};
pub use timing::{SchedulePolicy, TimingOptions};

/// Stable photon identity within one program; IDs survive deletion of measured sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhotonId(pub usize);

/// Supplied resource recipe; its photons occupy the next contiguous creation IDs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceInstance {
    /// Ideal resource state before transport.
    pub state: ResourceState,
    /// Source port, checked against the fabric when compiling.
    pub source: NodeId,
}
/// Destructive fusion request, validated in the context of its containing program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionRequest {
    /// Two distinct live photons, in measurement target order.
    pub photons: [PhotonId; 2],
    /// Analyzer destination, checked against the fabric when compiling.
    pub at: NodeId,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramData {
    version: u32,
    resources: Vec<ResourceInstance>,
    fusions: Vec<FusionRequest>,
}

/// Static FBQC network builder: supplied resources followed by destructive fusions.
///
/// This frontend expresses resource-level algorithms (e.g. entanglement swapping).
/// It does not silently translate arbitrary gate-model algorithms or retry a failed
/// fusion. All supplied resources are born at tick zero; all unconsumed photons
/// are outputs in creation order. Compiling does not allocate a quantum state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ProgramData", into = "ProgramData")]
pub struct FusionProgram {
    pub(crate) resources: Vec<ResourceInstance>,
    pub(crate) fusions: Vec<FusionRequest>,
    photons: usize,
    // Derived on validated deserialization; never accepted as trusted persisted data.
    // Constant-time consumption checks keep large lattice construction linear.
    consumed: Vec<bool>,
}
impl FusionProgram {
    /// Empty builder. A program must contain at least one resource to compile.
    pub fn new() -> Self {
        Self::default()
    }
    /// Supply an ideal resource at a source port and return its stable photon IDs.
    /// Source IDs are checked against the selected fabric at compilation.
    pub fn add_resource(&mut self, state: ResourceState, source: NodeId) -> Result<Vec<PhotonId>> {
        let end = self
            .photons
            .checked_add(state.qubits())
            .ok_or(Error::Overflow)?;
        if end > 1_048_576 {
            return Err(Error::ResourceLimit {
                resource: "program photons",
                required: end,
                limit: 1_048_576,
            });
        }
        let ids = (self.photons..end).map(PhotonId).collect();
        self.resources.push(ResourceInstance { state, source });
        self.consumed.resize(end, false);
        self.photons = end;
        Ok(ids)
    }
    /// Consume two distinct, still-live photons at a fusion node. Failure is atomic.
    pub fn fuse(&mut self, left: PhotonId, right: PhotonId, at: NodeId) -> Result<&mut Self> {
        if left == right
            || left.0 >= self.photons
            || right.0 >= self.photons
            || self.consumed[left.0]
            || self.consumed[right.0]
        {
            return Err(Error::Wires(
                "fusion references an absent, repeated or consumed photon".into(),
            ));
        }
        self.fusions.push(FusionRequest {
            photons: [left, right],
            at,
        });
        self.consumed[left.0] = true;
        self.consumed[right.0] = true;
        Ok(self)
    }
    /// Total physical photons supplied, before loss and destructive measurements.
    pub fn num_photons(&self) -> usize {
        self.photons
    }
    /// Number of two-photon destructive operations.
    pub fn num_fusions(&self) -> usize {
        self.fusions.len()
    }
    /// Read-only resource recipes in photon creation order, without dense allocation.
    pub fn resources(&self) -> &[ResourceInstance] {
        &self.resources
    }
    /// Read-only fusion requests in execution order.
    pub fn fusions(&self) -> &[FusionRequest] {
        &self.fusions
    }
    /// Compile a static network with heralded resource births and classical dependencies.
    pub fn compile_timed(
        &self,
        fabric: &Fabric,
        options: CompileOptions,
        timing: TimingOptions,
    ) -> Result<CompiledFusionNetwork> {
        compiler::compile_timed(self, fabric, options, timing)
    }
    /// Compile a supplied consumption schedule with finite packet buffers and switch delay.
    ///
    /// Greedy operand routing is bounded and atomic. An infeasible greedy placement is
    /// rejected; this is a schedule validator/router, not a global throughput optimizer.
    /// All outputs occupy their source buffers until the schedule finishes.
    pub fn compile_finite(
        &self,
        fabric: &Fabric,
        options: CompileOptions,
        timing: TimingOptions,
        constraints: crate::photonic::fabric::FabricConstraints,
        fusion_ticks: &[u64],
    ) -> Result<CompiledFusionNetwork> {
        compiler::compile_finite(self, fabric, options, timing, constraints, fusion_ticks)
    }
    /// Compile deterministic timed routes and attenuation under an explicit fabric.
    pub fn compile(
        &self,
        fabric: &Fabric,
        options: CompileOptions,
    ) -> Result<CompiledFusionNetwork> {
        compiler::compile(self, fabric, options)
    }
}
impl TryFrom<ProgramData> for FusionProgram {
    type Error = Error;
    fn try_from(data: ProgramData) -> Result<Self> {
        if data.version != 1 {
            return Err(Error::Unsupported(
                "unknown fusion program schema version".into(),
            ));
        }
        let mut program = Self::new();
        for resource in data.resources {
            program.add_resource(resource.state, resource.source)?;
        }
        for fusion in data.fusions {
            program.fuse(fusion.photons[0], fusion.photons[1], fusion.at)?;
        }
        Ok(program)
    }
}
impl From<FusionProgram> for ProgramData {
    fn from(v: FusionProgram) -> Self {
        Self {
            version: 1,
            resources: v.resources,
            fusions: v.fusions,
        }
    }
}
