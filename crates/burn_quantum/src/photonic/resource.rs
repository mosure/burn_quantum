//! Validated ideal small stabilizer resource states, independent of a tensor device.
use super::DualRail;
use crate::{Circuit, Error, Result, Unitary};
use serde::{Deserialize, Serialize};

/// Maximum qubits in a supplied resource description; dense execution has tighter budgets.
pub const MAX_RESOURCE_QUBITS: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceData {
    qubits: usize,
    edges: Vec<(usize, usize)>,
    hadamards: Vec<usize>,
}

/// Graph state followed by specified local Hadamards, with no source-generation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ResourceData", into = "ResourceData")]
pub struct ResourceState(ResourceData);
impl ResourceState {
    /// Four-photon GHZ resource in the X basis: ZZZZ and adjacent XX stabilizers.
    /// This is the four-star resource convention used in FBQC edge/face networks.
    pub fn four_star() -> Self {
        Self(ResourceData {
            qubits: 4,
            edges: vec![(0, 1), (0, 2), (0, 3)],
            hadamards: vec![0],
        })
    }
    /// Append local Hadamards, cancelling existing Hadamards on the same sites.
    /// Repeated or absent site arguments are rejected before changing the recipe.
    pub fn with_local_hadamards(&self, sites: &[usize]) -> Result<Self> {
        let mut sites = sites.to_vec();
        sites.sort_unstable();
        if sites.iter().any(|&q| q >= self.qubits()) || sites.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::Wires("invalid local Hadamard sites".into()));
        }
        let mut data = self.0.clone();
        for q in sites {
            if let Some(i) = data.hadamards.iter().position(|&s| s == q) {
                data.hadamards.remove(i);
            } else {
                data.hadamards.push(q);
            }
        }
        Self::try_from(data)
    }
    /// Ideal graph state: |+> on every site, followed by CZ on every edge.
    pub fn graph(qubits: usize, edges: Vec<(usize, usize)>) -> Result<Self> {
        Self::try_from(ResourceData {
            qubits,
            edges,
            hadamards: vec![],
        })
    }
    /// Ideal |Phi+>=(|00>+|11>)/sqrt(2), not the locally equivalent graph edge.
    pub fn bell_pair() -> Self {
        Self(ResourceData {
            qubits: 2,
            edges: vec![(0, 1)],
            hadamards: vec![1],
        })
    }
    /// Six-site ring graph resource, with edges (0,1),...,(5,0).
    pub fn six_ring() -> Self {
        Self::graph(6, (0..6).map(|i| (i, (i + 1) % 6)).collect()).unwrap()
    }
    /// Number of physical dual-rail photons supplied ideally.
    pub fn qubits(&self) -> usize {
        self.0.qubits
    }
    /// Canonically ordered undirected graph edges.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.0.edges
    }
    /// Sites receiving a Hadamard after graph preparation, in canonical order.
    pub fn hadamards(&self) -> &[usize] {
        &self.0.hadamards
    }
    /// Build a dense-reference preparation circuit; resource generation itself is idealized.
    pub fn circuit(&self) -> Result<Circuit> {
        let mut circuit = Circuit::new(DualRail::register(self.qubits())?, 0, 0);
        self.append_to(&mut circuit, 0)?;
        Ok(circuit)
    }
    pub(crate) fn append_to(&self, circuit: &mut Circuit, offset: usize) -> Result<()> {
        let h = DualRail::unitary(&Unitary::h())?;
        for wire in 0..self.qubits() {
            circuit.unitary(vec![offset + wire], h.clone())?;
        }
        for &(a, b) in &self.0.edges {
            circuit.phase_flip(vec![(offset + a, 1), (offset + b, 1)])?;
        }
        for &wire in &self.0.hadamards {
            circuit.unitary(vec![offset + wire], h.clone())?;
        }
        Ok(())
    }
}
impl TryFrom<ResourceData> for ResourceState {
    type Error = Error;
    fn try_from(mut data: ResourceData) -> Result<Self> {
        if data.qubits == 0 || data.qubits > MAX_RESOURCE_QUBITS {
            return Err(Error::InvalidDimensions(
                "resource size must be in 1..=64".into(),
            ));
        }
        for edge in &mut data.edges {
            if edge.0 == edge.1 || edge.0 >= data.qubits || edge.1 >= data.qubits {
                return Err(Error::Wires("invalid resource edge".into()));
            }
            if edge.0 > edge.1 {
                *edge = (edge.1, edge.0);
            }
        }
        data.edges.sort_unstable();
        data.hadamards.sort_unstable();
        if data.hadamards.windows(2).any(|x| x[0] == x[1])
            || data.edges.windows(2).any(|x| x[0] == x[1])
            || data.hadamards.iter().any(|&w| w >= data.qubits)
        {
            return Err(Error::Wires(
                "duplicate edge or invalid local Clifford".into(),
            ));
        }
        Ok(Self(data))
    }
}
impl From<ResourceState> for ResourceData {
    fn from(v: ResourceState) -> Self {
        v.0
    }
}
