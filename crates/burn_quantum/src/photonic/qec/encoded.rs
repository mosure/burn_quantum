//! Four-photon Shor graph encoding and destructive transversal Bell parity recovery.
use super::gf2::solve;
use crate::photonic::resource::MAX_RESOURCE_QUBITS;
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::photonic::{
    FusionOutcome, FusionParities, FusionProgram, NodeId, Parity, PhotonId, ResourceState,
};
use crate::{Error, Result};

/// One `[[4,1,2]]` block with ordered, distinct stable physical photon identities.
/// Stabilizers are XXXX, Z0Z2, Z1Z3; logical representatives are X0X2 and Z0Z1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShorBlock([PhotonId; 4], ShorOrientation);
/// Physical repetition order, with explicitly defined canonical logical operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShorOrientation {
    /// XXXX,Z0Z2,Z1Z3; logical X0X2 and Z0Z1.
    Standard,
    /// ZZZZ,X0X2,X1X3; logical X0X1 and Z0Z2.
    /// A physical Hadamard on every site maps Standard to Dual and implements logical H.
    Dual,
}
impl ShorBlock {
    /// Check that the four physical identities are distinct. Program membership is
    /// checked when the block is consumed by `EncodedFusion::append`.
    pub fn new(photons: [PhotonId; 4]) -> Result<Self> {
        if photons
            .iter()
            .enumerate()
            .any(|(i, q)| photons[..i].contains(q))
        {
            return Err(Error::Wires("repeated Shor block photon".into()));
        }
        Ok(Self(photons, ShorOrientation::Standard))
    }
    /// Construct a block in the declared physical code orientation.
    pub fn with_orientation(photons: [PhotonId; 4], orientation: ShorOrientation) -> Result<Self> {
        let mut block = Self::new(photons)?;
        block.1 = orientation;
        Ok(block)
    }
    /// Physical code orientation and its canonical logical Pauli convention.
    pub fn orientation(&self) -> ShorOrientation {
        self.1
    }
    /// Ordered physical identities; code coordinates must not be arbitrarily permuted.
    pub fn photons(&self) -> &[PhotonId; 4] {
        &self.0
    }
    /// The three positive stabilizer generators defining the code space.
    pub fn stabilizers(&self) -> [PauliProduct; 3] {
        let (a, b) = if self.1 == ShorOrientation::Standard {
            (Pauli::X, Pauli::Z)
        } else {
            (Pauli::Z, Pauli::X)
        };
        [
            self.product(&[0, 1, 2, 3], a),
            self.product(&[0, 2], b),
            self.product(&[1, 3], b),
        ]
    }
    /// Positive representative of logical X in the canonical orientation convention.
    pub fn logical_x(&self) -> PauliProduct {
        self.product(
            if self.1 == ShorOrientation::Standard {
                &[0, 2]
            } else {
                &[0, 1]
            },
            Pauli::X,
        )
    }
    /// Positive representative of logical Z in the canonical orientation convention.
    pub fn logical_z(&self) -> PauliProduct {
        self.product(
            if self.1 == ShorOrientation::Standard {
                &[0, 1]
            } else {
                &[0, 2]
            },
            Pauli::Z,
        )
    }
    fn product(&self, sites: &[usize], pauli: Pauli) -> PauliProduct {
        PauliProduct::new(sites.iter().map(|&q| (self.0[q].0, pauli)).collect())
            .expect("distinct checked block sites")
    }
}
/// A supplied graph resource with every logical vertex encoded in four physical photons.
/// Source-generation yield and circuit noise are deliberately not inferred.
#[derive(Clone, Debug)]
pub struct ShorEncodedResource {
    state: ResourceState,
    orientations: Vec<ShorOrientation>,
}
impl ShorEncodedResource {
    /// Encode a graph and its declared local logical Hadamards. Every edge becomes all four edges
    /// between block ports {0,1}; each block has Bell edges (0,2),(1,3) and H on 2,3.
    /// A local logical H is represented by physical H on all four block photons
    /// together with the explicit dual-code orientation, not a single physical H.
    /// Resource descriptions remain bounded to 64 physical photons (16 logical sites).
    pub fn new(graph: &ResourceState) -> Result<Self> {
        let n = graph.qubits().checked_mul(4).ok_or(Error::Overflow)?;
        if n > MAX_RESOURCE_QUBITS {
            return Err(Error::ResourceLimit {
                resource: "encoded resource photons",
                required: n,
                limit: MAX_RESOURCE_QUBITS,
            });
        }
        let mut edges = Vec::new();
        let mut hadamards = Vec::new();
        for q in 0..graph.qubits() {
            edges.extend([(4 * q, 4 * q + 2), (4 * q + 1, 4 * q + 3)]);
            hadamards.extend(if graph.hadamards().contains(&q) {
                [4 * q, 4 * q + 1]
            } else {
                [4 * q + 2, 4 * q + 3]
            });
        }
        for &(a, b) in graph.edges() {
            for i in 0..2 {
                for j in 0..2 {
                    edges.push((4 * a + i, 4 * b + j));
                }
            }
        }
        Ok(Self {
            state: ResourceState::graph(n, edges)?.with_local_hadamards(&hadamards)?,
            orientations: (0..graph.qubits())
                .map(|q| {
                    if graph.hadamards().contains(&q) {
                        ShorOrientation::Dual
                    } else {
                        ShorOrientation::Standard
                    }
                })
                .collect(),
        })
    }
    /// Actual small resource graph and local physical Hadamards, usable by every executor.
    pub fn resource(&self) -> &ResourceState {
        &self.state
    }
    /// Number of encoded logical graph vertices.
    pub fn logical_qubits(&self) -> usize {
        self.orientations.len()
    }
    /// Supply this actual physical resource and return stable typed block identities.
    pub fn add_to(&self, program: &mut FusionProgram, source: NodeId) -> Result<Vec<ShorBlock>> {
        let photons = program.add_resource(self.state.clone(), source)?;
        Ok(photons
            .chunks_exact(4)
            .zip(&self.orientations)
            .map(|(p, &orientation)| ShorBlock([p[0], p[1], p[2], p[3]], orientation))
            .collect())
    }
}
/// Recovery assumes input code stabilizers hold and unknown data are erasures.
/// Unheralded Pauli faults may violate checks or flip logical values; this is not
/// a Pauli-error-correcting decoder or a guaranteed successful logical measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodedFusionRecovery {
    /// Available logical XX and ZZ parities; each unavailable value remains an erasure.
    Consistent(FusionParities),
    /// Supplied records contradict an input-code stabilizer relation.
    Inconsistent,
}
/// Four physical unboosted destructive fusions implementing an encoded Bell attempt.
/// Eight supplied signal photons are consumed; no ancillary photons or success coin is added.
#[derive(Clone, Debug)]
pub struct EncodedFusion {
    indices: [usize; 4],
    blocks: [ShorBlock; 2],
    code: FusionCode,
}
impl EncodedFusion {
    /// Append four pairwise-coordinate fusions atomically. Any absent/consumed/overlapping
    /// block causes failure without consuming any additional program ports.
    pub fn append(
        program: &mut FusionProgram,
        left: ShorBlock,
        right: ShorBlock,
        analyzer: NodeId,
    ) -> Result<Self> {
        let start = program.num_fusions();
        let mut candidate = program.clone();
        let pairing = pairing(left.1, right.1);
        for (i, &j) in pairing.iter().enumerate() {
            candidate.fuse(left.0[i], right.0[j], analyzer)?;
        }
        *program = candidate;
        Ok(Self {
            indices: [start, start + 1, start + 2, start + 3],
            blocks: [left, right],
            code: FusionCode::new(left.1, right.1),
        })
    }
    /// Global program operation indices of the four physical fusions.
    pub fn physical_fusions(&self) -> &[usize; 4] {
        &self.indices
    }
    /// Physical signal photons consumed per attempted encoded fusion, including failures.
    pub fn photons_consumed(&self) -> usize {
        8
    }
    /// Logical XX and ZZ observables on the two input blocks.
    pub fn logical_observables(&self) -> [PauliProduct; 2] {
        [
            self.blocks[0]
                .logical_x()
                .multiply_commuting(&self.blocks[1].logical_x())
                .expect("disjoint fused blocks")
                .1,
            self.blocks[0]
                .logical_z()
                .multiply_commuting(&self.blocks[1].logical_z())
                .expect("disjoint fused blocks")
                .1,
        ]
    }
    /// Recover from complete per-program physical records using the recorded operation indices.
    pub fn recover(&self, records: &[FusionOutcome]) -> Result<EncodedFusionRecovery> {
        let mut local = [FusionOutcome::Loss; 4];
        for (i, &index) in self.indices.iter().enumerate() {
            local[i] = *records
                .get(index)
                .ok_or_else(|| Error::Shape("missing physical encoded-fusion record".into()))?;
        }
        Ok(self.code.recover_outcomes(local))
    }
    /// Recover from four ordered actual standard-fusion records. Individual failure bits
    /// additionally check each input block's available Z0Z2/Z1Z3 stabilizers.
    pub fn recover_outcomes(outcomes: [FusionOutcome; 4]) -> EncodedFusionRecovery {
        default_code().recover_outcomes(outcomes)
    }
    /// Recover local physical parity records using this fusion's actual code orientations
    /// and coordinate pairing. No inferred parity is substituted for an unresolved erasure.
    pub fn recover_local_parities(&self, records: [FusionParities; 4]) -> EncodedFusionRecovery {
        self.code.recover_parities(records)
    }
    /// Standard/standard complete-Bell reference recovery, preserving the original convention.
    pub fn recover_parities(records: [FusionParities; 4]) -> EncodedFusionRecovery {
        default_code().recover_parities(records)
    }
}
fn pairing(a: ShorOrientation, b: ShorOrientation) -> [usize; 4] {
    if a == b { [0, 1, 2, 3] } else { [0, 2, 1, 3] }
}
fn default_code() -> &'static FusionCode {
    static CODE: std::sync::OnceLock<FusionCode> = std::sync::OnceLock::new();
    CODE.get_or_init(|| FusionCode::new(ShorOrientation::Standard, ShorOrientation::Standard))
}
#[derive(Clone, Debug)]
struct FusionCode {
    checks: Vec<([bool; 8], bool)>,
    logicals: [[usize; 2]; 2],
    z_checks: Vec<Vec<usize>>,
}
impl FusionCode {
    fn new(left: ShorOrientation, right: ShorOrientation) -> Self {
        let a = ShorBlock([PhotonId(0), PhotonId(1), PhotonId(2), PhotonId(3)], left);
        let map = pairing(left, right); // This pairing is its own inverse.
        let b = ShorBlock(map.map(PhotonId), right);
        let group = |block: ShorBlock| {
            (0..8)
                .map(|mask| {
                    let mut product = PauliProduct::new(vec![]).unwrap();
                    let mut sign = false;
                    for (j, g) in block.stabilizers().iter().enumerate() {
                        if mask >> j & 1 != 0 {
                            let (s, p) = product.multiply_commuting(g).unwrap();
                            sign ^= s;
                            product = p;
                        }
                    }
                    (sign, product)
                })
                .collect::<Vec<_>>()
        };
        let mut checks = vec![];
        for (sa, pa) in group(a) {
            if pa.terms().is_empty() {
                continue;
            }
            for (sb, pb) in group(b) {
                if pa == pb {
                    let mut row = [false; 8];
                    let mut negative = sa ^ sb;
                    for &(q, p) in pa.terms() {
                        if p != Pauli::Z {
                            row[2 * q] = true;
                        }
                        if p != Pauli::X {
                            row[2 * q + 1] = true;
                        }
                        negative ^= p == Pauli::Y;
                    }
                    checks.push((row, negative));
                }
            }
        }
        let logicals = [a.logical_x(), a.logical_z()].map(|p| {
            let indices: Vec<_> = p
                .terms()
                .iter()
                .map(|&(q, p)| 2 * q + usize::from(p == Pauli::Z))
                .collect();
            [indices[0], indices[1]]
        });
        let z_checks = [a, b]
            .into_iter()
            .flat_map(|block| block.stabilizers())
            .filter(|p| p.terms().iter().all(|&(_, s)| s == Pauli::Z))
            .map(|p| p.terms().iter().map(|&(q, _)| q).collect())
            .collect();
        Self {
            checks,
            logicals,
            z_checks,
        }
    }
    fn recover_outcomes(&self, outcomes: [FusionOutcome; 4]) -> EncodedFusionRecovery {
        for check in &self.z_checks {
            let mut bit = false;
            let mut complete = true;
            for &q in check {
                if let FusionOutcome::Failure { bit: v } = outcomes[q] {
                    bit ^= v;
                } else {
                    complete = false;
                }
            }
            if complete && bit {
                return EncodedFusionRecovery::Inconsistent;
            }
        }
        self.recover_parities(outcomes.map(FusionOutcome::parities))
    }
    fn recover_parities(&self, records: [FusionParities; 4]) -> EncodedFusionRecovery {
        let values: Vec<_> = records
            .iter()
            .flat_map(|r| [r.xx, r.zz].map(|p| p.map(|v| v == Parity::Odd)))
            .collect();
        let unknown: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_none().then_some(i))
            .collect();
        let matrix: Vec<Vec<_>> = self
            .checks
            .iter()
            .map(|(r, _)| unknown.iter().map(|&i| r[i]).collect())
            .collect();
        let rhs: Vec<_> = self
            .checks
            .iter()
            .map(|(r, sign)| {
                r.iter()
                    .zip(&values)
                    .fold(*sign, |s, (&c, v)| s ^ (c && v.unwrap_or(false)))
            })
            .collect();
        let Some((solution, kernel)) = solve(&matrix, unknown.len(), &rhs) else {
            return EncodedFusionRecovery::Inconsistent;
        };
        let infer = |support: [usize; 2]| {
            let selected: Vec<_> = unknown.iter().map(|i| support.contains(i)).collect();
            if kernel.iter().any(|v| {
                v.iter()
                    .zip(&selected)
                    .fold(false, |s, (&a, &b)| s ^ (a && b))
            }) {
                return None;
            }
            let bit = support
                .iter()
                .fold(false, |s, &i| s ^ values[i].unwrap_or(false))
                ^ solution
                    .iter()
                    .zip(&selected)
                    .fold(false, |s, (&a, &b)| s ^ (a && b));
            Some(if bit { Parity::Odd } else { Parity::Even })
        };
        EncodedFusionRecovery::Consistent(FusionParities {
            xx: infer(self.logicals[0]),
            zz: infer(self.logicals[1]),
        })
    }
}
