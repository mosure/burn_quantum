use super::SixRingBoundaryNetwork;
use crate::photonic::qec::{MeasurementRelations, ParityRelation, gf2};
use crate::photonic::stabilizer::{Pauli, PauliProduct, Tableau};
use crate::{Error, Result};
/// Explicit work limits for the derived finite boundary proof.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryCertificateLimits {
    /// Dense GF(2) entries in each signed measurement-relation calculation.
    pub max_relation_entries: usize,
    /// Candidate Pauli supports examined per X/Z distance search.
    pub max_distance_candidates: usize,
    /// Largest support weight searched; exhaustion is an error, never a distance claim.
    pub max_distance_weight: usize,
}
impl Default for BoundaryCertificateLimits {
    fn default() -> Self {
        Self {
            max_relation_entries: 1 << 24,
            max_distance_candidates: 1 << 24,
            max_distance_weight: 8,
        }
    }
}
/// Exact minimum weight of a logical Pauli, with a witness and auditable work.
#[derive(Clone, Debug)]
pub struct BoundaryDistance {
    /// Exact minimum weight after excluding every smaller support.
    pub distance: usize,
    /// Global physical photon identities of one minimum-weight witness.
    pub photons: Vec<usize>,
    /// Minimum-weight witness in actual physical Pauli axes.
    pub operator: PauliProduct,
    /// Number of candidate supports actually tested.
    pub candidates: usize,
}
/// One derived CSS end code. No code-space pattern is assumed before elimination.
#[derive(Clone, Debug)]
pub struct BoundaryCode {
    /// Actual physical output photons, in binary coordinate order.
    pub photons: Vec<usize>,
    /// Photons whose local H defines the CSS coordinates; this is a declared frame,
    /// not an uncounted physical operation. Checks below are mapped back to physical axes.
    pub hadamard_frame: Vec<usize>,
    /// Independent checks that are pure X in the declared CSS frame, stored in physical axes.
    /// Each carries its signed physical measurement relation.
    pub x_checks: Vec<(PauliProduct, ParityRelation)>,
    /// Independent checks that are pure Z in the declared CSS frame, stored in physical axes.
    /// Each carries its signed physical measurement relation.
    pub z_checks: Vec<(PauliProduct, ParityRelation)>,
    /// A logical X representative, anticommuting with logical_z.
    pub logical_x: PauliProduct,
    /// A logical Z representative.
    pub logical_z: PauliProduct,
    /// Exact minimum X logical support.
    pub x_distance: BoundaryDistance,
    /// Exact minimum Z logical support.
    pub z_distance: BoundaryDistance,
}
/// Complete ideal output-code and logical-channel certificate of the actual fusion graph.
#[derive(Clone, Debug)]
pub struct BoundaryCertificate {
    /// Time end 0 blocks first, then time end 1 blocks.
    pub codes: Vec<BoundaryCode>,
    /// Independent logical Choi stabilizers, including all signs from actual records.
    pub logical_relations: Vec<(PauliProduct, ParityRelation)>,
    /// The same independent relations in encoded coordinates: code index selects the qubit.
    /// Pauli axes refer to each code's pure-X / pure-Z representatives, so local
    /// Hadamards are explicit rather than absorbed into a claimed identity channel.
    pub logical_observables: Vec<PauliProduct>,
    /// Encoded Pauli = (-1)^this_bit times the canonical physical target.
    /// XOR this lift phase into `logical_relations[i].1.negative` to obtain the
    /// signed relation for `logical_observables[i]`, including encoded Y axes.
    pub logical_phase_negative: Vec<bool>,
    /// Encoded Bell pairs across the two time ends, computed from stabilizer rank.
    /// A two-patch parity projection transmits one; disconnected memories transmit two.
    pub transmitted_logicals: usize,
}
impl SixRingBoundaryNetwork {
    /// Exact signed relations for this resource/measurement geometry, before physical loss.
    pub fn ideal_relations(&self, max_entries: usize) -> Result<MeasurementRelations> {
        let program = self.patch.program();
        let mut initial = Tableau::zero(program.num_photons(), Default::default())?;
        for cell in 0..self.centers.len() {
            for port in 0..6 {
                initial.h(6 * cell + port)?;
            }
            for port in 0..6 {
                initial.cz(6 * cell + port, 6 * cell + (port + 1) % 6)?;
            }
        }
        MeasurementRelations::derive(&initial, &self.patch.observables()?, max_entries)
    }
    /// Derive CSS checks, exact bounded X/Z distances and the complete signed logical channel.
    /// End blocks of at most 128 photons are supported by the distance verifier.
    /// This is an ideal geometry proof, not a noisy-fusion threshold certificate.
    pub fn certify(&self, limits: BoundaryCertificateLimits) -> Result<BoundaryCertificate> {
        self.certify_in_frame(limits, &[])
    }
    pub(super) fn certify_in_frame(
        &self,
        limits: BoundaryCertificateLimits,
        hadamards: &[usize],
    ) -> Result<BoundaryCertificate> {
        let relations = self.ideal_relations(limits.max_relation_entries)?;
        let mut codes = vec![];
        for end in 0..2 {
            for block in 0..self.blocks() {
                let photons = self
                    .end(end, block)?
                    .iter()
                    .map(|q| q.0)
                    .collect::<Vec<_>>();
                codes.push(code(&relations, photons, limits, hadamards)?);
            }
        }
        let outputs = codes
            .iter()
            .flat_map(|c| c.photons.iter().copied())
            .collect::<Vec<_>>();
        let all = relations.output_relations(&outputs, limits.max_relation_entries)?;
        let mut basis = Vec::new();
        let mut logical_relations = vec![];
        let mut logical_vectors = vec![];
        let mut logical_observables = vec![];
        let mut logical_phase_negative = vec![];
        for (p, _) in all {
            let mut bits = 0u128;
            for (i, c) in codes.iter().enumerate() {
                if p.anticommutes(&c.logical_z) {
                    bits |= 1 << (2 * i);
                }
                if p.anticommutes(&c.logical_x) {
                    bits |= 1 << (2 * i + 1);
                }
            }
            if insert(&mut basis, bits) {
                let mut terms = std::collections::BTreeMap::new();
                for (i, c) in codes.iter().enumerate() {
                    for (flag, operator) in [
                        (1u128 << (2 * i), &c.logical_x),
                        (1u128 << (2 * i + 1), &c.logical_z),
                    ] {
                        if bits & flag != 0 {
                            for &(q, p) in operator.terms() {
                                let entry = terms.entry(q).or_insert((false, false));
                                entry.0 ^= p != Pauli::Z;
                                entry.1 ^= p != Pauli::X;
                            }
                        }
                    }
                }
                let canonical = PauliProduct::new(
                    terms
                        .into_iter()
                        .filter_map(|(q, b)| match b {
                            (true, false) => Some((q, Pauli::X)),
                            (false, true) => Some((q, Pauli::Z)),
                            (true, true) => Some((q, Pauli::Y)),
                            _ => None,
                        })
                        .collect(),
                )?;
                let relation = relations
                    .correlation(&canonical)?
                    .ok_or_else(|| Error::InvalidValue("logical Choi relation vanished".into()))?;
                logical_observables.push(PauliProduct::new(
                    (0..codes.len())
                        .filter_map(|i| {
                            match ((bits >> (2 * i)) & 1 != 0, (bits >> (2 * i + 1)) & 1 != 0) {
                                (true, false) => Some((i, Pauli::X)),
                                (false, true) => Some((i, Pauli::Z)),
                                (true, true) => Some((i, Pauli::Y)),
                                _ => None,
                            }
                        })
                        .collect(),
                )?);
                let lift_negative = codes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (bits >> (2 * i)) & 3 == 3)
                    .fold(false, |sign, (_, c)| {
                        sign ^ y_lift_negative(&c.logical_x, &c.logical_z)
                    });
                logical_phase_negative.push(lift_negative);
                logical_vectors.push(bits);
                logical_relations.push((canonical, relation));
            }
        }
        if logical_relations.len() != codes.len() {
            return Err(Error::InvalidValue(
                "boundary output is not a pure complete encoded Choi state".into(),
            ));
        }
        let left_bits = 2 * self.blocks();
        let matrix = (left_bits..2 * codes.len())
            .map(|b| {
                logical_vectors
                    .iter()
                    .map(|&v| v & (1 << b) != 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (_, kernel) =
            gf2::solve(&matrix, logical_vectors.len(), &vec![false; matrix.len()]).unwrap();
        let local_left_rank = kernel.len();
        if local_left_rank > self.blocks() {
            return Err(Error::InvalidValue(
                "invalid boundary entanglement rank".into(),
            ));
        }
        Ok(BoundaryCertificate {
            codes,
            logical_relations,
            logical_observables,
            logical_phase_negative,
            transmitted_logicals: self.blocks() - local_left_rank,
        })
    }
}
fn code(
    relations: &MeasurementRelations,
    photons: Vec<usize>,
    limits: BoundaryCertificateLimits,
    hadamards: &[usize],
) -> Result<BoundaryCode> {
    let n = photons.len();
    if n == 0 || n > 128 {
        return Err(Error::Unsupported(
            "boundary distance certificate supports 1..=128 output photons".into(),
        ));
    }
    let local = relations.output_relations(&photons, limits.max_relation_entries)?;
    if local.len() != n - 1 {
        return Err(Error::InvalidValue(
            "end block does not encode exactly one qubit".into(),
        ));
    }
    let masks = local
        .iter()
        .map(|(p, _)| mask(p, &photons, hadamards))
        .collect::<Vec<_>>();
    let mut pure: [Vec<u128>; 2] = std::array::from_fn(|_| vec![]);
    let mut checks: [Vec<(PauliProduct, ParityRelation)>; 2] = std::array::from_fn(|_| vec![]);
    for (which, out) in pure.iter_mut().enumerate() {
        let matrix = (0..n)
            .map(|q| {
                masks
                    .iter()
                    .map(|m| m[1 - which] & (1 << q) != 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (_, kernel) = gf2::solve(&matrix, masks.len(), &vec![false; n]).unwrap();
        let mut basis = vec![];
        for vector in kernel {
            let value = vector
                .iter()
                .zip(&masks)
                .filter(|(v, _)| **v)
                .fold(0, |v, (_, m)| v ^ m[which]);
            if insert(&mut basis, value) {
                // This CSS row is already a known combination of signed physical
                // output generators. Propagate their Pauli phase and XOR record
                // masks exactly instead of refactoring the full lattice again.
                let mut physical = PauliProduct::new(vec![])?;
                let mut negative = false;
                let mut terms = std::collections::BTreeSet::new();
                for (selected, (p, r)) in vector.iter().zip(&local) {
                    if *selected {
                        let (phase, product) = physical.multiply_commuting(p)?;
                        physical = product;
                        negative ^= phase ^ r.negative;
                        for &j in &r.terms {
                            if !terms.insert(j) {
                                terms.remove(&j);
                            }
                        }
                    }
                }
                let canonical = product(
                    value,
                    if which == 0 { Pauli::X } else { Pauli::Z },
                    &photons,
                    hadamards,
                )?;
                if physical != canonical {
                    return Err(Error::InvalidValue(
                        "CSS signed generator combination changed its support".into(),
                    ));
                }
                checks[which].push((
                    physical,
                    ParityRelation {
                        terms: terms.into_iter().collect(),
                        negative,
                    },
                ));
                out.push(value);
            }
        }
    }
    if pure[0].len() + pure[1].len() != n - 1 {
        return Err(Error::Unsupported(
            "derived boundary end code is not CSS".into(),
        ));
    }
    let lx = logical(&pure[1], &pure[0], n)?;
    let lz = logical(&pure[0], &pure[1], n)?;
    if (lx & lz).count_ones() % 2 != 1 {
        return Err(Error::InvalidValue(
            "boundary logical representatives commute".into(),
        ));
    }
    let x_distance = distance(&pure[1], lz, &photons, limits, Pauli::X, hadamards)?;
    let z_distance = distance(&pure[0], lx, &photons, limits, Pauli::Z, hadamards)?;
    let [x_checks, z_checks] = checks;
    Ok(BoundaryCode {
        logical_x: product(lx, Pauli::X, &photons, hadamards)?,
        logical_z: product(lz, Pauli::Z, &photons, hadamards)?,
        hadamard_frame: photons
            .iter()
            .filter(|q| hadamards.contains(q))
            .copied()
            .collect(),
        photons,
        x_checks,
        z_checks,
        x_distance,
        z_distance,
    })
}
fn mask(p: &PauliProduct, photons: &[usize], hadamards: &[usize]) -> [u128; 2] {
    let mut result = [0; 2];
    for &(q, p) in p.terms() {
        let i = photons.iter().position(|&r| r == q).unwrap();
        let p = framed_axis(p, hadamards.contains(&q));
        if p != Pauli::Z {
            result[0] |= 1 << i;
        }
        if p != Pauli::X {
            result[1] |= 1 << i;
        }
    }
    result
}
fn insert(basis: &mut Vec<(usize, u128)>, mut value: u128) -> bool {
    for &(pivot, row) in basis.iter() {
        if value & (1 << pivot) != 0 {
            value ^= row;
        }
    }
    if value == 0 {
        return false;
    }
    let pivot = value.trailing_zeros() as usize;
    let position = basis.partition_point(|(p, _)| *p < pivot);
    basis.insert(position, (pivot, value));
    true
}
fn logical(opposite: &[u128], same: &[u128], n: usize) -> Result<u128> {
    let matrix = opposite
        .iter()
        .map(|v| (0..n).map(|q| v & (1 << q) != 0).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let (_, kernel) = gf2::solve(&matrix, n, &vec![false; matrix.len()]).unwrap();
    let mut basis = vec![];
    for &v in same {
        insert(&mut basis, v);
    }
    for vector in kernel {
        let v = vector
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .fold(0, |a, (q, _)| a | (1 << q));
        let mut trial = basis.clone();
        if insert(&mut trial, v) {
            return Ok(v);
        }
    }
    Err(Error::InvalidValue(
        "boundary code has no logical representative".into(),
    ))
}
fn product(
    bits: u128,
    pauli: Pauli,
    photons: &[usize],
    hadamards: &[usize],
) -> Result<PauliProduct> {
    PauliProduct::new(
        photons
            .iter()
            .enumerate()
            .filter(|(i, _)| bits & (1 << i) != 0)
            .map(|(_, &q)| (q, framed_axis(pauli, hadamards.contains(&q))))
            .collect(),
    )
}
fn distance(
    checks: &[u128],
    logical: u128,
    photons: &[usize],
    limits: BoundaryCertificateLimits,
    pauli: Pauli,
    hadamards: &[usize],
) -> Result<BoundaryDistance> {
    if limits.max_distance_weight == 0 || limits.max_distance_candidates == 0 {
        return Err(Error::InvalidValue(
            "zero boundary distance search budget".into(),
        ));
    }
    let columns = (0..photons.len())
        .map(|q| {
            checks
                .iter()
                .enumerate()
                .filter(|(_, r)| **r & (1 << q) != 0)
                .fold(0u128, |v, (i, _)| v | (1 << i))
        })
        .collect::<Vec<_>>();
    let mut work = 0;
    for weight in 1..=limits.max_distance_weight.min(photons.len()) {
        let mut selected = vec![];
        if search(
            &columns,
            logical,
            0,
            weight,
            0,
            false,
            &mut selected,
            &mut work,
            limits.max_distance_candidates,
        )? {
            return Ok(BoundaryDistance {
                distance: weight,
                operator: product(
                    selected.iter().fold(0, |v, &q| v | (1 << q)),
                    pauli,
                    photons,
                    hadamards,
                )?,
                photons: selected.iter().map(|&q| photons[q]).collect(),
                candidates: work,
            });
        }
    }
    Err(Error::ResourceLimit {
        resource: "boundary distance search weight",
        required: limits
            .max_distance_weight
            .min(photons.len())
            .saturating_add(1),
        limit: limits.max_distance_weight,
    })
}
#[allow(clippy::too_many_arguments)]
fn search(
    columns: &[u128],
    logical: u128,
    start: usize,
    left: usize,
    syndrome: u128,
    flip: bool,
    selected: &mut Vec<usize>,
    work: &mut usize,
    budget: usize,
) -> Result<bool> {
    if left == 0 {
        *work += 1;
        if *work > budget {
            return Err(Error::ResourceLimit {
                resource: "boundary distance candidates",
                required: *work,
                limit: budget,
            });
        }
        return Ok(syndrome == 0 && flip);
    }
    for q in start..=columns.len() - left {
        selected.push(q);
        if search(
            columns,
            logical,
            q + 1,
            left - 1,
            syndrome ^ columns[q],
            flip ^ (logical & (1 << q) != 0),
            selected,
            work,
            budget,
        )? {
            return Ok(true);
        }
        selected.pop();
    }
    Ok(false)
}

fn framed_axis(p: Pauli, hadamard: bool) -> Pauli {
    if hadamard {
        match p {
            Pauli::X => Pauli::Z,
            Pauli::Z => Pauli::X,
            Pauli::Y => Pauli::Y,
        }
    } else {
        p
    }
}
// i X_L Z_L, including the reversed physical ZX multiplication on H-frame sites.
fn y_lift_negative(x: &PauliProduct, z: &PauliProduct) -> bool {
    let mut phase = 1u8;
    for &(q, a) in x.terms() {
        if let Some(&(_, b)) = z.terms().iter().find(|(r, _)| *r == q) {
            phase = (phase
                + match (a, b) {
                    (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::X) => 1,
                    (Pauli::Y, Pauli::X) | (Pauli::Z, Pauli::Y) | (Pauli::X, Pauli::Z) => 3,
                    _ => 0,
                })
                % 4;
        }
    }
    debug_assert!(
        phase == 0 || phase == 2,
        "logical representatives must anticommute"
    );
    phase == 2
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoded_y_lift_tracks_triple_overlap_on_an_independent_state() {
        for n in [1, 3, 5, 7] {
            let x = PauliProduct::new((0..n).map(|q| (q, Pauli::X)).collect()).unwrap();
            let z = PauliProduct::new((0..n).map(|q| (q, Pauli::Z)).collect()).unwrap();
            let y = PauliProduct::new((0..n).map(|q| (q, Pauli::Y)).collect()).unwrap();
            // Encode the +1 eigenstate of logical Y as (|0...0>+i|1...1>)/sqrt2.
            let mut reference = Tableau::zero(n, Default::default()).unwrap();
            reference.h(0).unwrap();
            reference.s(0).unwrap();
            for q in 1..n {
                reference.cnot(0, q).unwrap();
            }
            assert_eq!(
                reference.expectation(&y).unwrap(),
                if y_lift_negative(&x, &z) { -1 } else { 1 }
            );
            // H on every physical site reverses ZX versus XZ and contributes HYH=-Y.
            for q in 0..n {
                reference.h(q).unwrap();
            }
            assert_eq!(
                reference.expectation(&y).unwrap(),
                if y_lift_negative(&z, &x) { -1 } else { 1 }
            );
        }
    }
}
