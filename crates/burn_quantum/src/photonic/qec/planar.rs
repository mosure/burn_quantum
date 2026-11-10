//! Open-boundary planar CSS circuit references, with explicit bounded decoding.
//!
//! Physical CNOTs and noiseless syndrome circuits are declared operations here;
//! this module does not assert a deterministic optical CNOT or FBQC surgery layout.
use super::{ErasureDecode, ParityCode};
use crate::photonic::stabilizer::{Pauli, PauliProduct, Tableau};
use crate::{Error, Result};

/// Auditable primitive counts for a reference operation. Reused syndrome ancillas
/// count once per preparation, not merely once per occupied tableau site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanarCost {
    /// Physical two-qubit controlled-X operations.
    pub cnots: usize,
    /// Physical Hadamards.
    pub hadamards: usize,
    /// Physical X or Z operations (Y may be implemented as both).
    pub paulis: usize,
    /// Physical S operations.
    pub phases: usize,
    /// Destructive single-site measurements.
    pub measurements: usize,
    /// Fresh |0> preparations, including replacements and reused ancillas.
    pub preparations: usize,
    /// Wire swaps, implemented exactly as three CNOTs and counted also in `cnots`.
    pub swaps: usize,
}
/// Explicit complexity limits for exact minimum-weight decoding. Erased sites
/// have zero cost, and all equally minimal explanations must agree logically.
#[derive(Clone, Copy, Debug)]
pub struct PlanarDecodeBudget {
    /// Maximum number of non-erased errors to enumerate per binary channel.
    pub max_weight: usize,
    /// Maximum candidate subsets per binary channel, including the empty subset.
    pub max_candidates: usize,
}
/// Result of bounded two-channel decoding, without hidden postselection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanarRecovery {
    /// Pauli correction; simultaneous X and Z denotes Y up to irrelevant phase.
    Correction {
        /// Physical X corrections.
        x: Vec<bool>,
        /// Physical Z corrections.
        z: Vec<bool>,
    },
    /// Erasures or equally likely minimal faults support distinct logical actions.
    Ambiguous,
    /// A search limit was reached before establishing a correction.
    BudgetExceeded,
}
/// Destructive encoded Z readout, preserving decoder ambiguity and work limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarReadout {
    /// True denotes the negative logical eigenvalue after classical correction.
    Known(bool),
    /// Unavailable or equally minimal records do not determine the logical bit.
    Ambiguous,
    /// The bounded classical search did not establish the logical bit.
    BudgetExceeded,
}
/// One open-boundary [[d²+(d−1)²,1,d]] planar surface-code patch.
///
/// Data lie on d² vertical and (d−1)² horizontal edges. Stars touch the smooth
/// left/right boundaries; plaquettes touch the rough top/bottom boundaries.
/// The independent CSS checks have weights three or four. Construction and
/// encoding use host GF(2) algebra, with no postselected state preparation.
#[derive(Clone, Debug)]
pub struct PlanarCode {
    distance: usize,
    n: usize,
    x: Vec<Vec<bool>>,
    z: Vec<Vec<bool>>,
    lx: Vec<bool>,
    lz: Vec<bool>,
    plus: Vec<usize>,
    input: usize,
    encoder: Vec<(usize, usize)>,
    rotation: Vec<usize>,
}
impl PlanarCode {
    /// Construct an odd-distance patch within a caller-supplied data-qubit budget.
    pub fn new(distance: usize, max_qubits: usize) -> Result<Self> {
        if !(3..=31).contains(&distance) || distance.is_multiple_of(2) {
            return Err(Error::InvalidDimensions(
                "planar distance must be odd in 3..=31".into(),
            ));
        }
        let d = distance;
        let n = d * d + (d - 1) * (d - 1);
        if n > max_qubits {
            return Err(Error::ResourceLimit {
                resource: "planar data qubits",
                required: n,
                limit: max_qubits,
            });
        }
        let v = |x: usize, y: usize| x * d + y;
        let h = |x: usize, y: usize| d * d + x * (d - 1) + y - 1;
        let mut x = Vec::new();
        for a in 0..d {
            for b in 1..d {
                let mut r = vec![false; n];
                r[v(a, b - 1)] = true;
                r[v(a, b)] = true;
                if a > 0 {
                    r[h(a - 1, b)] = true;
                }
                if a < d - 1 {
                    r[h(a, b)] = true;
                }
                x.push(r);
            }
        }
        let mut z = Vec::new();
        for a in 0..d - 1 {
            for b in 0..d {
                let mut r = vec![false; n];
                r[v(a, b)] = true;
                r[v(a + 1, b)] = true;
                if b > 0 {
                    r[h(a, b)] = true;
                }
                if b < d - 1 {
                    r[h(a, b + 1)] = true;
                }
                z.push(r);
            }
        }
        let mut lx = vec![false; n];
        let mut lz = vec![false; n];
        for a in 0..d {
            lx[v(a, 0)] = true;
            lz[v(0, a)] = true;
        }
        // Row echelonization never adds the logical row to a stabilizer row.
        let mut basis = x.clone();
        basis.push(lx.clone());
        let mut pivots = Vec::new();
        for i in 0..basis.len() {
            let (earlier, current) = basis.split_at_mut(i);
            let row = &mut current[0];
            for (prior, &p) in earlier.iter().zip(&pivots) {
                if row[p] {
                    for (value, &b) in row.iter_mut().zip(prior) {
                        *value ^= b;
                    }
                }
            }
            let p = row
                .iter()
                .position(|&b| b)
                .ok_or_else(|| Error::InvalidValue("dependent planar encoder rows".into()))?;
            pivots.push(p);
        }
        let input = *pivots.last().unwrap();
        let plus = pivots[..pivots.len() - 1].to_vec();
        let mut elimination = Vec::new();
        for i in (0..basis.len()).rev() {
            let p = pivots[i];
            for q in 0..n {
                if q != p && basis[i][q] {
                    elimination.push((p, q));
                    for r in &mut basis {
                        r[q] ^= r[p];
                    }
                }
            }
        }
        elimination.reverse();
        let mut rotation = vec![0; n];
        for a in 0..d {
            for b in 0..d {
                rotation[v(a, b)] = v(d - 1 - b, a);
            }
        }
        for a in 0..d - 1 {
            for b in 1..d {
                rotation[h(a, b)] = h(d - 1 - b, a + 1);
            }
        }
        Ok(Self {
            distance: d,
            n,
            x,
            z,
            lx,
            lz,
            plus,
            input,
            encoder: elimination,
            rotation,
        })
    }
    /// Code distance for both Pauli channels.
    pub fn distance(&self) -> usize {
        self.distance
    }
    /// Number of physical data sites, excluding check ancillas and references.
    pub fn qubits(&self) -> usize {
        self.n
    }
    /// Site that contains the arbitrary input before `encode`.
    pub fn input_site(&self) -> usize {
        self.input
    }
    /// Reversible physical encoder CNOTs, in execution order after preparation H gates.
    pub fn encoding_cnots(&self) -> &[(usize, usize)] {
        &self.encoder
    }
    /// Sites initialized with H before the encoder CNOTs.
    pub fn encoding_plus_sites(&self) -> &[usize] {
        &self.plus
    }
    fn range(&self, state: &Tableau, offset: usize) -> Result<()> {
        if offset
            .checked_add(self.n)
            .is_none_or(|end| end > state.qubits())
        {
            Err(Error::Wires("planar patch outside tableau".into()))
        } else {
            Ok(())
        }
    }
    fn product(row: &[bool], p: Pauli, offset: usize) -> Result<PauliProduct> {
        offset.checked_add(row.len()).ok_or(Error::Overflow)?;
        PauliProduct::new(
            row.iter()
                .enumerate()
                .filter_map(|(q, &v)| v.then_some((q + offset, p)))
                .collect(),
        )
    }
    /// Positive X stabilizer generators at the requested data offset.
    pub fn x_checks(&self, offset: usize) -> Result<Vec<PauliProduct>> {
        self.x
            .iter()
            .map(|r| Self::product(r, Pauli::X, offset))
            .collect()
    }
    /// Positive Z stabilizer generators at the requested data offset.
    pub fn z_checks(&self, offset: usize) -> Result<Vec<PauliProduct>> {
        self.z
            .iter()
            .map(|r| Self::product(r, Pauli::Z, offset))
            .collect()
    }
    /// Canonical positive logical X across the lower boundary.
    pub fn logical_x(&self, offset: usize) -> Result<PauliProduct> {
        Self::product(&self.lx, Pauli::X, offset)
    }
    /// Canonical positive logical Z along the left boundary.
    pub fn logical_z(&self, offset: usize) -> Result<PauliProduct> {
        Self::product(&self.lz, Pauli::Z, offset)
    }
    /// Code for X errors: Z checks and flips of the logical Z observable.
    pub fn x_error_code(&self) -> Result<ParityCode> {
        ParityCode::new(self.n, self.z.clone(), vec![self.lz.clone()])
    }
    /// Code for Z errors: X checks and flips of the logical X observable.
    pub fn z_error_code(&self) -> Result<ParityCode> {
        ParityCode::new(self.n, self.x.clone(), vec![self.lx.clone()])
    }
    /// Encode any state, including an unknown input entangled with external sites.
    /// All patch sites except `input_site()` must initially be independent |0>.
    pub fn encode(&self, state: &mut Tableau, offset: usize) -> Result<PlanarCost> {
        self.range(state, offset)?;
        for q in 0..self.n {
            if q != self.input
                && state.expectation(&PauliProduct::new(vec![(offset + q, Pauli::Z)])?)? != 1
            {
                return Err(Error::InvalidValue("encoder ancilla is not |0>".into()));
            }
        }
        for &q in &self.plus {
            state.h(offset + q)?;
        }
        for &(a, b) in &self.encoder {
            state.cnot(offset + a, offset + b)?;
        }
        Ok(PlanarCost {
            cnots: self.encoder.len(),
            hadamards: self.plus.len(),
            ..Default::default()
        })
    }
    /// Logical H using transversal H and a 90-degree physical patch permutation.
    /// Swaps are exact three-CNOT reference operations, not free optical routing.
    pub fn logical_h(&self, state: &mut Tableau, offset: usize) -> Result<PlanarCost> {
        self.range(state, offset)?;
        for q in 0..self.n {
            state.h(offset + q)?;
        }
        let mut current: Vec<_> = (0..self.n).collect();
        let mut swaps = 0;
        for dest in 0..self.n {
            let source = self.rotation.iter().position(|&p| p == dest).unwrap();
            let pos = current.iter().position(|&q| q == source).unwrap();
            if pos != dest {
                state.cnot(offset + dest, offset + pos)?;
                state.cnot(offset + pos, offset + dest)?;
                state.cnot(offset + dest, offset + pos)?;
                current.swap(dest, pos);
                swaps += 1;
            }
        }
        Ok(PlanarCost {
            cnots: 3 * swaps,
            hadamards: self.n,
            swaps,
            ..Default::default()
        })
    }
    /// Transversal CNOT between disjoint identical patches. Cross-patch connectivity
    /// is an explicit reference assumption; no nearest-neighbor surgery is implied.
    pub fn logical_cnot(
        &self,
        state: &mut Tableau,
        control: usize,
        target: usize,
    ) -> Result<PlanarCost> {
        self.range(state, control)?;
        self.range(state, target)?;
        if control.abs_diff(target) < self.n {
            return Err(Error::Wires("overlapping planar patches".into()));
        }
        for q in 0..self.n {
            state.cnot(control + q, target + q)?;
        }
        Ok(PlanarCost {
            cnots: self.n,
            ..Default::default()
        })
    }
    /// Logical S by an encoded |Y> ancilla, transversal CNOT, destructive physical
    /// Z readout and a conditional logical Z. `ancilla` must be a fresh disjoint
    /// |0> patch. Both measurement branches are accepted. Noise on this protocol
    /// can be inserted through the public primitive encoder/gate APIs.
    pub fn logical_s(
        &self,
        state: &mut Tableau,
        data: usize,
        ancilla: usize,
        mut bit: impl FnMut() -> bool,
    ) -> Result<(bool, PlanarCost)> {
        self.range(state, data)?;
        self.range(state, ancilla)?;
        if data.abs_diff(ancilla) < self.n {
            return Err(Error::Wires("overlapping planar patches".into()));
        }
        for q in 0..self.n {
            if state.expectation(&PauliProduct::new(vec![(ancilla + q, Pauli::Z)])?)? != 1 {
                return Err(Error::InvalidValue("phase ancilla is not |0>".into()));
            }
        }
        state.h(ancilla + self.input)?;
        state.s(ancilla + self.input)?;
        let mut cost = self.encode(state, ancilla)?;
        self.logical_cnot(state, data, ancilla)?;
        let mut outcome = false;
        for q in 0..self.n {
            let m = state
                .measure(&PauliProduct::new(vec![(ancilla + q, Pauli::Z)])?, bit())?
                .negative;
            outcome ^= m && self.lz[q];
        }
        if outcome {
            for (q, &on) in self.lz.iter().enumerate() {
                if on {
                    state.z(data + q)?;
                }
            }
        }
        cost.cnots += self.n;
        cost.hadamards += 1;
        cost.phases += 1;
        cost.paulis = if outcome { self.distance } else { 0 };
        cost.measurements = self.n;
        cost.preparations = self.n;
        Ok((outcome, cost))
    }
    /// Measure every check with actual one-ancilla CNOT circuits, resetting and
    /// reusing the supplied |0> site. Returns (X-check signs, Z-check signs, cost).
    /// These circuits have ideal gates/readout; no circuit-noise threshold is claimed.
    pub fn measure_syndrome(
        &self,
        state: &mut Tableau,
        offset: usize,
        ancilla: usize,
        mut bit: impl FnMut() -> bool,
    ) -> Result<(Vec<bool>, Vec<bool>, PlanarCost)> {
        self.range(state, offset)?;
        if ancilla >= state.qubits() || (offset..offset + self.n).contains(&ancilla) {
            return Err(Error::Wires("invalid syndrome ancilla".into()));
        }
        let az = PauliProduct::new(vec![(ancilla, Pauli::Z)])?;
        if state.expectation(&az)? != 1 {
            return Err(Error::InvalidValue("syndrome ancilla is not |0>".into()));
        }
        let mut cost = PlanarCost::default();
        let mut signs = [Vec::new(), Vec::new()];
        for (channel, rows) in [&self.x, &self.z].into_iter().enumerate() {
            for row in rows {
                if channel == 0 {
                    state.h(ancilla)?;
                    cost.hadamards += 2;
                }
                for (q, &on) in row.iter().enumerate() {
                    if on {
                        if channel == 0 {
                            state.cnot(ancilla, offset + q)?;
                        } else {
                            state.cnot(offset + q, ancilla)?;
                        }
                        cost.cnots += 1;
                    }
                }
                if channel == 0 {
                    state.h(ancilla)?;
                }
                let m = state.measure(&az, bit())?.negative;
                if m {
                    state.x(ancilla)?;
                }
                signs[channel].push(m);
                cost.measurements += 1;
                cost.preparations += 1;
            }
        }
        let [x, z] = signs;
        Ok((x, z, cost))
    }
    /// Trace out each known erased site through a hidden Z environment record and
    /// replace it with |0>. Random environment outcomes must never enter the decoder.
    /// The pure trajectories average to the physical trace-and-replacement channel.
    pub fn replace_erasures(
        &self,
        state: &mut Tableau,
        offset: usize,
        erased: &[bool],
        mut hidden_bit: impl FnMut() -> bool,
    ) -> Result<PlanarCost> {
        self.range(state, offset)?;
        if erased.len() != self.n {
            return Err(Error::Shape("planar erasure width".into()));
        }
        let mut cost = PlanarCost::default();
        for (q, &lost) in erased.iter().enumerate() {
            if lost {
                if state
                    .measure(
                        &PauliProduct::new(vec![(offset + q, Pauli::Z)])?,
                        hidden_bit(),
                    )?
                    .negative
                {
                    state.x(offset + q)?;
                }
                cost.preparations += 1;
            }
        }
        Ok(cost)
    }
    /// Destructively read out encoded Z with a loss mask and bounded classical
    /// correction of physical readout bits. Lost sites are traced through hidden
    /// environment outcomes, never substituted as observed zero outcomes. Only
    /// surviving sites incur physical measurement cost. Ideal Z stabilizers and
    /// a declared X/Y fault model are assumed; this is not noisy-check decoding.
    pub fn readout_z(
        &self,
        state: &mut Tableau,
        offset: usize,
        erased: &[bool],
        budget: PlanarDecodeBudget,
        mut bit: impl FnMut() -> bool,
    ) -> Result<(PlanarReadout, PlanarCost)> {
        self.range(state, offset)?;
        if erased.len() != self.n {
            return Err(Error::Shape("planar readout erasure width".into()));
        }
        if budget.max_weight > self.n {
            return Err(Error::InvalidDimensions(
                "decoder weight exceeds patch width".into(),
            ));
        }
        let mut records = vec![false; self.n];
        let mut cost = PlanarCost::default();
        for q in 0..self.n {
            let result = state
                .measure(&PauliProduct::new(vec![(offset + q, Pauli::Z)])?, bit())?
                .negative;
            if !erased[q] {
                records[q] = result;
                cost.measurements += 1;
            }
        }
        let code = self.x_error_code()?;
        let status = match decode_binary(&code, &code.syndrome(&records)?, erased, budget)? {
            BinaryRecovery::Correction(correction) => {
                for (value, flip) in records.iter_mut().zip(correction) {
                    *value ^= flip;
                }
                PlanarReadout::Known(code.logical_flips(&records)?[0])
            }
            BinaryRecovery::Ambiguous => PlanarReadout::Ambiguous,
            BinaryRecovery::BudgetExceeded => PlanarReadout::BudgetExceeded,
        };
        Ok((status, cost))
    }
    /// Decode both observed check channels, retaining loss and budget failures.
    pub fn decode(
        &self,
        x_signs: &[bool],
        z_signs: &[bool],
        erased: &[bool],
        budget: PlanarDecodeBudget,
    ) -> Result<PlanarRecovery> {
        let x = decode_binary(&self.x_error_code()?, z_signs, erased, budget)?;
        let z = decode_binary(&self.z_error_code()?, x_signs, erased, budget)?;
        match (x, z) {
            (BinaryRecovery::Correction(x), BinaryRecovery::Correction(z)) => {
                Ok(PlanarRecovery::Correction { x, z })
            }
            (BinaryRecovery::Ambiguous, _) | (_, BinaryRecovery::Ambiguous) => {
                Ok(PlanarRecovery::Ambiguous)
            }
            _ => Ok(PlanarRecovery::BudgetExceeded),
        }
    }
    /// Apply a returned correction, checking all dimensions before mutation.
    pub fn correct(
        &self,
        state: &mut Tableau,
        offset: usize,
        recovery: &PlanarRecovery,
    ) -> Result<bool> {
        self.range(state, offset)?;
        let PlanarRecovery::Correction { x, z } = recovery else {
            return Ok(false);
        };
        if x.len() != self.n || z.len() != self.n {
            return Err(Error::Shape("planar correction width".into()));
        }
        for q in 0..self.n {
            if x[q] {
                state.x(offset + q)?;
            }
            if z[q] {
                state.z(offset + q)?;
            }
        }
        Ok(true)
    }
}

enum BinaryRecovery {
    Correction(Vec<bool>),
    Ambiguous,
    BudgetExceeded,
}
fn decode_binary(
    code: &ParityCode,
    syndrome: &[bool],
    erased: &[bool],
    budget: PlanarDecodeBudget,
) -> Result<BinaryRecovery> {
    if syndrome.len() != code.checks().len() || erased.len() != code.variables() {
        return Err(Error::Shape("planar decoder dimensions".into()));
    }
    if budget.max_weight > code.variables() {
        return Err(Error::InvalidDimensions(
            "decoder weight exceeds patch width".into(),
        ));
    }
    if matches!(
        code.decode_erasures(&vec![false; syndrome.len()], erased)?,
        ErasureDecode::Ambiguous { .. }
    ) {
        return Ok(BinaryRecovery::Ambiguous);
    }
    let free: Vec<_> = erased
        .iter()
        .enumerate()
        .filter_map(|(q, &e)| (!e).then_some(q))
        .collect();
    let any_erased = erased.iter().any(|&e| e);
    let mut tested = 0;
    for weight in 0..=budget.max_weight.min(free.len()) {
        let mut indices: Vec<_> = (0..weight).collect();
        let mut found: Option<(Vec<bool>, Vec<bool>)> = None;
        loop {
            if tested >= budget.max_candidates {
                return Ok(BinaryRecovery::BudgetExceeded);
            }
            tested += 1;
            let mut candidate = vec![false; code.variables()];
            for &i in &indices {
                candidate[free[i]] = true;
            }
            let predicted = code.syndrome(&candidate)?;
            let remainder: Vec<_> = syndrome
                .iter()
                .zip(&predicted)
                .map(|(&a, &b)| a ^ b)
                .collect();
            let completion = if any_erased {
                match code.decode_erasures(&remainder, erased)? {
                    ErasureDecode::Decoded { correction, .. } => Some(correction),
                    ErasureDecode::Inconsistent => None,
                    ErasureDecode::Ambiguous { .. } => return Ok(BinaryRecovery::Ambiguous),
                }
            } else if remainder.iter().all(|&b| !b) {
                Some(vec![false; code.variables()])
            } else {
                None
            };
            if let Some(completion) = completion {
                for (v, c) in candidate.iter_mut().zip(completion) {
                    *v ^= c;
                }
                let logical = code.logical_flips(&candidate)?;
                if let Some((_, prior)) = &found {
                    if *prior != logical {
                        return Ok(BinaryRecovery::Ambiguous);
                    }
                } else {
                    found = Some((candidate, logical));
                }
            }
            let Some(i) = (0..weight)
                .rev()
                .find(|&i| indices[i] < free.len() - weight + i)
            else {
                break;
            };
            indices[i] += 1;
            for j in i + 1..weight {
                indices[j] = indices[j - 1] + 1;
            }
        }
        if let Some((correction, _)) = found {
            return Ok(BinaryRecovery::Correction(correction));
        }
    }
    Ok(BinaryRecovery::BudgetExceeded)
}
