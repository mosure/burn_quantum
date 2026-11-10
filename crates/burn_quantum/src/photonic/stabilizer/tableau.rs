use super::PauliProduct;
use super::pauli::Row;
use crate::{Error, Result};

/// Explicit limits for the bit-packed Clifford representation.
#[derive(Clone, Copy, Debug)]
pub struct TableauLimits {
    /// Maximum qubits, including consumed sites retained for stable indexing.
    pub max_qubits: usize,
    /// Maximum u64 words in both X/Z planes of all stabilizer/destabilizer rows.
    pub max_words: usize,
}
impl Default for TableauLimits {
    fn default() -> Self {
        Self {
            max_qubits: 16384,
            max_words: 1 << 24,
        }
    }
}
/// A Pauli measurement result; random outcomes each have probability one half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Measurement {
    /// True denotes eigenvalue -1.
    pub negative: bool,
    /// False means the result was already fixed by the state.
    pub random: bool,
}
/// Pure stabilizer state with 2n bit-packed rows and exact Pauli phases.
///
/// The first n rows are destabilizers and the last n rows are stabilizers.
/// Storage is O(n²), Clifford gates O(n), and a general Pauli measurement O(n²)
/// in bit operations. This is a host engine, independent of Burn tensor devices.
#[derive(Clone, Debug)]
pub struct Tableau {
    qubits: usize,
    rows: Vec<Row>,
}
impl Tableau {
    /// Allocate |0...0>, checking the complete tableau word count first.
    pub fn zero(qubits: usize, limits: TableauLimits) -> Result<Self> {
        if qubits == 0 {
            return Err(Error::InvalidDimensions("empty tableau".into()));
        }
        let words = qubits.div_ceil(64);
        let required = qubits
            .checked_mul(4)
            .and_then(|n| n.checked_mul(words))
            .ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("tableau qubits", qubits, limits.max_qubits),
            ("tableau words", required, limits.max_words),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        let mut rows = vec![Row::zero(words); qubits * 2];
        for q in 0..qubits {
            Row::toggle(&mut rows[q].x, q);
            Row::toggle(&mut rows[qubits + q].z, q);
        }
        Ok(Self { qubits, rows })
    }
    /// Number of indexed qubits; indices never shift after measurement.
    pub fn qubits(&self) -> usize {
        self.qubits
    }
    /// Heap bytes reserved by this tableau (rows and packed planes, excluding allocator overhead).
    pub fn allocated_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<Row>()
            + self
                .rows
                .iter()
                .map(|r| (r.x.capacity() + r.z.capacity()) * 8)
                .sum::<usize>()
    }
    fn check(&self, q: usize) -> Result<()> {
        if q >= self.qubits {
            Err(Error::Wires("tableau site out of range".into()))
        } else {
            Ok(())
        }
    }
    /// Apply Hadamard.
    pub fn h(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        for row in &mut self.rows {
            let x = Row::bit(&row.x, q);
            let z = Row::bit(&row.z, q);
            if x && z {
                row.phase ^= 2;
            }
            if x != z {
                Row::toggle(&mut row.x, q);
                Row::toggle(&mut row.z, q);
            }
        }
        Ok(())
    }
    /// Apply S = diag(1,i).
    pub fn s(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        for row in &mut self.rows {
            if Row::bit(&row.x, q) {
                row.phase = (row.phase + 1) & 3;
                Row::toggle(&mut row.z, q);
            }
        }
        Ok(())
    }
    /// Apply Pauli X.
    pub fn x(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        for row in &mut self.rows {
            if Row::bit(&row.z, q) {
                row.phase ^= 2;
            }
        }
        Ok(())
    }
    /// Apply Pauli Z.
    pub fn z(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        for row in &mut self.rows {
            if Row::bit(&row.x, q) {
                row.phase ^= 2;
            }
        }
        Ok(())
    }
    /// Apply controlled X, rejecting repeated or out-of-range operands before mutation.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<()> {
        self.check(control)?;
        self.check(target)?;
        if control == target {
            return Err(Error::Wires("repeated CNOT site".into()));
        }
        for row in &mut self.rows {
            if Row::bit(&row.x, control) {
                Row::toggle(&mut row.x, target);
            }
            if Row::bit(&row.z, target) {
                Row::toggle(&mut row.z, control);
            }
        }
        Ok(())
    }
    /// Apply controlled Z.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<()> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err(Error::Wires("repeated CZ site".into()));
        }
        self.h(b)?;
        self.cnot(a, b)?;
        self.h(b)
    }
    fn fixed_sign(&self, observable: &Row) -> bool {
        let mut product = Row::zero(self.qubits.div_ceil(64));
        for q in 0..self.qubits {
            if self.rows[q].anticommutes(observable) {
                product.multiply(&self.rows[self.qubits + q]);
            }
        }
        debug_assert_eq!(product.x, observable.x);
        debug_assert_eq!(product.z, observable.z);
        let phase = (product.phase + 4 - observable.phase) & 3;
        debug_assert!(phase == 0 || phase == 2);
        phase == 2
    }
    /// Exact expectation of a Hermitian Pauli product: -1, 0 or +1.
    pub fn expectation(&self, product: &PauliProduct) -> Result<i8> {
        let row = Row::from_product(product, self.qubits)?;
        if self.rows[self.qubits..]
            .iter()
            .any(|r| r.anticommutes(&row))
        {
            Ok(0)
        } else {
            Ok(if self.fixed_sign(&row) { -1 } else { 1 })
        }
    }
    /// Measure a Pauli product; consume `random_bit` only for a genuinely random result.
    ///
    /// Supplying each bit with probability 1/2 gives the physical Born distribution.
    /// Exposing the bit makes deterministic replay and exhaustive branch tests possible.
    pub fn measure(&mut self, product: &PauliProduct, random_bit: bool) -> Result<Measurement> {
        Ok(self.measure_internal(product, random_bit, false)?.0)
    }
    pub(super) fn reference_measure(
        &mut self,
        product: &PauliProduct,
    ) -> Result<(Measurement, Option<PauliProduct>)> {
        self.measure_internal(product, false, true)
    }
    pub(crate) fn commutations(&self, product: &PauliProduct) -> Result<Vec<bool>> {
        let row = Row::from_product(product, self.qubits)?;
        Ok(self.rows[self.qubits..]
            .iter()
            .map(|s| s.anticommutes(&row))
            .collect())
    }
    fn measure_internal(
        &mut self,
        product: &PauliProduct,
        random_bit: bool,
        capture: bool,
    ) -> Result<(Measurement, Option<PauliProduct>)> {
        let mut observable = Row::from_product(product, self.qubits)?;
        let Some(pivot) =
            (self.qubits..self.rows.len()).find(|&p| self.rows[p].anticommutes(&observable))
        else {
            return Ok((
                Measurement {
                    negative: self.fixed_sign(&observable),
                    random: false,
                },
                None,
            ));
        };
        let previous = self.rows[pivot].clone();
        let kickback = capture.then(|| previous.product(self.qubits));
        for i in 0..self.rows.len() {
            if i != pivot && i != pivot - self.qubits && self.rows[i].anticommutes(&observable) {
                self.rows[i].multiply(&previous);
            }
        }
        self.rows[pivot - self.qubits] = previous;
        if random_bit {
            observable.phase ^= 2;
        }
        self.rows[pivot] = observable;
        Ok((
            Measurement {
                negative: random_bit,
                random: true,
            },
            kickback,
        ))
    }
}
