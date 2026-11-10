use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// A nonidentity single-qubit Pauli observable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pauli {
    /// X.
    X,
    /// Y, with Y = i X Z.
    Y,
    /// Z.
    Z,
}
/// A positive-sign Hermitian Pauli product on distinct indexed qubits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<(usize, Pauli)>", into = "Vec<(usize, Pauli)>")]
pub struct PauliProduct(Vec<(usize, Pauli)>);
impl PauliProduct {
    /// Validate and sort sparse terms. An empty product is identity.
    pub fn new(mut terms: Vec<(usize, Pauli)>) -> Result<Self> {
        terms.sort_unstable_by_key(|t| t.0);
        if terms.windows(2).any(|t| t[0].0 == t[1].0) {
            return Err(Error::Wires("repeated Pauli site".into()));
        }
        Ok(Self(terms))
    }
    /// Canonically ordered sites and factors.
    pub fn terms(&self) -> &[(usize, Pauli)] {
        &self.0
    }
    /// Whether two Hermitian products anticommute.
    pub fn anticommutes(&self, other: &Self) -> bool {
        self.0
            .iter()
            .filter(|(q, p)| other.0.iter().any(|(r, s)| q == r && p != s))
            .count()
            % 2
            == 1
    }
    /// Multiply commuting Hermitian products, returning the minus sign and positive product.
    pub fn multiply_commuting(&self, other: &Self) -> Result<(bool, Self)> {
        if self.anticommutes(other) {
            return Err(Error::InvalidValue("noncommuting Pauli product".into()));
        }
        let mut terms = std::collections::BTreeMap::new();
        for &(q, p) in &self.0 {
            terms.insert(q, p);
        }
        let mut phase = 0u8;
        for &(q, p) in &other.0 {
            if let Some(old) = terms.remove(&q) {
                if old != p {
                    let (out, sign) = match (old, p) {
                        (Pauli::X, Pauli::Y) => (Pauli::Z, 1),
                        (Pauli::Y, Pauli::Z) => (Pauli::X, 1),
                        (Pauli::Z, Pauli::X) => (Pauli::Y, 1),
                        (Pauli::Y, Pauli::X) => (Pauli::Z, 3),
                        (Pauli::Z, Pauli::Y) => (Pauli::X, 3),
                        (Pauli::X, Pauli::Z) => (Pauli::Y, 3),
                        _ => unreachable!(),
                    };
                    terms.insert(q, out);
                    phase = (phase + sign) & 3;
                }
            } else {
                terms.insert(q, p);
            }
        }
        Ok((phase == 2, Self(terms.into_iter().collect())))
    }
}
impl TryFrom<Vec<(usize, Pauli)>> for PauliProduct {
    type Error = Error;
    fn try_from(value: Vec<(usize, Pauli)>) -> Result<Self> {
        Self::new(value)
    }
}
impl From<PauliProduct> for Vec<(usize, Pauli)> {
    fn from(value: PauliProduct) -> Self {
        value.0
    }
}

#[derive(Clone, Debug)]
pub(super) struct Row {
    pub x: Vec<u64>,
    pub z: Vec<u64>,
    pub phase: u8,
}
impl Row {
    pub fn product(&self, qubits: usize) -> PauliProduct {
        PauliProduct(
            (0..qubits)
                .filter_map(|q| match (Self::bit(&self.x, q), Self::bit(&self.z, q)) {
                    (true, false) => Some((q, Pauli::X)),
                    (false, true) => Some((q, Pauli::Z)),
                    (true, true) => Some((q, Pauli::Y)),
                    _ => None,
                })
                .collect(),
        )
    }
    pub fn zero(words: usize) -> Self {
        Self {
            x: vec![0; words],
            z: vec![0; words],
            phase: 0,
        }
    }
    pub fn bit(v: &[u64], q: usize) -> bool {
        (v[q / 64] >> (q % 64)) & 1 != 0
    }
    pub fn toggle(v: &mut [u64], q: usize) {
        v[q / 64] ^= 1 << (q % 64);
    }
    pub fn anticommutes(&self, rhs: &Self) -> bool {
        self.x
            .iter()
            .zip(&self.z)
            .zip(rhs.x.iter().zip(&rhs.z))
            .fold(0, |parity, ((x, z), (rx, rz))| {
                parity ^ ((x & rz).count_ones() ^ (z & rx).count_ones())
            })
            & 1
            != 0
    }
    pub fn multiply(&mut self, rhs: &Self) {
        let parity = self
            .z
            .iter()
            .zip(&rhs.x)
            .fold(0, |p, (z, x)| p ^ (z & x).count_ones());
        self.phase = (self.phase + rhs.phase + 2 * (parity as u8 & 1)) & 3;
        for i in 0..self.x.len() {
            self.x[i] ^= rhs.x[i];
            self.z[i] ^= rhs.z[i];
        }
    }
    pub fn from_product(product: &PauliProduct, qubits: usize) -> Result<Self> {
        if product.terms().iter().any(|(q, _)| *q >= qubits) {
            return Err(Error::Wires("Pauli site out of range".into()));
        }
        let mut row = Self::zero(qubits.div_ceil(64));
        for &(q, p) in product.terms() {
            if p != Pauli::Z {
                Self::toggle(&mut row.x, q);
            }
            if p != Pauli::X {
                Self::toggle(&mut row.z, q);
            }
            if p == Pauli::Y {
                row.phase = (row.phase + 1) & 3;
            }
        }
        Ok(row)
    }
}
