//! Mixed-radix layouts. Wire zero is the most significant digit.
use crate::error::{Error, Result, checked_product};
use serde::{Deserialize, Serialize};

/// Validated, ordered finite-dimensional quantum subsystems.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<usize>", into = "Vec<usize>")]
pub struct Register {
    dimensions: Vec<usize>,
    size: usize,
}

impl Register {
    /// Create a nonempty mixed-radix register. Every dimension must be at least two.
    pub fn new(dimensions: Vec<usize>) -> Result<Self> {
        if dimensions.is_empty() || dimensions.iter().any(|&d| d < 2) {
            return Err(Error::InvalidDimensions(format!("{dimensions:?}")));
        }
        let size = checked_product(dimensions.iter().copied())?;
        Ok(Self { dimensions, size })
    }

    /// Construct `count` qubits.
    pub fn qubits(count: usize) -> Result<Self> {
        // Reject overflowing requests before allocating their dimension list.
        if count >= usize::BITS as usize {
            return Err(Error::Overflow);
        }
        Self::new(vec![2; count])
    }

    /// Local dimensions in wire order.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    /// Number of subsystems.
    pub fn wires(&self) -> usize {
        self.dimensions.len()
    }
    /// Total Hilbert-space dimension.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Encode a big-endian mixed-radix basis label.
    pub fn encode(&self, digits: &[usize]) -> Result<usize> {
        if digits.len() != self.wires() {
            return Err(Error::Basis);
        }
        let mut index = 0;
        for (&x, &d) in digits.iter().zip(&self.dimensions) {
            if x >= d {
                return Err(Error::Basis);
            }
            index = index * d + x;
        }
        Ok(index)
    }

    /// Decode a flat index into a digit for each wire.
    pub fn decode(&self, mut index: usize) -> Result<Vec<usize>> {
        if index >= self.size {
            return Err(Error::Basis);
        }
        let mut digits = vec![0; self.wires()];
        for (x, &d) in digits.iter_mut().zip(&self.dimensions).rev() {
            *x = index % d;
            index /= d;
        }
        Ok(digits)
    }

    /// Validate ordered target wires and return their combined local dimension.
    pub fn local_size(&self, wires: &[usize]) -> Result<usize> {
        if wires.is_empty() {
            return Err(Error::Wires("empty target list".into()));
        }
        for (i, &wire) in wires.iter().enumerate() {
            if wire >= self.wires() || wires[..i].contains(&wire) {
                return Err(Error::Wires(format!("{wires:?}")));
            }
        }
        checked_product(wires.iter().map(|&w| self.dimensions[w]))
    }

    pub(crate) fn permutation(&self, wires: &[usize]) -> Result<(Vec<i64>, Vec<i64>)> {
        self.local_size(wires)?;
        if self.size > i32::MAX as usize {
            return Err(Error::Unsupported(
                "dense gather indices exceed signed 32-bit range".into(),
            ));
        }
        let order: Vec<_> = (0..self.wires())
            .filter(|w| !wires.contains(w))
            .chain(wires.iter().copied())
            .collect();
        let mut forward = vec![0; self.size];
        let mut inverse = vec![0; self.size];
        for (old, inv) in inverse.iter_mut().enumerate() {
            let digits = self.decode(old)?;
            let packed = order
                .iter()
                .fold(0, |index, &w| index * self.dimensions[w] + digits[w]);
            forward[packed] = old as i64;
            *inv = packed as i64;
        }
        Ok((forward, inverse))
    }
}

impl TryFrom<Vec<usize>> for Register {
    type Error = Error;
    fn try_from(value: Vec<usize>) -> Result<Self> {
        Self::new(value)
    }
}
impl From<Register> for Vec<usize> {
    fn from(value: Register) -> Self {
        value.dimensions
    }
}
