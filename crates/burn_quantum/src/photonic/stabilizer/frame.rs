use super::{Pauli, PauliProduct};
use crate::{Error, Result};

/// Classical Pauli correction frame; global phase is deliberately ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliFrame {
    x: Vec<bool>,
    z: Vec<bool>,
}
impl PauliFrame {
    /// Identity frame with an explicit site budget.
    pub fn new(qubits: usize, max_qubits: usize) -> Result<Self> {
        if qubits > max_qubits {
            return Err(Error::ResourceLimit {
                resource: "frame qubits",
                required: qubits,
                limit: max_qubits,
            });
        }
        Ok(Self {
            x: vec![false; qubits],
            z: vec![false; qubits],
        })
    }
    fn check(&self, q: usize) -> Result<()> {
        if q >= self.x.len() {
            Err(Error::Wires("frame site out of range".into()))
        } else {
            Ok(())
        }
    }
    /// Toggle a Pauli correction at one site.
    pub fn toggle(&mut self, q: usize, p: Pauli) -> Result<()> {
        self.check(q)?;
        if p != Pauli::Z {
            self.x[q] ^= true;
        }
        if p != Pauli::X {
            self.z[q] ^= true;
        }
        Ok(())
    }
    /// Propagate through Hadamard.
    pub fn h(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        std::mem::swap(&mut self.x[q], &mut self.z[q]);
        Ok(())
    }
    /// Propagate through S.
    pub fn s(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        self.z[q] ^= self.x[q];
        Ok(())
    }
    /// Propagate through CNOT.
    pub fn cnot(&mut self, c: usize, t: usize) -> Result<()> {
        self.check(c)?;
        self.check(t)?;
        if c == t {
            return Err(Error::Wires("repeated frame CNOT site".into()));
        }
        self.x[t] ^= self.x[c];
        self.z[c] ^= self.z[t];
        Ok(())
    }
    /// Whether this frame reverses the sign of a requested Pauli observable.
    pub fn flips(&self, product: &PauliProduct) -> Result<bool> {
        let mut sign = false;
        for &(q, p) in product.terms() {
            self.check(q)?;
            if p != Pauli::Z {
                sign ^= self.z[q];
            }
            if p != Pauli::X {
                sign ^= self.x[q];
            }
        }
        Ok(sign)
    }
}
