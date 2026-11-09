//! Local Hermitian observables and sparse sums of local terms.
use crate::{DenseSimulator, Error, Matrix, Result, StateVector, Unitary};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ObservableData {
    wires: Vec<usize>,
    matrix: Matrix,
}

/// Hermitian local operator. Target order uses the register's mixed-radix convention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ObservableData", into = "ObservableData")]
pub struct Observable {
    wires: Vec<usize>,
    matrix: Matrix,
}
impl Observable {
    /// Validate Hermiticity; target dimensions are validated against the state at execution.
    pub fn new(wires: Vec<usize>, matrix: Matrix) -> Result<Self> {
        if wires.is_empty()
            || wires
                .iter()
                .enumerate()
                .any(|(i, w)| wires[..i].contains(w))
        {
            return Err(Error::Wires(
                "observable targets empty or duplicated".into(),
            ));
        }
        matrix.check_hermitian(1e-10)?;
        Ok(Self { wires, matrix })
    }
    /// Pauli Z on one qubit.
    pub fn z(wire: usize) -> Self {
        Self::new(vec![wire], Unitary::z().matrix().clone()).unwrap()
    }
    /// Pauli X on one qubit.
    pub fn x(wire: usize) -> Self {
        Self::new(vec![wire], Unitary::x().matrix().clone()).unwrap()
    }
    /// Pauli Y on one qubit.
    pub fn y(wire: usize) -> Self {
        Self::new(vec![wire], Unitary::y().matrix().clone()).unwrap()
    }
    /// Ordered targets.
    pub fn wires(&self) -> &[usize] {
        &self.wires
    }
    /// Hermitian matrix.
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }
    /// Real expectation `<psi|O|psi>` per batch item, retaining gradients.
    pub fn expectation(&self, engine: &DenseSimulator, state: &StateVector) -> Result<Tensor<1>> {
        engine.limits.check_matrix(1, self.matrix.dimension())?;
        let op = self
            .matrix
            .to_tensor(&state.device(), state.amplitudes().dtype())?;
        let out = engine.apply_tensor(state, &self.wires, &op)?;
        let a = state.amplitudes();
        let b = out.amplitudes();
        Ok((a.real() * b.real() + a.imag() * b.imag())
            .sum_dim(1)
            .reshape([state.batch_size()]))
    }
    /// Variance `<O²> - <O>²`, computed without constructing `O²`.
    pub fn variance(&self, engine: &DenseSimulator, state: &StateVector) -> Result<Tensor<1>> {
        engine.limits.check_matrix(1, self.matrix.dimension())?;
        let op = self
            .matrix
            .to_tensor(&state.device(), state.amplitudes().dtype())?;
        let out = engine.apply_tensor(state, &self.wires, &op)?;
        Ok(out.norm_squared() - self.expectation(engine, state)?.square())
    }
}
impl TryFrom<ObservableData> for Observable {
    type Error = Error;
    fn try_from(v: ObservableData) -> Result<Self> {
        Self::new(v.wires, v.matrix)
    }
}
impl From<Observable> for ObservableData {
    fn from(v: Observable) -> Self {
        Self {
            wires: v.wires,
            matrix: v.matrix,
        }
    }
}

/// Real linear combination of local terms, without a dense global Hamiltonian.
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    terms: Vec<(f64, Observable)>,
}
impl Hamiltonian {
    /// Construct a sum; the empty sum represents zero.
    pub fn new(terms: Vec<(f64, Observable)>) -> Result<Self> {
        if terms.iter().any(|(weight, _)| !weight.is_finite()) {
            return Err(Error::InvalidValue(
                "nonfinite Hamiltonian coefficient".into(),
            ));
        }
        Ok(Self { terms })
    }
    /// Local terms and coefficients.
    pub fn terms(&self) -> &[(f64, Observable)] {
        &self.terms
    }
    /// Energy per batch element.
    pub fn expectation(&self, engine: &DenseSimulator, state: &StateVector) -> Result<Tensor<1>> {
        let mut result = state.norm_squared() * 0.;
        for (weight, observable) in &self.terms {
            result = result + observable.expectation(engine, state)? * *weight;
        }
        Ok(result)
    }
}
