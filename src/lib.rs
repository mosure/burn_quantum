//! Quantum simulation and differentiable circuits on Burn 0.22.
//!
//! Public tensors use Burn's runtime [`Device`](burn::tensor::Device) dispatch.
//! [`QuantumBackend`] is a quantum execution contract, not a replacement for
//! Burn's fixed catalog of classical tensor backends. See the repository's
//! architecture document for extension and scaling boundaries.
//!
//! ```
//! use burn::tensor::Device;
//! use burn_quantum::{Circuit, DenseSimulator, QuantumBackend, Register, Unitary};
//! let mut circuit = Circuit::new(Register::qubits(2)?, 0, 0);
//! circuit.unitary(vec![0], Unitary::h())?;
//! circuit.unitary(vec![0, 1], Unitary::cnot())?;
//! let state = DenseSimulator::default()
//!     .execute(&circuit, None, None, 1, &Device::flex())?;
//! state.validate(1e-5)?;
//! # Ok::<(), burn_quantum::Error>(())
//! ```

pub mod backend;
pub mod circuit;
pub mod complex;
pub mod error;
pub mod gate;
pub mod gradient;
pub mod layout;
pub mod noise;
pub mod observable;
mod precision;
pub mod qml;
pub mod quantum_module;
pub mod sampling;
pub mod state;

pub use backend::{Capabilities, DenseSimulator, QuantumBackend, ResourceLimits};
pub use circuit::{Angle, Circuit, Operation, Rotation};
pub use complex::ComplexTensor;
pub use error::{Error, Result};
pub use gate::{Matrix, Unitary};
pub use layout::Register;
pub use noise::{DensityMatrix, KrausChannel};
pub use observable::{Hamiltonian, Observable};
pub use quantum_module::{QuantumLayer, QuantumLayerConfig};
pub use sampling::Samples;
pub use state::StateVector;
