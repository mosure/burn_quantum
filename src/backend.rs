//! Execution contract and the native Burn dense simulator.
use crate::complex::compatible;
use crate::error::checked_product;
use crate::{Circuit, ComplexTensor, Error, Observable, Register, Result, StateVector, Unitary};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// Preflight element limits. These are not a device allocator or an OOM guarantee.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Complex entries in a state (includes batches; density matrices count N²).
    pub max_state_elements: usize,
    /// Complex entries in a local operator (includes gate batches).
    pub max_matrix_elements: usize,
    /// Total forward and inverse indices for one local contraction.
    pub max_plan_indices: usize,
}
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_state_elements: 1 << 22,
            max_matrix_elements: 1 << 20,
            max_plan_indices: 1 << 24,
        }
    }
}
impl ResourceLimits {
    /// Check storage before allocating tensors or host vectors.
    pub fn check_state(&self, batch: usize, dimension: usize) -> Result<usize> {
        if batch == 0 {
            return Err(Error::Shape("batch must be positive".into()));
        }
        let n = checked_product([batch, dimension])?;
        check("state", n, self.max_state_elements)?;
        Ok(n)
    }
    pub(crate) fn check_matrix(&self, batch: usize, size: usize) -> Result<()> {
        check(
            "local matrix",
            checked_product([batch, size, size])?,
            self.max_matrix_elements,
        )
    }
    pub(crate) fn check_plan(&self, size: usize) -> Result<()> {
        check(
            "gather plan",
            checked_product([2, size])?,
            self.max_plan_indices,
        )
    }
}
fn check(resource: &'static str, required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::ResourceLimit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

/// Explicit engine properties; unsupported representations are never implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Supports heterogeneous finite local dimensions.
    pub mixed_qudits: bool,
    /// Native Burn reverse-mode differentiation through analytic execution.
    pub reverse_mode: bool,
    /// Exact open-system evolution through the separate density API.
    pub density_matrices: bool,
    /// Mid-circuit measurement and conditional branching.
    pub dynamic_control: bool,
}

/// Quantum executor, separate from Burn's classical tensor `Backend` trait.
///
/// Associated states allow alternative simulation representations. This initial
/// contract is synchronous and analytic; remote job submission needs a separate
/// asynchronous contract rather than pretending a QPU returns a dense tensor.
pub trait QuantumBackend {
    /// Engine-specific quantum state representation.
    type State;
    /// Supported capabilities.
    fn capabilities(&self) -> Capabilities;
    /// Execute a validated circuit, starting in the all-zero state.
    fn execute(
        &self,
        circuit: &Circuit,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        batch: usize,
        device: &Device,
    ) -> Result<Self::State>;
    /// Exact computational-basis distribution.
    fn probabilities(&self, state: &Self::State) -> Result<Tensor<2>>;
    /// Exact local Hermitian expectation per batch element.
    fn expectation(&self, state: &Self::State, observable: &Observable) -> Result<Tensor<1>>;
}

/// Portable composite-operation implementation on Burn's selected tensor device.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseSimulator {
    /// Allocation preflight budgets.
    pub limits: ResourceLimits,
}
impl DenseSimulator {
    /// Construct with explicit budgets.
    pub fn new(limits: ResourceLimits) -> Self {
        Self { limits }
    }
    /// Apply a checked constant unitary to ordered targets.
    pub fn apply(
        &self,
        state: &StateVector,
        wires: &[usize],
        gate: &Unitary,
    ) -> Result<StateVector> {
        self.limits.check_matrix(1, gate.matrix().dimension())?;
        let tensor = gate
            .matrix()
            .to_tensor(&state.device(), state.amplitudes().dtype())?;
        self.apply_tensor(state, wires, &tensor)
    }
    /// Apply an analytic tensor operator to ordered targets.
    ///
    /// Shape and resources are checked. Unitarity is the caller's responsibility;
    /// this path also supports nonunitary linear maps for observables/channels.
    pub fn apply_tensor(
        &self,
        state: &StateVector,
        wires: &[usize],
        operator: &ComplexTensor<3>,
    ) -> Result<StateVector> {
        let amplitudes = self.contract(state.register(), state.amplitudes(), wires, operator)?;
        StateVector::from_amplitudes(state.register().clone(), amplitudes, self.limits)
    }
    pub(crate) fn contract(
        &self,
        register: &Register,
        state: &ComplexTensor<2>,
        wires: &[usize],
        op: &ComplexTensor<3>,
    ) -> Result<ComplexTensor<2>> {
        let [batch, n] = state.dims();
        let k = register.local_size(wires)?;
        let [gb, rows, cols] = op.dims();
        if n != register.size() || rows != k || cols != k || (gb != 1 && gb != batch) {
            return Err(Error::Shape(
                "local operator or batch does not match state".into(),
            ));
        }
        compatible(&state.re, &op.re)?;
        compatible(&state.re, &op.im)?;
        compatible(&state.im, &op.re)?;
        compatible(&state.im, &op.im)?;
        self.limits.check_state(batch, n)?;
        self.limits.check_matrix(gb, k)?;
        self.limits.check_plan(n)?;
        let (forward, inverse) = register.permutation(wires)?;
        let device = state.device();
        let index = Tensor::<1, Int>::from_data(TensorData::new(forward, [n]), &device);
        let undo = Tensor::<1, Int>::from_data(TensorData::new(inverse, [n]), &device);
        let re = state
            .re
            .clone()
            .select(1, index.clone())
            .reshape([batch, n / k, k]);
        let im = state.im.clone().select(1, index).reshape([batch, n / k, k]);
        let ur = op.re.clone().transpose();
        let ui = op.im.clone().transpose();
        let out_re = crate::precision::matmul(re.clone(), ur.clone())
            - crate::precision::matmul(im.clone(), ui.clone());
        let out_im = crate::precision::matmul(re, ui) + crate::precision::matmul(im, ur);
        ComplexTensor::new(
            out_re.reshape([batch, n]).select(1, undo.clone()),
            out_im.reshape([batch, n]).select(1, undo),
        )
    }
}
impl QuantumBackend for DenseSimulator {
    type State = StateVector;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            mixed_qudits: true,
            reverse_mode: cfg!(feature = "autodiff"),
            density_matrices: true,
            dynamic_control: false,
        }
    }
    fn execute(
        &self,
        circuit: &Circuit,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        batch: usize,
        device: &Device,
    ) -> Result<StateVector> {
        circuit.run(self, parameters, inputs, batch, device)
    }
    fn probabilities(&self, state: &StateVector) -> Result<Tensor<2>> {
        Ok(state.probabilities())
    }
    fn expectation(&self, state: &StateVector, observable: &Observable) -> Result<Tensor<1>> {
        observable.expectation(self, state)
    }
}

pub(crate) fn supported_dtype(dtype: DType) -> Result<()> {
    if matches!(dtype, DType::F32 | DType::F64) {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "only F32 and F64 quantum simulation".into(),
        ))
    }
}
