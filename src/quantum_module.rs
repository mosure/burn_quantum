//! Native Burn trainable quantum layer.
use crate::{
    Circuit, DenseSimulator, Error, Observable, ResourceLimits, Result as QuantumResult,
    StateVector,
};
use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor, TensorData};

/// Serializable Burn configuration, stored alongside the layer's weight record.
#[derive(Config, Debug)]
pub struct QuantumLayerConfig {
    /// Validated circuit structure.
    pub circuit: Circuit,
    /// Ordered Hermitian readouts. May be empty for state/probability losses.
    pub observables: Vec<Observable>,
    /// Explicit initialization values, one per circuit parameter.
    pub initial: Vec<f64>,
    /// Simulator allocation budgets.
    #[config(default = "ResourceLimits::default()")]
    pub limits: ResourceLimits,
}

impl QuantumLayerConfig {
    /// Initialize native Burn parameters on the chosen device.
    pub fn init(&self, device: &Device) -> QuantumResult<QuantumLayer> {
        QuantumLayer::new(
            self.circuit.clone(),
            self.observables.clone(),
            self.initial.clone(),
            DenseSimulator::new(self.limits),
            device,
        )
    }
}

/// A variational circuit with ordinary Burn parameters and readout tensors.
///
/// Structural fields are skipped by Burn parameter traversal. Persist the
/// serialized [`Circuit`] and observables beside the Burn weight checkpoint.
#[derive(Module, Debug)]
pub struct QuantumLayer {
    /// Trainable angles, compatible with Burn optimizers and module traversal.
    pub parameters: Param<Tensor<1>>,
    #[module(skip)]
    circuit: Circuit,
    #[module(skip)]
    observables: Vec<Observable>,
    #[module(skip)]
    engine: DenseSimulator,
}
impl QuantumLayer {
    /// Initialize from explicit parameters for reproducible experiments.
    pub fn new(
        circuit: Circuit,
        observables: Vec<Observable>,
        initial: Vec<f64>,
        engine: DenseSimulator,
        device: &Device,
    ) -> QuantumResult<Self> {
        if initial.is_empty()
            || initial.len() != circuit.num_parameters()
            || initial.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Shape(
                "layer requires finite, nonempty initial parameters matching its circuit".into(),
            ));
        }
        for o in &observables {
            if circuit.register().local_size(o.wires())? != o.matrix().dimension() {
                return Err(Error::Shape("readout dimension mismatch".into()));
            }
        }
        let n = initial.len();
        let parameters = Param::from_data(TensorData::new(initial, [n]), device);
        Ok(Self {
            parameters,
            circuit,
            observables,
            engine,
        })
    }
    /// Immutable circuit structure.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// Ordered output observables.
    pub fn observables(&self) -> &[Observable] {
        &self.observables
    }
    /// Execute the underlying state, useful for custom losses and observables.
    pub fn state(&self, inputs: Option<Tensor<2>>, batch: usize) -> QuantumResult<StateVector> {
        let parameters = self.parameters.val();
        let device = parameters.device();
        self.circuit
            .run(&self.engine, Some(parameters), inputs, batch, &device)
    }
    /// Return differentiable features `[batch, num_observables]`.
    ///
    /// The explicit batch supports circuits without classical inputs. Input
    /// features, if declared, must have shape `[batch, circuit.num_inputs()]`.
    /// Returns an error if no readouts were configured; use [`Self::state`]
    /// for Born probabilities, Hamiltonian energies, or other state-based losses.
    pub fn forward(&self, inputs: Option<Tensor<2>>, batch: usize) -> QuantumResult<Tensor<2>> {
        if self.observables.is_empty() {
            return Err(Error::Shape(
                "forward requires at least one readout; use state for state-based losses".into(),
            ));
        }
        let state = self.state(inputs, batch)?;
        let columns: QuantumResult<Vec<_>> = self
            .observables
            .iter()
            .map(|o| Ok(o.expectation(&self.engine, &state)?.reshape([batch, 1])))
            .collect();
        Ok(Tensor::cat(columns?, 1))
    }
}
