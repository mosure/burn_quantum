//! Variational ground-state preparation for an open transverse-field Ising chain.
//!
//! Training uses local observables and exact Burn autodiff. Independent dense
//! diagonalization supplies a bounded small-system reference, not a scalable solver.
use crate::{
    Angle, Circuit, Error, Matrix, Observable, QuantumLayerConfig, Register, Result, Rotation,
    Unitary, gate::Axis,
};
use burn::{
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor},
};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

/// Portable system size and ansatz budget.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IsingConfig {
    /// Four to eight spin-one-half sites.
    pub qubits: usize,
    /// One to three entangling layers.
    pub depth: usize,
    /// One to 128 Adam updates.
    pub steps: usize,
}
impl Default for IsingConfig {
    fn default() -> Self {
        Self {
            qubits: 4,
            depth: 2,
            steps: 80,
        }
    }
}
impl IsingConfig {
    /// Validate before any exponentially sized allocation.
    pub fn validate(self) -> Result<()> {
        if !(4..=8).contains(&self.qubits)
            || !(1..=3).contains(&self.depth)
            || !(1..=128).contains(&self.steps)
        {
            return Err(Error::InvalidValue(
                "Ising VQE supports 4–8 spins, 1–3 layers and 1–128 updates".into(),
            ));
        }
        Ok(())
    }
}
/// Actual objective and angles at a completed optimizer update.
#[derive(Clone, Debug)]
pub struct IsingStep {
    /// Energy in units J=h=1.
    pub energy: f64,
    /// Trainable Ry angles.
    pub parameters: Vec<f64>,
}
/// Complete bounded experiment with an independent reference.
#[derive(Clone, Debug)]
pub struct IsingRun {
    /// Parameterized preparation circuit.
    pub circuit: Circuit,
    /// One descriptive label per operation.
    pub labels: Vec<String>,
    /// Initialization followed by every actual update.
    pub history: Vec<IsingStep>,
    /// Smallest eigenvalue of an independently assembled Hamiltonian.
    pub exact_energy: f64,
    /// Best spatially uniform product-state energy, an explicit classical baseline.
    pub product_energy: f64,
}
/// Real-amplitude layered ansatz; all spins are initialized in |+>.
pub fn ansatz(config: &IsingConfig) -> Result<(Circuit, Vec<String>)> {
    config.validate()?;
    let mut circuit = Circuit::new(
        Register::qubits(config.qubits)?,
        config.qubits * (config.depth + 1),
        0,
    );
    let mut labels = Vec::new();
    for wire in 0..config.qubits {
        circuit.unitary(vec![wire], Unitary::h())?;
        labels.push(format!("Prepare |+> q{wire}"));
    }
    for layer in 0..=config.depth {
        for wire in 0..config.qubits {
            circuit.rotate(
                wire,
                Rotation::qubit(Axis::Y, Angle::parameter(layer * config.qubits + wire)),
            )?;
            labels.push(format!("Variational Ry · layer {} · q{wire}", layer + 1));
        }
        if layer < config.depth {
            for wire in 0..config.qubits - 1 {
                circuit.push(crate::Operation::ControlledX {
                    controls: vec![(wire, 1)],
                    target: wire + 1,
                })?;
                labels.push(format!("Entangle q{wire} → q{}", wire + 1));
            }
        }
    }
    Ok((circuit, labels))
}
/// Local terms for H = -sum Z_i Z_(i+1) - sum X_i, with open boundaries.
pub fn observables(config: &IsingConfig) -> Result<Vec<Observable>> {
    config.validate()?;
    let mut terms = (0..config.qubits).map(Observable::x).collect::<Vec<_>>();
    for wire in 0..config.qubits - 1 {
        terms.push(Observable::new(
            vec![wire, wire + 1],
            Matrix::real(
                4,
                vec![
                    1., 0., 0., 0., 0., -1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1.,
                ],
            )?,
        )?);
    }
    Ok(terms)
}
/// Independent exact reference, strictly bounded to the supported small systems.
pub fn exact_ground_energy(config: &IsingConfig) -> Result<f64> {
    config.validate()?;
    let n = config.qubits;
    let dim = 1usize << n;
    let mut h = DMatrix::<f64>::zeros(dim, dim);
    for basis in 0..dim {
        for wire in 0..n {
            let mask = 1usize << (n - 1 - wire);
            h[(basis, basis ^ mask)] -= 1.;
            if wire + 1 < n {
                h[(basis, basis)] -= if (basis & mask == 0) == (basis & (mask >> 1) == 0) {
                    1.
                } else {
                    -1.
                };
            }
        }
    }
    Ok(SymmetricEigen::new(h).eigenvalues.min())
}
async fn read<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f64>> {
    tensor
        .into_data_async()
        .await
        .map_err(|e| Error::Readback(format!("{e:?}")))?
        .convert::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| Error::Readback(format!("{e:?}")))
}
/// Minimize energy with exact simulator gradients; no finite-shot or hardware claim.
pub async fn run(config: &IsingConfig, device: &Device) -> Result<IsingRun> {
    run_with_progress(config, device, |_| std::future::ready(())).await
}
/// Run with a cooperative callback after each measured iterate, for UI scheduling.
pub async fn run_with_progress<F, Fut>(
    config: &IsingConfig,
    device: &Device,
    mut on_update: F,
) -> Result<IsingRun>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let (circuit, labels) = ansatz(config)?;
    let training_device = device.clone().autodiff();
    let initial = (0..circuit.num_parameters())
        .map(|i| 0.12 * ((i as f64 + 1.) * 1.71).sin())
        .collect();
    let mut model = QuantumLayerConfig::new(circuit.clone(), observables(config)?, initial)
        .init(&training_device)?;
    let mut optimizer = AdamConfig::new().init();
    let mut history = Vec::with_capacity(config.steps + 1);
    for iteration in 0..=config.steps {
        let energy = -model.forward(None, 1)?.sum();
        history.push(IsingStep {
            energy: read(energy.clone()).await?[0],
            parameters: read(model.parameters.val()).await?,
        });
        if iteration < config.steps {
            let gradients = GradientsParams::from_grads(energy.backward(), &model);
            model = optimizer.step(0.045, model, gradients);
        }
        on_update(iteration).await;
    }
    let n = config.qubits as f64;
    // Minimize -(n-1) cos²(theta) - n sin(theta) analytically.
    let product_energy = -(n - 1.) - n * n / (4. * (n - 1.));
    Ok(IsingRun {
        circuit,
        labels,
        history,
        exact_energy: exact_ground_energy(config)?,
        product_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn open_chain_reference_matches_independent_free_fermion_spectrum() {
        for qubits in 4..=8 {
            let config = IsingConfig {
                qubits,
                ..Default::default()
            };
            let reference = -(1..=qubits)
                .map(|k| 2. * ((k as f64) * std::f64::consts::PI / (2. * qubits as f64 + 1.)).cos())
                .sum::<f64>();
            assert!((exact_ground_energy(&config).unwrap() - reference).abs() < 1e-10);
        }
        assert!(
            exact_ground_energy(&IsingConfig {
                qubits: 32,
                ..Default::default()
            })
            .is_err()
        );
    }
}
