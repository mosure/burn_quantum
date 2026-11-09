//! Evaluation-based quantum optimizers using native Burn parameter tensors.
//!
//! QN-SPSA needs new loss and fidelity evaluations at perturbed parameters; it
//! has an explicit callback-based step API rather than consuming backpropagated
//! gradients. Quantum evaluations remain on the caller's selected tensor device.
mod qnspsa;

pub use qnspsa::{
    QnSpsa, QnSpsaConfig, QnSpsaSample, QnSpsaState, QnSpsaStep, QnSpsaStepReport, qn_spsa_sample,
};
