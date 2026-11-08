//! Fidelity-based QN-SPSA with a dense Fubini–Study metric (QFI / 4).
mod config;
mod estimator;
mod metric;
mod optimizer;
mod state;
mod validation;

pub use config::QnSpsaConfig;
pub use estimator::{QnSpsaSample, qn_spsa_sample};
pub use optimizer::QnSpsa;
pub use state::{QnSpsaState, QnSpsaStep, QnSpsaStepReport};
