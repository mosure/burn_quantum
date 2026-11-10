//! Fidelity-based QN-SPSA with a bounded dense host metric.
//! Squared pure-state overlaps give Fubini–Study geometry; squared Uhlmann fidelity
//! gives Bures geometry for smooth mixed-state families. Evaluators declare their
//! state family and overlap protocol; rank-changing boundaries require care.
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
