//! Bit-packed Clifford trajectories for balanced-loss resource/fusion networks.
//!
//! This host simulator has polynomial tableau storage and samples complete attempts.
//! Lost entangled photons are traced via hidden measurement trajectories, never just
//! deleted from the graph. Asymmetric attenuation is rejected because its no-loss
//! filter generally leaves the stabilizer family. No Burn GPU execution is claimed.
mod frame;
mod pauli;
mod reference;
mod simulator;
mod tableau;
pub use frame::PauliFrame;
pub use pauli::{Pauli, PauliProduct};
pub use reference::{
    FrameCheckpoint, PackedMeasurements, PauliNoise, ReferenceSampler, ReferenceSession,
};
pub use simulator::{StabilizerLimits, StabilizerShot, StabilizerSimulator};
pub use tableau::{Measurement, Tableau, TableauLimits};

pub(crate) use simulator::execute_trajectory;
