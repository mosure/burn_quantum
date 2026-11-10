//! Bounded exact Fock optics with explicit bosonic normalization and native Burn tensors.
//!
//! Fine modes are coherent; explicit internal-mode kets model distinguishability.
//! Passive interferometers
//! conserve total photon number; loss requires a basis containing every lower sector.
//! Component matrices use `[output_mode, input_mode]` creation-operator conventions.
mod allocation;
mod basis;
mod bell;
mod boosted_noise;
mod calibration;
mod cost;
mod detector;
mod dynamics;
mod interferometer;
mod learning;
mod source;
mod state;
mod tomography;
mod wavepacket;

pub use basis::{FockBasis, ModeId, PhotonSector};
pub use detector::{DetectionDistribution, Detector};
pub use interferometer::Interferometer;
pub use state::{FockDensity, FockState};

pub use dynamics::{DetectorBank, DetectorEvent};
pub use source::CalibratedSource;
pub use wavepacket::{PhotonPacket, WavepacketLayout};

pub use bell::{BellEffect, BellRecord, GriceConfig, OpticalBellBranch, OpticalBellInstrument};

pub use learning::{
    ClassicalFisherEstimate, PhaseGradient, ShotEstimate, ShotHistogram,
    SinglePhotonPhaseExperiment, bures_metric_from_fidelities,
};

pub use boosted_noise::GriceNoiseConfig;

pub use calibration::{
    CalibratedNoiseEnsemble, CalibrationProbe, JointPhotonSource, MachZehnderCalibration,
    OpticalNoiseBranch,
};

pub use tomography::{
    OpticalOverlapEstimate, OpticalTomography, OpticalTomographyReport, OpticalTomographySetting,
};

pub use cost::{OpticalCostModel, OpticalResourceCost};

pub use allocation::{ShotAccumulator, ShotAllocation, ShotAllocationItem, ShotAllocationPolicy};
