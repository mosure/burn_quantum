//! Bounded scientific visualization and interactive Bevy exploration of quantum experiments.
//!
//! The model and asynchronous simulation adapters are usable without a window:
//! disable default features to integrate them with another application. Enable
//! `viewer` for the Feathers workbench and `gpu-bridge` for shared WGPU resources.
#[cfg(feature = "viewer")]
pub mod app;
#[cfg(feature = "gpu-bridge")]
pub mod bridge;
pub mod demo;
pub mod io;
pub mod model;
pub mod simulation;
#[cfg(feature = "viewer")]
mod view;
