//! Explicit detector-graph decoders and fixed-shot threshold experiments.
//!
//! Erasure homology is exact. The unweighted union-find decoder is a bounded
//! synchronous-growth implementation, not minimum-weight matching. Reported
//! finite-size crossings do not prove an asymptotic or physical-device threshold.
//!
//! Primary model references: [Delfosse–Nickerson](https://arxiv.org/abs/1709.06218),
//! [Delfosse–Zémor](https://arxiv.org/abs/1703.01517),
//! [Dennis et al.](https://arxiv.org/abs/quant-ph/0110143), and
//! [Bartolucci et al.](https://doi.org/10.1038/s41467-023-36493-1).
//! Exact binomial interval equations follow the
//! [NIST handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
mod distance;
mod graph;
mod study;
pub use distance::LogicalDistance;
pub use graph::{
    DecoderBudget, DecoderGraph, DecoderWork, DetectorEdge, ErasureAnalysis, GraphDecode,
    GraphLimits, GraphStatus,
};
pub use study::{GraphNoiseStudy, SixRingErasureModel, ThresholdCounts};

mod stats;
pub use stats::{
    BinomialInterval, FiniteSizeBracket, FiniteSizeTrend, compare_sizes, crossing_bracket,
};

mod phenomenological;
pub use phenomenological::{PhenomenologicalMemory, PhenomenologicalNoise, RepeatedMeasurements};

mod records;
pub use records::{RecordCode, RecordCodeLimits, RecordDistance, RecordGraph};

mod cubical;
pub use cubical::CubicalRecordModel;
