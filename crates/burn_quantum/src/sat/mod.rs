//! Grover search for classical Boolean CNF satisfiability.
//!
//! Clauses compile into reversible compute/phase/uncompute circuits, without
//! enumerating solutions or constructing a truth-table oracle. This is a dense
//! simulation, not a polynomial-time SAT algorithm or a solver for QMA quantum SAT.
pub mod benchmarks;
mod cnf;
mod oracle;
mod search;

pub use cnf::Cnf;
pub use oracle::GroverSat;
pub use search::{SatOutcome, SearchConfig, SearchReport};
