//! Display conventions for reduced-qubit Bloch vectors; these never change the quantum state.
use serde::{Deserialize, Serialize};
/// Shared presentation for one or many qubit panels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlochMode {
    /// Shaded orthographic three-dimensional sphere with orbit and depth-cued great circles.
    #[default]
    Spatial,
    /// Probability/phase disk: filled area is P(1), azimuth is coherence phase.
    /// Purity is reported separately so a mixed state cannot look like a pure superposition.
    Planar,
    /// A labeled Cartesian projection disk with the omitted component reported separately.
    Projection,
}

/// Measurement and coherence readouts for a validated reduced-qubit Bloch vector.
///
/// For `rho = (I + x X + y Y + z Z) / 2`, computational-basis probabilities
/// depend only on `z`. A probability/phase disk uses a filled **area** fraction
/// of P(1), so its radius is `sqrt(P(1))` times the outer disk radius. Its angle
/// is the phase of `rho[1,0]`; this equals the relative amplitude phase for a
/// pure state. Probabilities and phase alone do not characterize mixed states:
/// retain [`Self::length`] or [`Self::purity`] as an independent readout.
///
/// These are display summaries, not state validation or normalization. Use a
/// validated [`super::SubsystemState`] as the source. The small clamps prevent
/// accepted floating-point roundoff from drawing beyond the physical bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlochReadout {
    /// Computational-basis probabilities, `[P(0), P(1)]`.
    pub probabilities: [f64; 2],
    /// Radius of the P(1) fill relative to the full disk, in `[0, 1]`.
    pub probability_radius: f64,
    /// Coherence phase in radians, in `[-pi, pi]`, or absent below display precision.
    /// At the poles and for diagonal mixtures there is no observable phase.
    pub phase: Option<f64>,
    /// Bloch-vector length, zero for a maximally mixed qubit and one for a pure qubit.
    pub length: f64,
    /// Reduced-state purity, `Tr(rho²) = (1 + length²) / 2`.
    pub purity: f64,
    /// Transverse magnitude `sqrt(x² + y²) = 2 |rho[0,1]|`, in `[0, 1]`.
    pub coherence: f64,
}
impl BlochReadout {
    /// Suppress meaningless phase arrows caused by roundoff in f32 simulation output.
    pub const PHASE_DISPLAY_TOLERANCE: f64 = 1e-6;

    /// Summarize a finite, physically valid `[X, Y, Z]` expectation vector.
    pub fn from_vector([x, y, z]: [f64; 3]) -> Self {
        let p1 = ((1. - z) * 0.5).clamp(0., 1.);
        let transverse = x.hypot(y);
        let length = transverse.hypot(z).min(1.);
        Self {
            probabilities: [1. - p1, p1],
            probability_radius: p1.sqrt(),
            phase: (transverse > Self::PHASE_DISPLAY_TOLERANCE).then(|| y.atan2(x)),
            length,
            purity: (1. + length * length) * 0.5,
            coherence: transverse.min(1.),
        }
    }
}
/// Plane shown by a two-dimensional Bloch projection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlochPlane {
    /// Horizontal X, vertical Y, hidden Z.
    Xy,
    /// Horizontal X, vertical Z, hidden Y.
    #[default]
    Xz,
    /// Horizontal Y, vertical Z, hidden X.
    Yz,
}
impl BlochPlane {
    /// Axis labels in horizontal, vertical, hidden order.
    pub fn axes(self) -> [&'static str; 3] {
        match self {
            Self::Xy => ["x", "y", "z"],
            Self::Xz => ["x", "z", "y"],
            Self::Yz => ["y", "z", "x"],
        }
    }
    /// Project without discarding the omitted component: [horizontal, vertical, hidden].
    pub fn project(self, v: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Xy => v,
            Self::Xz => [v[0], v[2], v[1]],
            Self::Yz => [v[1], v[2], v[0]],
        }
    }
    /// Cycle through all three coordinate planes.
    pub fn next(self) -> Self {
        match self {
            Self::Xy => Self::Xz,
            Self::Xz => Self::Yz,
            Self::Yz => Self::Xy,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn basis_states_have_certain_measurements_and_no_phase() {
        for (vector, expected) in [([0., 0., 1.], [1., 0.]), ([0., 0., -1.], [0., 1.])] {
            let readout = BlochReadout::from_vector(vector);
            assert_eq!(readout.probabilities, expected);
            close(readout.probability_radius.powi(2), expected[1]);
            assert_eq!(readout.phase, None);
            close(readout.length, 1.);
            close(readout.purity, 1.);
            close(readout.coherence, 0.);
        }
    }

    #[test]
    fn equatorial_eigenstates_preserve_phase_sign() {
        use std::f64::consts::{FRAC_PI_2, PI};
        for (vector, phase) in [
            ([1., 0., 0.], 0.),
            ([-1., 0., 0.], PI),
            ([0., 1., 0.], FRAC_PI_2),
            ([0., -1., 0.], -FRAC_PI_2),
        ] {
            let readout = BlochReadout::from_vector(vector);
            assert_eq!(readout.probabilities, [0.5, 0.5]);
            close(readout.probability_radius.powi(2), 0.5);
            close(readout.phase.unwrap(), phase);
            close(readout.purity, 1.);
            close(readout.coherence, 1.);
        }
    }

    #[test]
    fn density_matrix_readouts_distinguish_mixtures_from_superpositions() {
        // rho = [[0.75, 0], [0, 0.25]] has the same measurement probabilities
        // as sqrt(0.75)|0> + sqrt(0.25)|1>, but no coherence and lower purity.
        let mixed = BlochReadout::from_vector([0., 0., 0.5]);
        let pure = BlochReadout::from_vector([0.75_f64.sqrt(), 0., 0.5]);
        assert_eq!(mixed.probabilities, pure.probabilities);
        close(mixed.probability_radius, 0.5);
        assert_eq!(mixed.phase, None);
        close(mixed.purity, 0.625);
        close(pure.purity, 1.);
        let maximally_mixed = BlochReadout::from_vector([0.; 3]);
        assert_eq!(maximally_mixed.probabilities, [0.5, 0.5]);
        assert_eq!(maximally_mixed.phase, None);
        close(maximally_mixed.length, 0.);
        close(maximally_mixed.purity, 0.5);
    }

    #[test]
    fn readouts_match_amplitudes_for_unequal_superpositions() {
        // Construct alpha=sqrt(1-p), beta=sqrt(p)*exp(i*phase), independently
        // form rho10=beta*conj(alpha), and recover its Born probabilities.
        for p in [0.01_f64, 0.25, 0.5, 0.8, 0.99] {
            for phase in [-2.4_f64, -0.6, 0., 1.1, 2.9] {
                let alpha = (1. - p).sqrt();
                let beta_re = p.sqrt() * phase.cos();
                let beta_im = p.sqrt() * phase.sin();
                let vector = [
                    2. * alpha * beta_re,
                    2. * alpha * beta_im,
                    alpha * alpha - beta_re * beta_re - beta_im * beta_im,
                ];
                let readout = BlochReadout::from_vector(vector);
                close(readout.probabilities[0], alpha * alpha);
                close(
                    readout.probabilities[1],
                    beta_re * beta_re + beta_im * beta_im,
                );
                close(readout.probability_radius.powi(2), p);
                close(readout.phase.unwrap(), phase);
                close(readout.purity, 1.);
            }
        }
    }

    #[test]
    fn display_roundoff_does_not_create_false_phase_or_overflow_the_disk() {
        let at_north_pole = BlochReadout::from_vector([1e-8, -1e-8, 1.00000001]);
        assert_eq!(at_north_pole.phase, None);
        assert_eq!(at_north_pole.probabilities, [1., 0.]);
        close(at_north_pole.length, 1.);
        let at_south_pole = BlochReadout::from_vector([0., 0., -1.00000001]);
        close(at_south_pole.probability_radius, 1.);
    }
    #[test]
    fn projections_preserve_the_hidden_component_and_squared_length() {
        for plane in [BlochPlane::Xy, BlochPlane::Xz, BlochPlane::Yz] {
            for v in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.2, -0.4, 0.3]] {
                let p = plane.project(v);
                assert!(
                    (v.iter().map(|x| x * x).sum::<f64>() - p.iter().map(|x| x * x).sum::<f64>())
                        .abs()
                        < 1e-12
                );
                let axes = plane.axes();
                for (i, name) in axes.iter().enumerate() {
                    assert_eq!(
                        p[i],
                        v[["x", "y", "z"].iter().position(|a| a == name).unwrap()]
                    );
                }
            }
        }
    }
}
