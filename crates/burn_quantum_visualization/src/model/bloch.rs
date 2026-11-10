//! Display conventions for reduced-qubit Bloch vectors; these never change the quantum state.
use serde::{Deserialize, Serialize};
/// Shared presentation for one or many qubit panels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlochMode {
    /// Shaded orthographic three-dimensional sphere with orbit and depth-cued great circles.
    #[default]
    Spatial,
    /// A labeled Cartesian projection disk with the omitted component reported separately.
    Planar,
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
