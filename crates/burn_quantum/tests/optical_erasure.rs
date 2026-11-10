//! Joint optical loss law versus existing independent stabilizer-code recovery.
use burn_quantum::photonic::{qec::*, *};

#[test]
fn optical_joint_law_matches_encoded_stabilizer_recovery_and_published_marginals() {
    for loss in [0., 0.002, 0.027, 0.2, 1.] {
        for boost in [0, 1, 3] {
            let model = OpticalFusionLoss::new(loss, boost, true).unwrap();
            let p = model.physical_distribution().probabilities();
            let mut expected = [0.; 4];
            for orientation in [ShorOrientation::Standard, ShorOrientation::Dual] {
                let mut program = FusionProgram::new();
                let photons = program
                    .add_resource(ResourceState::graph(8, vec![]).unwrap(), NodeId(0))
                    .unwrap();
                let fusion = EncodedFusion::append(
                    &mut program,
                    ShorBlock::with_orientation(photons[..4].try_into().unwrap(), orientation)
                        .unwrap(),
                    ShorBlock::with_orientation(photons[4..].try_into().unwrap(), orientation)
                        .unwrap(),
                    NodeId(1),
                )
                .unwrap();
                for pattern in 0usize..256 {
                    let masks: [usize; 4] = std::array::from_fn(|q| (pattern >> (2 * q)) & 3);
                    let weight: f64 = masks.iter().map(|&m| p[m]).product();
                    let records = masks.map(|m| FusionParities {
                        xx: (m & 1 == 0).then_some(Parity::Even),
                        zz: (m & 2 == 0).then_some(Parity::Even),
                    });
                    let EncodedFusionRecovery::Consistent(logical) =
                        fusion.recover_local_parities(records)
                    else {
                        panic!("zero records must be consistent")
                    };
                    let mask =
                        usize::from(logical.xx.is_none()) + 2 * usize::from(logical.zz.is_none());
                    expected[mask] += weight / 2.;
                }
            }
            let actual = model.distribution();
            for (a, b) in actual.probabilities().iter().zip(expected) {
                assert!((a - b).abs() < 2e-13);
            }
            assert!((actual.probabilities().iter().sum::<f64>() - 1.).abs() < 2e-13);
            let p0 =
                1. - (1. - model.failure_probability() / 2.) * (1. - loss).powi(1 << (boost + 1));
            let penc = ((1. - (1. - p0).powi(2)).powi(2) + 1. - (1. - p0.powi(2)).powi(2)) / 2.;
            for marginal in actual.marginals() {
                assert!((marginal - penc).abs() < 2e-13);
            }
        }
    }
}

#[test]
fn joint_losses_are_not_independent_and_costs_include_boosting() {
    let model = OpticalFusionLoss::new(0.01, 1, false).unwrap();
    let distribution = model.distribution();
    assert!((distribution.probabilities()[3] - (1. - 0.99f64.powi(4))).abs() < 1e-14);
    assert!(
        (distribution.probabilities()[3]
            - distribution.marginals()[0] * distribution.marginals()[1])
            .abs()
            > 0.01
    );
    let encoded = OpticalFusionLoss::new(0., 1, true).unwrap();
    assert_eq!(
        (
            encoded.physical_fusions(),
            encoded.signal_photons(),
            encoded.boosting_photons()
        ),
        (4, 8, 8)
    );
    assert!((encoded.distribution().marginals()[0] - 11. / 256.).abs() < 1e-14);
    for mask in 0..4 {
        let ps = distribution.probabilities();
        let u = ps[..mask].iter().sum::<f64>() + ps[mask] / 2.;
        assert_eq!(
            distribution.sample(u).unwrap(),
            [mask & 1 != 0, mask & 2 != 0]
        );
    }
}

#[test]
fn inverse_and_invalid_models_are_explicit() {
    assert!(OpticalFusionLoss::new(f64::NAN, 1, true).is_err());
    assert!(OpticalFusionLoss::new(0., 9, true).is_err());
    let unboosted = OpticalFusionLoss::new(0., 0, false).unwrap();
    assert_eq!(unboosted.photon_loss_at_marginal(0.1198).unwrap(), None);
    let model = OpticalFusionLoss::new(0., 1, true).unwrap();
    let loss = model.photon_loss_at_marginal(0.1198).unwrap().unwrap();
    assert!(loss > 0.02 && loss < 0.03);
    let mapped = OpticalFusionLoss::new(loss, 1, true)
        .unwrap()
        .distribution()
        .marginals()[0];
    assert!((mapped - 0.1198).abs() < 1e-13);
    assert!(model.photon_loss_at_marginal(-0.1).is_err());
    assert!(model.distribution().sample(1.).is_err());
    assert!(model.distribution().sample(f64::NAN).is_err());
}
