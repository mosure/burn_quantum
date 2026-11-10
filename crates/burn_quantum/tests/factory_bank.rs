//! Shared-mode source correlation, exact enumeration and immutable calibration roundtrip.
use burn_quantum::photonic::factories::*;
#[test]
fn joint_outage_model_matches_exact_joint_distribution_and_fresh_invocations() {
    let source = |p| ResourceFactory::new(p, 2, 3, 1).unwrap();
    let bank = CorrelatedFactoryBank::new(
        "outage-and-brightness-v1".into(),
        vec![
            FactoryMode {
                probability: 0.25,
                sources: vec![source(0.), source(0.)],
            },
            FactoryMode {
                probability: 0.75,
                sources: vec![source(0.8), source(0.6)],
            },
        ],
    )
    .unwrap();
    assert!((bank.joint_yield() - 0.36).abs() < 1e-12);
    let restored: CorrelatedFactoryBank =
        serde_json::from_str(&serde_json::to_string(&bank).unwrap()).unwrap();
    let mut bins = [0; 4];
    let mut modes = [0; 2];
    let mut distinct = 0;
    for shot in 0..8192 {
        let r = bank.generate(123, shot, 0, 0, 3).unwrap();
        let r2 = restored.generate(123, shot, 0, 0, 3).unwrap();
        assert_eq!(r.mode, r2.mode);
        assert_eq!(r.sources, r2.sources);
        let pattern = usize::from(r.sources[0].available_at.is_some())
            + 2 * usize::from(r.sources[1].available_at.is_some());
        bins[pattern] += 1;
        modes[r.mode] += 1;
        assert_eq!(
            r.sources.iter().map(|s| s.photons_consumed).sum::<usize>(),
            4
        );
        if r.mode == 0 {
            assert_eq!(pattern, 0);
        }
        let next = bank.generate(123, shot, 1, 0, 3).unwrap();
        distinct += usize::from(next.sources != r.sources);
    }
    // Exact finite mixture, not a product of marginal yields.
    for (observed, expected) in bins.into_iter().zip([
        0.25 + 0.75 * 0.2 * 0.4,
        0.75 * 0.8 * 0.4,
        0.75 * 0.2 * 0.6,
        0.75 * 0.8 * 0.6,
    ]) {
        assert!((observed as f64 / 8192. - expected).abs() < 0.02);
    }
    assert!((modes[0] as f64 / 8192. - 0.25).abs() < 0.02);
    assert!(distinct > 2000);
    assert!(bank.generate(123, 0, 0, 0, 2).is_err());
    assert!(
        CorrelatedFactoryBank::new(
            "bad".into(),
            vec![FactoryMode {
                probability: 0.9,
                sources: vec![source(1.)]
            }]
        )
        .is_err()
    );
}

#[test]
fn deadline_preflight_covers_every_mode_before_sampling() {
    let short = ResourceFactory::new(1., 2, 1, 1).unwrap();
    let long = ResourceFactory::new(0., 2, 3, 2).unwrap();
    assert_eq!(short.worst_case_ticks(), 1);
    assert_eq!(long.worst_case_ticks(), 6);
    let bank = |long_probability| {
        CorrelatedFactoryBank::new(
            "mixed-duration-v1".into(),
            vec![
                FactoryMode {
                    probability: 1. - long_probability,
                    sources: vec![short.clone()],
                },
                FactoryMode {
                    probability: long_probability,
                    sources: vec![long.clone()],
                },
            ],
        )
        .unwrap()
    };
    for long_probability in [0., 0.5, 1.] {
        let bank = bank(long_probability);
        for shot in 0..64 {
            assert!(bank.generate(92, shot, 0, 5, 10).is_err());
            let result = bank.generate(92, shot, 0, 5, 11).unwrap();
            assert_eq!(result.sources[0].finish_tick, [6, 11][result.mode]);
            assert_eq!(result.sources[0].available_at, [Some(6), None][result.mode]);
        }
        assert!(bank.generate(92, 0, 0, u64::MAX - 5, u64::MAX).is_err());
    }
}

#[test]
fn mixture_shape_and_deserialized_source_budgets_are_validated() {
    let source = ResourceFactory::new(0.5, 2, 1, 2).unwrap();
    assert!(
        CorrelatedFactoryBank::new(
            "mismatched-source-counts".into(),
            vec![
                FactoryMode {
                    probability: 0.5,
                    sources: vec![source.clone()],
                },
                FactoryMode {
                    probability: 0.5,
                    sources: vec![source.clone(), source.clone()],
                },
            ],
        )
        .is_err()
    );
    let bank = CorrelatedFactoryBank::new(
        "overflowing-source-budget".into(),
        vec![FactoryMode {
            probability: 1.,
            sources: vec![source],
        }],
    )
    .unwrap();
    let mut serialized = serde_json::to_value(bank).unwrap();
    serialized["modes"][0]["sources"][0]["cycle_ticks"] = u64::MAX.into();
    assert!(serde_json::from_value::<CorrelatedFactoryBank>(serialized).is_err());
}
