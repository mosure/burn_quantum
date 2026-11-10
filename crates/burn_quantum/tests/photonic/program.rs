use super::*;

#[test]
fn compiler_rejects_consumed_photons_and_roundtrips_validated_programs() {
    let mut program = swap_program();
    let before = program.clone();
    assert!(program.fuse(PhotonId(1), PhotonId(3), NodeId(3)).is_err());
    assert_eq!(program, before);
    let text = serde_json::to_string(&program).unwrap();
    assert_eq!(program.resources()[0].state.hadamards(), &[1]);
    assert_eq!(program.fusions()[0].photons, [PhotonId(1), PhotonId(2)]);
    assert_eq!(
        serde_json::from_str::<FusionProgram>(&text).unwrap(),
        program
    );
    assert!(
        serde_json::from_str::<FusionProgram>(&text.replace("\"version\":1", "\"version\":8"))
            .is_err()
    );
    assert!(
        serde_json::from_str::<FusionProgram>(
            &text.replace("\"photons\":[1,2]", "\"photons\":[1,1]")
        )
        .is_err()
    );
    // A future adaptive field must not be silently ignored by this static executor.
    let mut adaptive: serde_json::Value = serde_json::from_str(&text).unwrap();
    adaptive["fusions"][0]["condition"] = serde_json::json!("previous_success");
    assert!(serde_json::from_value::<FusionProgram>(adaptive).is_err());
    assert!(
        program
            .compile(
                &fabric(),
                CompileOptions {
                    max_photons: 3,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(
        program
            .compile(
                &fabric(),
                CompileOptions {
                    max_fusions: 0,
                    ..Default::default()
                }
            )
            .is_err()
    );
    let a = program
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    let b = program
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
}
#[test]
fn complete_loss_remains_a_counted_attempt_with_vacuum_outputs() {
    let erased_fabric = Fabric::new(
        vec![
            node(NodeKind::Source, 0.),
            node(NodeKind::Source, 0.),
            node(NodeKind::Switch, 1.),
            node(NodeKind::Fusion, 1.),
        ],
        vec![link(0, 2, 1, 0.), link(1, 2, 1, 0.), link(2, 3, 2, 1.)],
    )
    .unwrap();
    let plan = swap_program()
        .compile(&erased_fabric, CompileOptions::default())
        .unwrap();
    let run = DenseFusionSimulator::default()
        .execute(&plan, 1, &device(), DType::F32)
        .unwrap();
    close(&values(run.branches()[4].probability()), &[1.], 2e-5);
    assert!(run.branches()[0].conditional_state(1e-8).is_err());
    let shots = run.sample(128, 17, 128).unwrap();
    assert_eq!(shots[0].len(), 128);
    assert!(
        shots[0]
            .iter()
            .all(|s| s.branch == 4 && s.output_digits == [2, 2])
    );
}
#[test]
fn lossy_bell_swap_matches_analytic_probabilities_conditional_entanglement_and_shots() {
    let plan = swap_program()
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    let run = DenseFusionSimulator::default()
        .execute(&plan, 2, &device(), DType::F32)
        .unwrap();
    close(&values(run.total_probability()), &[1., 1.], 2e-5);
    let expected = [0.18, 0.18, 0.18, 0.18, 0.28];
    for (branch, &p) in run.branches().iter().zip(&expected) {
        close(&values(branch.probability()), &[p, p], 2e-5);
        branch
            .conditional_state(1e-8)
            .unwrap()
            .unwrap()
            .validate(2e-5)
            .unwrap();
    }
    for (i, branch) in run.branches()[..2].iter().enumerate() {
        let rho = branch.conditional_state(1e-8).unwrap().unwrap();
        let re = values(rho.entries().real());
        let sign = if i == 0 { 1. } else { -1. };
        for batch in 0..2 {
            let values = &re[batch * 81..(batch + 1) * 81];
            assert!((values[10] - 0.5).abs() < 2e-5);
            assert!((values[30] - 0.5).abs() < 2e-5);
            assert!((values[12] - 0.5 * sign).abs() < 2e-5);
        }
    }
    let samples = run.sample(5000, 19, 10000).unwrap();
    assert_eq!(samples, run.sample(5000, 19, 10000).unwrap());
    assert!(run.sample(5000, 19, 9999).is_err());
    for batch in samples {
        let success = batch.iter().filter(|s| s.branch < 2).count() as f64 / 5000.;
        assert!((success - 0.36).abs() < 0.025);
        assert!(batch.iter().any(|s| s.branch == 4));
        assert!(
            batch
                .iter()
                .filter(|s| s.branch < 2)
                .all(|s| s.output_digits[0] != s.output_digits[1])
        );
    }
}
#[test]
fn repeated_fusions_consume_stable_ids_and_can_return_only_classical_results() {
    let mut program = swap_program();
    program.fuse(PhotonId(0), PhotonId(3), NodeId(3)).unwrap();
    let plan = program
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    let run = DenseFusionSimulator::default()
        .execute(&plan, 1, &device(), DType::F32)
        .unwrap();
    assert!(run.outputs().is_empty());
    assert_eq!(run.branches().len(), 25);
    assert!(
        run.branches()
            .iter()
            .all(|b| b.outcomes().len() == 2 && b.unnormalized_state().is_none())
    );
    close(&values(run.total_probability()), &[1.], 2e-5);
    assert!(
        run.sample(64, 3, 64).unwrap()[0]
            .iter()
            .all(|s| s.output_digits.is_empty())
    );
    let bounded = DenseFusionSimulator {
        limits: SimulationLimits {
            max_branches: 24,
            ..Default::default()
        },
    };
    assert!(matches!(
        bounded.execute(&plan, 1, &device(), DType::F32),
        Err(Error::ResourceLimit {
            resource: "fusion branches",
            ..
        })
    ));
}
#[test]
fn compiler_can_plan_more_photons_than_the_dense_reference_can_allocate() {
    let mut program = FusionProgram::new();
    for _ in 0..100 {
        program
            .add_resource(ResourceState::bell_pair(), NodeId(0))
            .unwrap();
    }
    let plan = program
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    assert_eq!(plan.outputs().len(), 200);
    assert!(
        DenseFusionSimulator::default()
            .execute(&plan, 1, &device(), DType::F32)
            .is_err()
    );
    let plan = swap_program()
        .compile(&fabric(), CompileOptions::default())
        .unwrap();
    let limited = DenseFusionSimulator {
        limits: SimulationLimits {
            dense: ResourceLimits {
                max_state_elements: 100,
                ..Default::default()
            },
            ..Default::default()
        },
    };
    assert!(matches!(
        limited.execute(&plan, 1, &device(), DType::F32),
        Err(Error::ResourceLimit { .. })
    ));
    assert!(
        DenseFusionSimulator::default()
            .execute(&plan, 0, &device(), DType::F32)
            .is_err()
    );
}
