use super::*;

#[test]
fn routes_choose_earliest_arrival_and_account_for_asymmetric_transmission() {
    let f = Fabric::new(
        vec![
            node(NodeKind::Source, 1.),
            node(NodeKind::Switch, 1.),
            node(NodeKind::Fusion, 1.),
        ],
        vec![
            link(0, 2, 8, 1.),
            link(0, 1, 2, 0.8),
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::new(0.5, 0.25).unwrap(),
            },
        ],
    )
    .unwrap();
    let route = f.route(NodeId(0), NodeId(2), 5, 20).unwrap();
    assert_eq!(route.arrival, 8);
    assert_eq!(
        route.hops.iter().map(|h| h.link).collect::<Vec<_>>(),
        vec![1, 2]
    );
    close(&route.transmission.values(), &[0.4, 0.2], 1e-12);
    assert!(f.route(NodeId(2), NodeId(0), 0, 10).is_err());
    assert!(f.route(NodeId(0), NodeId(2), 0, 2).is_err());
    assert!(f.route(NodeId(0), NodeId(2), u64::MAX, u64::MAX).is_err());
    assert!(f.route(NodeId(9), NodeId(0), 0, 10).is_err());
    assert_eq!(
        f.route(NodeId(1), NodeId(1), 5, 5).unwrap().transmission,
        RailTransmission::IDEAL
    );
}
#[test]
fn shared_links_are_reserved_and_waits_including_outputs_have_physical_loss() {
    let f = Fabric::new(
        vec![
            node(NodeKind::Source, 0.99),
            node(NodeKind::Source, 0.98),
            node(NodeKind::Switch, 0.9),
            node(NodeKind::Fusion, 0.8),
        ],
        vec![link(0, 2, 1, 1.), link(1, 2, 1, 1.), link(2, 3, 2, 1.)],
    )
    .unwrap();
    let plan = swap_program()
        .compile(
            &f,
            CompileOptions {
                detector_transmission: RailTransmission::uniform(0.95).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(plan.routes()[0].route.arrival, 3);
    assert_eq!(plan.routes()[1].route.arrival, 5);
    assert_eq!(plan.fusions()[0].tick, 5);
    assert_eq!(plan.finish_tick(), 6);
    close(
        &plan.transmissions()[1].values(),
        &[0.8f64.powi(2) * 0.95; 2],
        1e-12,
    );
    close(
        &plan.transmissions()[2].values(),
        &[0.9f64.powi(2) * 0.95; 2],
        1e-12,
    );
    close(
        &plan.transmissions()[0].values(),
        &[0.99f64.powi(6); 2],
        1e-12,
    );
    close(
        &plan.transmissions()[3].values(),
        &[0.98f64.powi(6); 2],
        1e-12,
    );
    assert_eq!(plan.outputs(), &[PhotonId(0), PhotonId(3)]);
    assert!(
        swap_program()
            .compile(
                &f,
                CompileOptions {
                    max_ticks: 5,
                    ..Default::default()
                }
            )
            .is_err()
    );
}
#[test]
fn topology_validation_and_serialization_reject_invalid_physics() {
    let f = fabric();
    let text = serde_json::to_string(&f).unwrap();
    assert_eq!(serde_json::from_str::<Fabric>(&text).unwrap(), f);
    assert!(
        serde_json::from_str::<Fabric>(&text.replace("\"version\":1", "\"version\":2")).is_err()
    );
    let mut unknown_noise: serde_json::Value = serde_json::from_str(&text).unwrap();
    unknown_noise["links"][0]["dephasing"] = serde_json::json!(0.5);
    assert!(serde_json::from_value::<Fabric>(unknown_noise).is_err());
    assert!(Fabric::new(vec![node(NodeKind::Source, 1.)], vec![link(0, 0, 0, 1.)]).is_err());
    let invalid = text.replace("\"ticks\":1", "\"ticks\":0");
    assert!(serde_json::from_str::<Fabric>(&invalid).is_err());
    assert!(
        FusionProgram::new()
            .compile(&f, CompileOptions::default())
            .is_err()
    );
    let mut bad = FusionProgram::new();
    bad.add_resource(ResourceState::bell_pair(), NodeId(3))
        .unwrap();
    assert!(bad.compile(&f, CompileOptions::default()).is_err());
}
