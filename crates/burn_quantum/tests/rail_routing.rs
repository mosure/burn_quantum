//! Coherent rail channel occupancy, wavelength isolation, skew and survival objectives.
use burn_quantum::photonic::{fabric::*, *};
fn constraints(nodes: usize, capacity: usize) -> FabricConstraints {
    FabricConstraints {
        buffers: vec![capacity; nodes],
        configure_ticks: vec![0; nodes],
        max_search_states: 1000,
        max_reservations: 100,
    }
}
fn node(kind: NodeKind) -> FabricNode {
    FabricNode {
        kind,
        storage: RailTransmission::IDEAL,
    }
}
#[test]
fn survival_objective_uses_complete_wait_and_route_loss() {
    let fabric = Fabric::new(
        vec![
            node(NodeKind::Source),
            node(NodeKind::Switch),
            node(NodeKind::Fusion),
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(0.2).unwrap(),
            },
            FabricLink {
                from: NodeId(0),
                to: NodeId(1),
                ticks: 2,
                transmission: RailTransmission::uniform(0.95).unwrap(),
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 2,
                transmission: RailTransmission::uniform(0.95).unwrap(),
            },
        ],
    )
    .unwrap();
    let request = PacketRequest {
        from: NodeId(0),
        to: NodeId(2),
        birth: 0,
        release: 0,
        consume: 5,
    };
    let mut earliest = PacketScheduler::new(fabric.clone(), constraints(3, 1)).unwrap();
    let mut survival = earliest.clone();
    let a = earliest
        .route_with_policy(request, RoutePolicy::EarliestArrival)
        .unwrap();
    let b = survival
        .route_with_policy(request, RoutePolicy::MaximumSurvival)
        .unwrap();
    assert_eq!(a.arrival, 1);
    assert!((4..=5).contains(&b.arrival));
    assert_eq!(a.hops.len(), 1);
    assert_eq!(b.hops.len(), 2);
    assert!((b.transmission.values()[0] - 0.9025).abs() < 1e-12);
}
#[test]
fn independent_wavelengths_support_parallel_rails_and_skew_rejection_is_atomic() {
    let mut nodes = vec![node(NodeKind::Source), node(NodeKind::Fusion)];
    nodes[1].storage = RailTransmission::uniform(0.9).unwrap();
    let fabric = Fabric::new(
        nodes,
        vec![FabricLink {
            from: NodeId(0),
            to: NodeId(1),
            ticks: 2,
            transmission: RailTransmission::new(0.8, 0.7).unwrap(),
        }],
    )
    .unwrap();
    let config = |skew| WavelengthConstraints {
        wavelengths: vec!["lambda-a".into(), "lambda-b".into()],
        channels: vec![constraints(2, 2); 2],
        max_skew_ticks: skew,
        policy: RoutePolicy::EarliestArrival,
    };
    let request = PacketRequest {
        from: NodeId(0),
        to: NodeId(1),
        birth: 0,
        release: 0,
        consume: 5,
    };
    let mut scheduler = RailPacketScheduler::new(fabric.clone(), config(0)).unwrap();
    assert!(scheduler.route_at(request, [0, 0]).is_err());
    assert_eq!(scheduler.reservation_count(), 0);
    let route = scheduler.route_at(request, [0, 1]).unwrap();
    assert_eq!(route.skew_ticks, 0);
    assert_eq!(route.packet.arrival, 2);
    assert_eq!(route.packet.transmission.values(), [0.8, 0.7]);
    assert_eq!(scheduler.reservation_count(), 4);
    let mut scheduler = RailPacketScheduler::new(fabric, config(2)).unwrap();
    let route = scheduler.route_at(request, [0, 0]).unwrap();
    assert_eq!(route.skew_ticks, 2);
    assert!((route.packet.transmission.values()[0] - 0.8 * 0.9f64.powi(2)).abs() < 1e-12);
    assert_eq!(route.packet.transmission.values()[1], 0.7);
}
