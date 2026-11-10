use super::*;
use burn_quantum::photonic::fabric::{FabricConstraints, PacketRequest, PacketScheduler};
use burn_quantum::photonic::program::TimingOptions;
fn constraints(buffers: Vec<usize>) -> FabricConstraints {
    FabricConstraints {
        configure_ticks: vec![0; buffers.len()],
        buffers,
        max_search_states: 1024,
        max_reservations: 128,
    }
}
#[test]
fn finite_schedule_charges_every_wait_once_and_keeps_outputs_in_buffers() {
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
    let p = swap_program();
    let c = constraints(vec![2, 2, 1, 2]);
    let plan = p
        .compile_finite(
            &f,
            CompileOptions::default(),
            TimingOptions::serial(2, 1),
            c.clone(),
            &[5],
        )
        .unwrap();
    assert_eq!(plan.constraints(), Some(&c));
    assert_eq!(plan.fusions()[0].tick, 5);
    close(
        &plan.transmissions()[1].values(),
        &[0.8f64.powi(2); 2],
        1e-12,
    );
    // Stable earliest-arrival tie breaking retains the second photon at its source.
    assert_eq!(plan.routes()[1].route.hops[0].departure, 2);
    close(
        &plan.transmissions()[2].values(),
        &[0.98f64.powi(2); 2],
        1e-12,
    );
    close(
        &plan.transmissions()[0].values(),
        &[0.99f64.powi(6); 2],
        1e-12,
    );
    // Outputs plus temporarily stored consumed photons must fit, not only routes.
    let mut t = TimingOptions::serial(2, 1);
    t.resource_births = vec![0, 2];
    assert!(
        p.compile_finite(
            &f,
            CompileOptions::default(),
            t,
            constraints(vec![1, 2, 1, 2]),
            &[8]
        )
        .is_err()
    );
    assert!(
        p.compile_finite(
            &f,
            CompileOptions::default(),
            TimingOptions::serial(2, 1),
            constraints(vec![2, 2, 1, 1]),
            &[5]
        )
        .is_err()
    );
}
#[test]
fn finite_router_handles_zero_queue_and_switch_reconfiguration() {
    let f = fabric();
    let mut c = constraints(vec![2, 2, 0, 2]);
    let mut scheduler = PacketScheduler::new(f.clone(), c.clone()).unwrap();
    let requests = [0, 1].map(|from| PacketRequest {
        from: NodeId(from),
        to: NodeId(3),
        birth: 0,
        release: 0,
        consume: 5,
    });
    let pair = scheduler.pair_at(requests).unwrap();
    // The second packet waits at its source; there is no switch queue.
    assert_eq!(pair[1].hops[0].departure, 2);
    assert_eq!(pair[1].hops[1].departure, 3);
    c.configure_ticks[2] = 1;
    let mut scheduler = PacketScheduler::new(f, c).unwrap();
    assert!(scheduler.pair_at(requests).is_err());
    assert_eq!(scheduler.reservation_count(), 0);
}
#[test]
fn finite_pair_failure_is_atomic_and_budgeted() {
    let mut scheduler = PacketScheduler::new(fabric(), constraints(vec![2, 2, 1, 2])).unwrap();
    let impossible = [0, 1].map(|from| PacketRequest {
        from: NodeId(from),
        to: NodeId(3),
        birth: 0,
        release: 0,
        consume: 3,
    });
    assert!(scheduler.pair_at(impossible).is_err());
    assert_eq!(scheduler.reservation_count(), 0);
    let good = impossible.map(|r| PacketRequest { consume: 5, ..r });
    assert!(scheduler.pair_at(good).is_ok());
    let mut c = constraints(vec![2, 2, 1, 2]);
    c.max_search_states = 1;
    let mut scheduler = PacketScheduler::new(fabric(), c).unwrap();
    assert!(matches!(
        scheduler.route_at(good[0]),
        Err(Error::ResourceLimit { .. })
    ));
    assert_eq!(scheduler.reservation_count(), 0);
}

#[test]
fn source_configuration_overlaps_pre_release_storage_and_capacity_failure_is_atomic() {
    let f = Fabric::new(
        vec![
            node(NodeKind::Source, 0.9),
            node(NodeKind::Switch, 0.8),
            node(NodeKind::Fusion, 0.7),
        ],
        vec![link(0, 1, 1, 1.), link(1, 2, 1, 1.)],
    )
    .unwrap();
    let mut c = constraints(vec![1, 1, 1]);
    c.configure_ticks = vec![5, 1, 1];
    let request = PacketRequest {
        from: NodeId(0),
        to: NodeId(2),
        birth: 1,
        release: 3,
        consume: 11,
    };
    let mut schedule = PacketScheduler::new(f.clone(), c.clone()).unwrap();
    let route = schedule.route_at(request).unwrap();
    assert_eq!(route.hops[0].departure, 6);
    assert_eq!(route.hops[1].departure, 8);
    assert_eq!(route.arrival, 9);
    close(
        &route.transmission.values(),
        &[0.9f64.powi(3) * 0.8; 2],
        1e-12,
    );
    let complete = RailTransmission::uniform(0.9)
        .unwrap()
        .over_ticks(2)
        .then(route.transmission)
        .then(RailTransmission::uniform(0.7).unwrap().over_ticks(2));
    close(
        &complete.values(),
        &[0.9f64.powi(5) * 0.8 * 0.7f64.powi(2); 2],
        1e-12,
    );
    let reservations = schedule.reservation_count();
    assert!(
        schedule
            .route_at(PacketRequest {
                consume: u64::MAX,
                ..request
            })
            .is_err()
    );
    assert_eq!(schedule.reservation_count(), reservations);
    c.max_reservations = 1;
    let mut limited = PacketScheduler::new(f, c).unwrap();
    assert!(matches!(
        limited.route_at(request),
        Err(Error::ResourceLimit {
            resource: "packet reservations",
            ..
        })
    ));
    assert_eq!(limited.reservation_count(), 0);
}
