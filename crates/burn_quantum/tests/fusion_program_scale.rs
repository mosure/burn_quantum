//! Large lattice construction, cached ownership and persistence invariants.
use burn_quantum::photonic::{qec::SixRingTorus, *};

#[test]
fn large_lattice_roundtrip_rebuilds_consumption_and_rejects_duplicate_records() {
    let network = SixRingTorus::new(12, NodeId(0), NodeId(1), 6 * 12usize.pow(3)).unwrap();
    let original = network.program();
    assert_eq!(original.num_fusions(), 3 * 12usize.pow(3));
    let text = serde_json::to_string(original).unwrap();
    assert!(!text.contains("consumed"));
    let mut restored: FusionProgram = serde_json::from_str(&text).unwrap();
    assert_eq!(&restored, original);
    let before = restored.clone();
    let f = &original.fusions()[0];
    assert!(
        restored
            .fuse(f.photons[0], f.photons[1], NodeId(1))
            .is_err()
    );
    assert_eq!(restored, before);
    let fresh = restored
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    assert!(
        restored
            .fuse(fresh[0], PhotonId(usize::MAX), NodeId(1))
            .is_err()
    );
    restored.fuse(fresh[0], fresh[1], NodeId(1)).unwrap();
    let mut wire: serde_json::Value = serde_json::from_str(&text).unwrap();
    let fusions = wire["fusions"].as_array_mut().unwrap();
    fusions.push(fusions[0].clone());
    assert!(serde_json::from_value::<FusionProgram>(wire).is_err());
}
