//! Cnf conformance using independent references.
use super::*;

#[test]
fn cnf_and_dimacs_validate_normalize_and_round_trip() {
    let formula =
        Cnf::from_dimacs(" c comments\np cnf 3 4\n1 1\n-2 0 1 -1 0\n3 0\n1 -2 0\n").unwrap();
    assert_eq!(formula, Cnf::new(3, vec![vec![1, -2], vec![3]]).unwrap());
    assert!(formula.evaluate(&[true, false, true]).unwrap());
    assert!(!formula.evaluate(&[false, true, true]).unwrap());
    assert!(formula.evaluate(&[false]).is_err());
    assert_eq!(
        serde_json::from_str::<Cnf>(&serde_json::to_string(&formula).unwrap()).unwrap(),
        formula
    );
    assert!(
        Cnf::from_dimacs("p cnf 0 0")
            .unwrap()
            .evaluate(&[])
            .unwrap()
    );
    assert!(
        !Cnf::from_dimacs("p cnf 0 1\n0")
            .unwrap()
            .evaluate(&[])
            .unwrap()
    );
    for text in [
        "",
        "1 0",
        "p cnf 1 1\n1",
        "p cnf 1 0\n0",
        "p cnf 1 2\n1 0",
        "p cnf 1 1\n2 0",
        "p cnf 1 1\n-2147483648 0",
        "p cnf -1 0",
        "p cnf 1 0\np cnf 1 0",
        "p dnf 1 0",
        "p cnf 1 1\nx 0",
    ] {
        assert!(Cnf::from_dimacs(text).is_err(), "accepted {text:?}");
    }
    for clauses in [
        vec![vec![0]],
        vec![vec![-2]],
        vec![vec![i32::MIN]],
        vec![vec![1, -1, 9]],
    ] {
        assert!(Cnf::new(1, clauses).is_err());
    }
    assert!(Cnf::new(usize::MAX, vec![]).is_err());
    assert!(serde_json::from_str::<Cnf>(r#"{"variables":1,"clauses":[[2]]}"#).is_err());
}
