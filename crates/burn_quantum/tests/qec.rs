//! Exhaustive erasure-supported fault enumeration for binary syndrome relations.
use burn_quantum::photonic::qec::{ErasureDecode, ParityCode};
fn bits(n: usize, value: usize) -> Vec<bool> {
    (0..n).map(|i| value & (1 << i) != 0).collect()
}
#[test]
fn repetition_erasure_decoder_matches_exhaustive_faults_and_logical_ambiguity() {
    for n in [3, 5, 7] {
        let code = ParityCode::repetition(n).unwrap();
        for mask in 0..1usize << n {
            let erased = bits(n, mask);
            for error in 0..1usize << n {
                if error & !mask != 0 {
                    continue;
                }
                let fault = bits(n, error);
                let syndrome = code.syndrome(&fault).unwrap();
                match code.decode_erasures(&syndrome, &erased).unwrap() {
                    ErasureDecode::Decoded {
                        correction,
                        logical_flips,
                    } => {
                        assert_ne!(mask, (1 << n) - 1);
                        assert_eq!(code.syndrome(&correction).unwrap(), syndrome);
                        assert_eq!(logical_flips, code.logical_flips(&fault).unwrap());
                        assert!(correction.iter().zip(&erased).all(|(&c, &e)| !c || e));
                    }
                    ErasureDecode::Ambiguous { logicals } => {
                        assert_eq!(mask, (1 << n) - 1);
                        assert_eq!(logicals, vec![0]);
                    }
                    ErasureDecode::Inconsistent => panic!("valid erasure fault rejected"),
                }
            }
        }
    }
}
#[test]
fn dependent_checks_distinguish_inconsistency_from_harmless_degeneracy() {
    let code = ParityCode::new(
        3,
        vec![vec![true, true, false], vec![true, true, false]],
        vec![vec![true, true, false]],
    )
    .unwrap();
    assert_eq!(
        code.decode_erasures(&[true, false], &[true, true, false])
            .unwrap(),
        ErasureDecode::Inconsistent
    );
    assert!(
        matches!(code.decode_erasures(&[true,true],&[true,true,true]).unwrap(),ErasureDecode::Decoded{logical_flips,..} if logical_flips==[true])
    );
    assert_eq!(
        code.decode_erasures(&[true, true], &[false, false, true])
            .unwrap(),
        ErasureDecode::Inconsistent
    );
    let ambiguous = ParityCode::new(2, vec![], vec![vec![true, false], vec![false, true]]).unwrap();
    assert_eq!(
        ambiguous.decode_erasures(&[], &[true, false]).unwrap(),
        ErasureDecode::Ambiguous { logicals: vec![0] }
    );
}
#[test]
fn parity_relations_validate_persistence_widths_and_budgets() {
    let code = ParityCode::repetition(5).unwrap();
    let text = serde_json::to_string(&code).unwrap();
    assert_eq!(serde_json::from_str::<ParityCode>(&text).unwrap(), code);
    assert!(
        serde_json::from_str::<ParityCode>(&text.replace("\"version\":1", "\"version\":2"))
            .is_err()
    );
    assert!(ParityCode::new(3, vec![vec![true]], vec![]).is_err());
    assert!(ParityCode::repetition(4).is_err());
    assert!(ParityCode::repetition(usize::MAX).is_err());
    assert!(code.decode_erasures(&[], &[false; 5]).is_err());
    assert!(code.syndrome(&[false; 4]).is_err());
}
