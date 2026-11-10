//! Independent pinned PsiQDK 2.5.0 local simulations, with explicit endian/phase maps.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::Tensor;
use burn_quantum::{gate::Axis, photonic::interop::*, *};
use common::values;
use num_complex::Complex64 as C;

#[test]
fn pinned_sdk_complex_states_and_terminal_readouts_match_native_execution() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("interop/psiqdk-2.5.0.json")).unwrap();
    assert_eq!(fixture["packages"]["psiqdk"], "2.5.0");
    assert_eq!(fixture["packages"]["psiqdk-workbench"], "4.55.3");
    assert_eq!(fixture["packages"]["workbench-sim-native"], "0.3.0");
    assert_eq!(fixture["packages"]["antlr4-python3-runtime"], "4.13.2");
    assert_eq!(fixture["packages"]["numpy"], "2.5.4");
    assert_eq!(fixture["packages"].as_object().unwrap().len(), 5);
    assert_eq!(fixture["schema"], 1);
    assert_eq!(fixture["python"], "3.12.13");
    assert_eq!(
        fixture["execution"],
        "local PsiQDK state-vector-sim and buffer filters; no provider/hardware"
    );
    assert_eq!(
        fixture["wire_mapping"],
        "Burn wire i is q[i]; Burn amplitude index is big-endian, PsiQDK index uses bit i"
    );
    assert_eq!(
        fixture["global_phase_policy"],
        "Equivalent up to one global phase per complete circuit; not a controlled-subroutine export"
    );
    let device = runtime::execution().unwrap().device;
    let engine = DenseSimulator::default();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 4);
    // These externally audited program hashes pin each independently simulated
    // fixture. The Python generator computes SHA256; Rust also regenerates and
    // compares the entire exact program below, without a new hashing dependency.
    let identities = [
        (
            "asymmetric_bound_u2",
            "4ca209b47b622216bf90c7440cfc4e4377376548130a518a10ba33c35ae3f2e9",
        ),
        (
            "mixed_polarity_multicontrol",
            "f1d1c986bbdfbaf5e87f80d1134dfd12396d0108d2323e397324bf36d2197784",
        ),
        (
            "dense_fourier_reordered_levels",
            "3c0d07b6a6335abf798874cf2be4df3ada374dc3cbfd425cdf033dc90a940826",
        ),
        (
            "nontrivial_readout_order",
            "709395361dcf89367341888e7d1e9a8e4f01007116ea751e7f64e3ef11304781",
        ),
    ];
    for (case, (name, hash)) in fixture["cases"].as_array().unwrap().iter().zip(identities) {
        assert_eq!(case["name"], name);
        assert_eq!(case["qasm_sha256"], hash);
        let source: Circuit = serde_json::from_value(case["circuit"].clone()).unwrap();
        let parameters: Vec<f64> = serde_json::from_value(case["parameters"].clone()).unwrap();
        let inputs: Vec<f64> = serde_json::from_value(case["inputs"].clone()).unwrap();
        let readouts: Vec<usize> =
            serde_json::from_value(case["export"]["readout_wires"].clone()).unwrap();
        let exported =
            OpenQasm3::export(&source, &parameters, &inputs, &readouts, Default::default())
                .unwrap();
        // Catch a stale fixture or changes to exported gates/bindings/readout order.
        assert_eq!(
            exported.program(),
            case["export"]["program"].as_str().unwrap()
        );
        assert_eq!(
            exported.qubits(),
            case["export"]["qubits"].as_u64().unwrap() as usize
        );
        assert_eq!(
            exported.emitted_gates(),
            case["export"]["emitted_gates"].as_u64().unwrap() as usize
        );
        assert_eq!(
            exported.native_gates(),
            case["export"]["native_gates"].as_u64().unwrap() as usize
        );
        assert!(exported.numerical_error() < 1e-7);
        let bound_parameters = (!parameters.is_empty())
            .then(|| Tensor::<1>::from_floats(parameters.as_slice(), &device));
        let bound_inputs = (!inputs.is_empty()).then(|| {
            Tensor::<1>::from_floats(inputs.as_slice(), &device).reshape([1, inputs.len()])
        });
        let native = engine
            .execute(&source, bound_parameters, bound_inputs, 1, &device)
            .unwrap();
        let real = values(native.amplitudes().real());
        let imag = values(native.amplitudes().imag());
        let native: Vec<C> = real
            .into_iter()
            .zip(imag)
            .map(|(r, i)| C::new(r, i))
            .collect();
        let sdk: Vec<[f64; 2]> =
            serde_json::from_value(case["psiqdk_amplitudes_little_endian"].clone()).unwrap();
        assert_eq!(sdk.len(), source.register().size());
        let sdk: Vec<C> = (0..sdk.len())
            .map(|index| {
                let v = sdk[exported.psiqdk_index(index).unwrap()];
                C::new(v[0], v[1])
            })
            .collect();
        let overlap: C = sdk
            .iter()
            .zip(&native)
            .map(|(external, native)| external.conj() * native)
            .sum();
        assert!(
            (overlap.norm() - 1.).abs() < 3e-6,
            "{} overlap {overlap}",
            case["name"]
        );
        let phase = overlap / overlap.norm();
        for (i, (external, native)) in sdk.iter().zip(&native).enumerate() {
            assert!(
                (*external * phase - *native).norm() < 3e-6,
                "{} amplitude {i}: {external} vs {native}, phase{phase}",
                case["name"]
            );
            assert!((external.norm_sqr() - native.norm_sqr()).abs() < 3e-6);
        }
        let measured: Vec<u64> =
            serde_json::from_value(case["psiqdk_weighted_readouts"].clone()).unwrap();
        let decoded = exported.decode_psiqdk_readouts(&measured).unwrap();
        if !case["expected_readouts"].is_null() {
            assert_eq!(case["name"], "nontrivial_readout_order");
            assert_eq!(measured, vec![4, 1, 0]);
            assert_eq!(exported.readout_wires(), &[2, 0, 1]);
            let expected: Vec<bool> =
                serde_json::from_value(case["expected_readouts"].clone()).unwrap();
            assert_eq!(decoded, expected);
            for (q, &bit) in exported.readout_wires().iter().zip(&decoded) {
                let probability: f64 = native
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        ((index >> (source.register().wires() - 1 - q)) & 1 == 1) == bit
                    })
                    .map(|(_, v)| v.norm_sqr())
                    .sum();
                assert!((probability - 1.).abs() < 3e-6);
            }
        } else {
            assert_ne!(case["name"], "nontrivial_readout_order");
            assert!(decoded.is_empty());
        }
    }
}

#[test]
fn export_rejects_invalid_bindings_unbounded_data_and_ambiguous_readout_mapping() {
    let mut source = Circuit::new(Register::qubits(3).unwrap(), 1, 1);
    source
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    source
        .rotate(2, Rotation::qubit(Axis::X, Angle::input(0)))
        .unwrap();
    let default = OpenQasmOptions::default();
    let export = OpenQasm3::export(&source, &[0.2], &[0.7], &[2, 0, 1], default).unwrap();
    assert_eq!(export.psiqdk_index(4).unwrap(), 1);
    assert_eq!(export.psiqdk_index(3).unwrap(), 6);
    assert!(export.psiqdk_index(8).is_err());
    assert_eq!(
        export.decode_psiqdk_readouts(&[4, 1, 0]).unwrap(),
        vec![true, true, false]
    );
    assert!(export.decode_psiqdk_readouts(&[1, 1, 0]).is_err());
    assert!(export.decode_psiqdk_readouts(&[4, 1]).is_err());
    for (parameters, inputs, wires) in [
        (vec![], vec![0.7], vec![]),
        (vec![f64::NAN], vec![0.7], vec![]),
        (vec![0.2], vec![0.7], vec![3]),
        (vec![0.2], vec![0.7], vec![1, 1]),
    ] {
        assert!(OpenQasm3::export(&source, &parameters, &inputs, &wires, default).is_err());
    }
    assert!(
        OpenQasm3::export(
            &source,
            &[0.2],
            &[0.7],
            &[],
            OpenQasmOptions {
                max_qubits: 2,
                ..default
            }
        )
        .is_err()
    );
    assert!(
        OpenQasm3::export(
            &source,
            &[0.2],
            &[0.7],
            &[],
            OpenQasmOptions {
                max_bytes: 32,
                ..default
            }
        )
        .is_err()
    );
    assert!(
        OpenQasm3::export(
            &Circuit::new(Register::new(vec![3]).unwrap(), 0, 0),
            &[],
            &[],
            &[],
            default
        )
        .is_err()
    );
    let mut overflow = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    overflow
        .rotate(
            0,
            Rotation::qubit(
                Axis::Y,
                Angle::Parameter {
                    index: 0,
                    scale: f64::MAX,
                    offset: 0.,
                },
            ),
        )
        .unwrap();
    assert!(OpenQasm3::export(&overflow, &[2.], &[], &[], default).is_err());
    let mut expansion = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    expansion
        .controlled_x(vec![(0, 1), (1, 1), (2, 1)], 3)
        .unwrap();
    assert!(
        OpenQasm3::export(
            &expansion,
            &[],
            &[],
            &[],
            OpenQasmOptions {
                synthesis: burn_quantum::photonic::compiler::SynthesisOptions {
                    max_native_gates: 2,
                    ..Default::default()
                },
                ..default
            }
        )
        .is_err()
    );
}
