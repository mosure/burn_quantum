//! Oracle conformance using independent references.
use super::*;

#[test]
fn structured_gates_match_batched_mixed_radix_scalar_permutations() {
    let register = Register::new(vec![2, 3, 2]).unwrap();
    let data: Vec<_> = normalized(12, 2)
        .into_iter()
        .chain(normalized(12, 9))
        .collect();
    let original = state(register.clone(), &data, 2);
    let engine = DenseSimulator::default();
    for (controls, target) in [
        (vec![(1, 2), (0, 0)], 2),
        (vec![(2, 0), (1, 1)], 0),
        (vec![], 2),
    ] {
        let mut expected = vec![Complex64::new(0., 0.); 24];
        let mut phase_expected = data.clone();
        for b in 0..2 {
            for x in 0..12 {
                let mut digits = register.decode(x).unwrap();
                if controls.iter().all(|&(w, l)| digits[w] == l) {
                    digits[target] ^= 1;
                    phase_expected[b * 12 + x] = -data[b * 12 + x];
                }
                expected[b * 12 + register.encode(&digits).unwrap()] = data[b * 12 + x];
            }
        }
        let flipped = engine.controlled_x(&original, &controls, target).unwrap();
        complex_close(&complex_values(&flipped), &expected, 2e-6);
        complex_close(
            &complex_values(&engine.controlled_x(&flipped, &controls, target).unwrap()),
            &data,
            2e-6,
        );
        complex_close(
            &complex_values(&engine.phase_flip(&original, &controls).unwrap()),
            &phase_expected,
            2e-6,
        );
    }
}

#[test]
fn reversible_oracle_matches_every_assignment_and_uncomputes_all_ancillas() {
    let cases = [
        (3, vec![vec![1, -2], vec![2, 3], vec![-1, -3]]),
        (2, vec![vec![1, 2], vec![-1, 2]]),
        (2, vec![vec![1], vec![-1]]),
        (2, vec![vec![1, 1, -2], vec![1, -1]]),
        (2, vec![]),
        (2, vec![vec![]]),
        (0, vec![]),
        (0, vec![vec![]]),
    ];
    for (n, clauses) in cases {
        let sat = solver(n, clauses.clone());
        let register = sat.oracle().register().clone();
        let assignments = 1 << n;
        let width = register.size();
        let stride = width / assignments;
        let mut data = vec![Complex64::new(0., 0.); assignments * width];
        let mut expected = data.clone();
        for x in 0..assignments {
            let offset = x * width + x * stride;
            data[offset] = Complex64::from_polar(1.0, (x + 1) as f64 * 0.19);
            expected[offset] = data[offset] * if truth(&clauses, n, x) { -1. } else { 1. };
        }
        let input = state(register, &data, assignments);
        let output = sat
            .oracle()
            .run_from(&DenseSimulator::default(), input, None, None)
            .unwrap();
        complex_close(&complex_values(&output), &expected, 2e-6);
        output.validate(2e-6).unwrap();
        let restored = sat
            .oracle()
            .run_from(&DenseSimulator::default(), output, None, None)
            .unwrap();
        complex_close(&complex_values(&restored), &data, 2e-6);
    }
}

#[test]
fn diffusion_matches_complex_inversion_about_the_mean_for_each_ancilla() {
    let sat = solver(3, vec![vec![1, 2], vec![-2, 3]]);
    let register = sat.diffusion().register().clone();
    let n = register.size();
    let data: Vec<_> = normalized(n, 3)
        .into_iter()
        .chain(normalized(n, 11))
        .collect();
    let mut expected = data.clone();
    let work_states = n / 8;
    for b in 0..2 {
        for ancilla in 0..work_states {
            let mean: Complex64 = (0..8)
                .map(|x| data[b * n + x * work_states + ancilla])
                .sum::<Complex64>()
                / 8.;
            for x in 0..8 {
                expected[b * n + x * work_states + ancilla] =
                    2. * mean - data[b * n + x * work_states + ancilla];
            }
        }
    }
    let output = sat
        .diffusion()
        .run_from(
            &DenseSimulator::default(),
            state(register, &data, 2),
            None,
            None,
        )
        .unwrap();
    complex_close(&complex_values(&output), &expected, 2e-5);
}

#[test]
fn grover_probabilities_match_analytic_curve_for_zero_one_and_many_solutions() {
    let cases = [
        (3, vec![vec![1], vec![-2], vec![3]]),
        (3, vec![vec![1, 2]]),
        (2, vec![vec![1]]),
        (2, vec![vec![1], vec![-1]]),
        (2, vec![]),
    ];
    for (n, clauses) in cases {
        let sat = solver(n, clauses.clone());
        let count = (0..1 << n).filter(|&x| truth(&clauses, n, x)).count();
        let theta = (count as f64 / (1 << n) as f64).sqrt().asin();
        let wires: Vec<_> = (0..n).collect();
        for iterations in [0, 1, 2, 4] {
            let output = sat.amplify(iterations, &device()).unwrap();
            output.validate(1e-5).unwrap();
            let actual = values(output.marginal(&wires, ResourceLimits::default()).unwrap());
            let success = ((2 * iterations + 1) as f64 * theta).sin().powi(2);
            let expected: Vec<_> = (0..1 << n)
                .map(|x| {
                    if truth(&clauses, n, x) {
                        success / count as f64
                    } else {
                        (1. - success) / ((1 << n) - count) as f64
                    }
                })
                .collect();
            close(&actual, &expected, 2e-5);
        }
    }
}
