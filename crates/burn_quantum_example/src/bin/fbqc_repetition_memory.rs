//! Binary X-error erasure decoding for an encoded Bell memory, with a dense device oracle.
//! This is a syndrome/Pauli-frame reference, not photon-vacuum QEC or a topological FBQC threshold.
use burn_quantum::photonic::{
    qec::{ErasureDecode, ParityCode},
    stabilizer::{Pauli, PauliProduct, Tableau, TableauLimits},
};
use burn_quantum::{Circuit, DenseSimulator, Matrix, Observable, Register, Unitary};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut dense = Circuit::new(Register::qubits(4)?, 0, 0);
    dense.unitary(vec![0], Unitary::h())?;
    for q in 1..4 {
        dense.unitary(vec![0, q], Unitary::cnot())?;
    }
    dense.unitary(vec![1], Unitary::x())?; // A known erased-coordinate X fault.
    dense.unitary(vec![1], Unitary::x())?; // Its decoded frame correction.
    let state = dense.run(&engine, None, None, 1, &device)?;
    state.validate(2e-5)?;
    let zz = Observable::new(
        vec![0, 3],
        Matrix::real(
            4,
            vec![
                1., 0., 0., 0., 0., -1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1.,
            ],
        )?,
    )?;
    let mut all_x = vec![0.; 256];
    for i in 0..16 {
        all_x[(i ^ 15) * 16 + i] = 1.;
    }
    let xxxx = Observable::new(vec![0, 1, 2, 3], Matrix::real(16, all_x)?)?;
    let dense_z = values(zz.expectation(&engine, &state)?)?[0];
    let dense_x = values(xxxx.expectation(&engine, &state)?)?[0];
    ensure(
        dense_z > 1. - 2e-5 && dense_x > 1. - 2e-5,
        "dense logical Bell correlations were not restored",
    )?;

    let erasure_probability = 0.2f64;
    let mut ambiguity = vec![];
    let mut corrected_cases = 0usize;
    for distance in [3usize, 5, 7] {
        let code = ParityCode::repetition(distance)?;
        let mut ideal = Tableau::zero(distance + 1, TableauLimits::default())?;
        ideal.h(0)?;
        for q in 1..=distance {
            ideal.cnot(0, q)?;
        }
        let logical_z = PauliProduct::new(vec![(0, Pauli::Z), (distance, Pauli::Z)])?;
        let logical_x = PauliProduct::new((0..=distance).map(|q| (q, Pauli::X)).collect())?;
        let mut ambiguous_weight = 0.;
        let mut total = 0.;
        // Exact enumeration: each coordinate is unerased, erased/no X, or erased/X.
        for mask in 0..1usize << distance {
            let erased: Vec<_> = (0..distance).map(|q| mask & (1 << q) != 0).collect();
            let count = mask.count_ones() as i32;
            let weight = erasure_probability.powi(count)
                * (1. - erasure_probability).powi(distance as i32 - count)
                / 2f64.powi(count);
            for error in 0..1usize << distance {
                if error & !mask != 0 {
                    continue;
                }
                let errors: Vec<_> = (0..distance).map(|q| error & (1 << q) != 0).collect();
                let syndrome = code.syndrome(&errors)?;
                total += weight;
                match code.decode_erasures(&syndrome, &erased)? {
                    ErasureDecode::Decoded { correction, .. } => {
                        let mut memory = ideal.clone();
                        for (q, &fault) in errors.iter().enumerate() {
                            if fault {
                                memory.x(q)?;
                            }
                        }
                        for (q, &flip) in correction.iter().enumerate() {
                            if flip {
                                memory.x(q)?;
                            }
                        }
                        ensure(
                            memory.expectation(&logical_z)? == 1
                                && memory.expectation(&logical_x)? == 1,
                            "decoded memory changed its logical reference correlations",
                        )?;
                        corrected_cases += 1;
                    }
                    ErasureDecode::Ambiguous { .. } => ambiguous_weight += weight,
                    ErasureDecode::Inconsistent => {
                        return Err("generated erasure fault was inconsistent".into());
                    }
                }
            }
        }
        ensure(
            (total - 1.).abs() < 1e-10,
            "exact memory ensemble lost probability",
        )?;
        ensure(
            (ambiguous_weight - erasure_probability.powi(distance as i32)).abs() < 1e-12,
            "erasure ambiguity disagrees with exact repetition formula",
        )?;
        ambiguity.push(ambiguous_weight);
    }
    Report::new(
        "fbqc_repetition_memory",
        0,
        3,
        &[
            ("dense_logical_z_correlation", dense_z),
            ("dense_logical_x_correlation", dense_x),
            ("host_corrected_fault_cases", corrected_cases as f64),
            ("distance3_ambiguity_probability", ambiguity[0]),
            ("distance5_ambiguity_probability", ambiguity[1]),
            ("distance7_ambiguity_probability", ambiguity[2]),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn erasure_decoding_preserves_reference_entanglement_when_logically_determined() {
        super::run().unwrap();
    }
}
