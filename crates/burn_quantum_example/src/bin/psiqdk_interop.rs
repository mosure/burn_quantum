//! Bound OpenQASM3 interchange inputs for the pinned local PsiQDK qualification.
//! This executable uses native Burn; the companion regeneration script runs PsiQDK.
use burn::tensor::Tensor;
use burn_quantum::{gate::Axis, photonic::interop::*, *};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device};
use serde::Serialize;
#[derive(Serialize)]
struct Case {
    name: &'static str,
    circuit: Circuit,
    parameters: Vec<f64>,
    inputs: Vec<f64>,
    export: OpenQasm3,
    expected_readouts: Option<Vec<bool>>,
}
#[derive(Serialize)]
struct Run {
    #[serde(flatten)]
    report: Report,
    global_phase_policy: &'static str,
    cases: Vec<Case>,
}
fn run() -> ExampleResult<Run> {
    let mut circuits = vec![];
    let mut asymmetric = Circuit::new(Register::qubits(3)?, 1, 1);
    asymmetric.rotate(
        0,
        Rotation::qubit(
            Axis::Y,
            Angle::Parameter {
                index: 0,
                scale: 1.25,
                offset: -0.2,
            },
        ),
    )?;
    asymmetric.rotate(
        2,
        Rotation::qubit(
            Axis::X,
            Angle::Input {
                index: 0,
                scale: -0.7,
                offset: 0.13,
            },
        ),
    )?;
    let phase = Unitary::phase(0.47)?.matrix().entries()[3];
    let global = Unitary::phase(0.31)?.matrix().entries()[3];
    let c = (0.71f64 / 2.).cos();
    let s = (0.71f64 / 2.).sin();
    let gate = Unitary::new(Matrix::new(
        2,
        vec![c.into(), -s * phase.conj(), s * phase, c.into()]
            .into_iter()
            .map(|v| v * global)
            .collect(),
    )?)?;
    asymmetric
        .unitary(vec![1], gate)?
        .unitary(vec![2, 0], Unitary::cnot())?;
    asymmetric.unitary(vec![0], Unitary::t())?;
    circuits.push((
        "asymmetric_bound_u2",
        asymmetric,
        vec![0.63],
        vec![-0.41],
        vec![],
        None,
    ));
    let mut controls = Circuit::new(Register::qubits(4)?, 0, 0);
    for q in 0..3 {
        controls.unitary(vec![q], Unitary::h())?;
    }
    controls.rotate(3, Rotation::qubit(Axis::Y, Angle::Constant(0.81)))?;
    controls.controlled_x(vec![(0, 0), (2, 1), (1, 1)], 3)?;
    controls.phase_flip(vec![(3, 1), (1, 0), (0, 1)])?;
    controls
        .unitary(vec![2], Unitary::t())?
        .unitary(vec![3, 0], Unitary::swap(2)?)?
        .unitary(vec![1, 2], Unitary::z().controlled(2, 1)?)?;
    controls.phase_flip(vec![])?; // global minus is explicitly not preserved by interchange.
    circuits.push((
        "mixed_polarity_multicontrol",
        controls,
        vec![],
        vec![],
        vec![],
        None,
    ));
    let mut dense = Circuit::new(Register::qubits(3)?, 1, 1);
    dense.unitary(vec![2], Unitary::h())?;
    dense.rotate(
        1,
        Rotation {
            axis: Axis::Y,
            levels: [1, 0],
            angle: Angle::input(0),
        },
    )?;
    dense.rotate(
        2,
        Rotation {
            axis: Axis::Z,
            levels: [1, 0],
            angle: Angle::parameter(0),
        },
    )?;
    dense.unitary(vec![2, 0], Unitary::fourier(4)?)?;
    dense.rotate(0, Rotation::qubit(Axis::X, Angle::Constant(-0.28)))?;
    circuits.push((
        "dense_fourier_reordered_levels",
        dense,
        vec![0.77],
        vec![0.93],
        vec![],
        None,
    ));
    let mut readout = Circuit::new(Register::qubits(3)?, 0, 0);
    readout
        .unitary(vec![0], Unitary::x())?
        .unitary(vec![2], Unitary::x())?;
    circuits.push((
        "nontrivial_readout_order",
        readout,
        vec![],
        vec![],
        vec![2, 0, 1],
        Some(vec![true, true, false]),
    ));
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut cases = vec![];
    let mut gates = 0;
    let mut bytes = 0;
    for (name, circuit, parameters, inputs, wires, expected_readouts) in circuits {
        let export = OpenQasm3::export(&circuit, &parameters, &inputs, &wires, Default::default())?;
        let parameters_tensor =
            (!parameters.is_empty()).then(|| Tensor::from_floats(parameters.as_slice(), &device));
        let inputs_tensor = (!inputs.is_empty()).then(|| {
            Tensor::<1>::from_floats(inputs.as_slice(), &device).reshape([1, inputs.len()])
        });
        let state = engine.execute(&circuit, parameters_tensor, inputs_tensor, 1, &device)?;
        state.validate(3e-6)?;
        ensure(
            export.numerical_error() < 1e-7,
            "QASM numerical synthesis budget",
        )?;
        gates += export.emitted_gates();
        bytes += export.program().len();
        cases.push(Case {
            name,
            circuit,
            parameters,
            inputs,
            export,
            expected_readouts,
        });
    }
    Ok(Run {
        report: Report::new(
            "psiqdk_interop",
            0,
            0,
            &[
                ("cases", cases.len() as f64),
                ("qasm_bytes", bytes as f64),
                ("emitted_gates", gates as f64),
            ],
        )?,
        global_phase_policy: "Equivalent up to one global phase per complete circuit; not a controlled-subroutine export",
        cases,
    })
}
fn main() -> ExampleResult<()> {
    println!("{}", serde_json::to_string_pretty(&run()?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn bounded_sdk_interchange_examples_execute_natively() {
        super::run().unwrap();
    }
}
