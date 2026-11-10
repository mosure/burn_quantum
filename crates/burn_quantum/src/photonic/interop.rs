//! Bounded local OpenQASM 3 interchange, qualified with PsiQDK Workbench.
//!
//! This module exports a bound qubit circuit for a fresh local SDK simulator.
//! It does not submit jobs, authenticate, or describe a photonic hardware fabric.
use super::compiler::{GateSynthesis, SynthesisOptions};
use crate::{Angle, Circuit, Error, Operation, Result, Unitary, gate::Axis};
use num_complex::Complex64 as C;
use serde::Serialize;

/// Explicit export and host-synthesis budgets.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OpenQasmOptions {
    /// Maximum logical wires. The index conversion also respects usize width.
    pub max_qubits: usize,
    /// Maximum output UTF-8 bytes, including initialization and terminal measurements.
    pub max_bytes: usize,
    /// Existing native synthesis bounds; emitted gates and Euler residual share its error budget.
    pub synthesis: SynthesisOptions,
}
impl Default for OpenQasmOptions {
    fn default() -> Self {
        Self {
            max_qubits: 32,
            max_bytes: 1 << 22,
            synthesis: Default::default(),
        }
    }
}
/// Inspection-only export and explicit wire/readout/global-phase conventions.
#[derive(Clone, Debug, Serialize)]
pub struct OpenQasm3 {
    program: String,
    qubits: usize,
    readout_wires: Vec<usize>,
    native_gates: usize,
    emitted_gates: usize,
    numerical_error: f64,
}
impl OpenQasm3 {
    /// Bind one parameter vector and one feature row, then synthesize and export.
    /// Original wire i maps to `q[i]`; Burn state indices are big-endian, whereas
    /// PsiQDK state indices use `q[i]` as bit i. Terminal readouts follow the supplied
    /// order, mapping `c[j]` to `readout_wires[j]`. Inputs start in |0> via explicit reset.
    /// Global phase may change, including structured empty-predicate phase flips.
    /// Reusing this program as a controlled subroutine therefore requires a separate
    /// phase-preserving contract. No unsupported gate is silently omitted.
    pub fn export(
        source: &Circuit,
        parameters: &[f64],
        inputs: &[f64],
        readout_wires: &[usize],
        options: OpenQasmOptions,
    ) -> Result<Self> {
        let qubits = source.register().wires();
        if options.max_qubits == 0
            || options.max_qubits > 64
            || options.max_bytes == 0
            || options.max_bytes > 1 << 26
        {
            return Err(Error::InvalidValue(
                "invalid OpenQASM export budgets".into(),
            ));
        }
        if source.register().dimensions().iter().any(|&d| d != 2) {
            return Err(Error::Unsupported(
                "OpenQASM interchange supports qubits only".into(),
            ));
        }
        if qubits > options.max_qubits {
            return Err(Error::ResourceLimit {
                resource: "OpenQASM wires",
                required: qubits,
                limit: options.max_qubits,
            });
        }
        if qubits >= usize::BITS as usize {
            return Err(Error::Unsupported(
                "OpenQASM index mapping exceeds usize width".into(),
            ));
        }
        if parameters.len() != source.num_parameters() || inputs.len() != source.num_inputs() {
            return Err(Error::Shape("OpenQASM binding width".into()));
        }
        if parameters.iter().chain(inputs).any(|x| !x.is_finite()) {
            return Err(Error::InvalidValue("nonfinite OpenQASM binding".into()));
        }
        let mut used = vec![false; qubits];
        for &q in readout_wires {
            if q >= qubits || used[q] {
                return Err(Error::Wires("invalid or repeated terminal readout".into()));
            }
            used[q] = true;
        }
        let mut synthesis_options = options.synthesis;
        synthesis_options.max_native_gates =
            synthesis_options.max_native_gates.min(options.max_bytes);
        let synthesized = GateSynthesis::compile(source, synthesis_options)?;
        let mut output = Self {
            program: String::new(),
            qubits,
            readout_wires: readout_wires.to_vec(),
            native_gates: synthesized.circuit().operations().len(),
            emitted_gates: 0,
            numerical_error: synthesized.numerical_error(),
        };
        output.line(
            "OPENQASM 3;\ninclude \"stdgates.inc\";\n// Global phase unspecified; source wire i maps to q[i].\n",
            options.max_bytes,
        )?;
        output.line(&format!("qubit[{qubits}] q;\n"), options.max_bytes)?;
        if !readout_wires.is_empty() {
            output.line(
                &format!("bit[{}] c;\n", readout_wires.len()),
                options.max_bytes,
            )?;
        }
        for q in 0..qubits {
            output.line(&format!("reset q[{q}];\n"), options.max_bytes)?;
        }
        for operation in synthesized.circuit().operations() {
            match operation {
                Operation::Rotate { wire, rotation } => {
                    let mut value = match rotation.angle {
                        Angle::Constant(x) => x,
                        Angle::Parameter {
                            index,
                            scale,
                            offset,
                        } => scale * parameters[index] + offset,
                        Angle::Input {
                            index,
                            scale,
                            offset,
                        } => scale * inputs[index] + offset,
                    };
                    if !value.is_finite() {
                        return Err(Error::InvalidValue("OpenQASM affine angle overflow".into()));
                    }
                    if rotation.levels == [1, 0] && rotation.axis != Axis::X {
                        value = -value;
                    }
                    let name = match rotation.axis {
                        Axis::X => "rx",
                        Axis::Y => "ry",
                        Axis::Z => "rz",
                    };
                    output.rotation(name, *wire, value, options.max_bytes)?;
                }
                Operation::Unitary { wires, gate } if wires.len() == 1 => {
                    let (a, b, c, residual) = euler(gate);
                    // QASM statement order acts right-to-left on the state.
                    output.rotation("rz", wires[0], c, options.max_bytes)?;
                    output.rotation("ry", wires[0], b, options.max_bytes)?;
                    output.rotation("rz", wires[0], a, options.max_bytes)?;
                    output.numerical_error += residual;
                }
                Operation::Unitary { wires, gate } if wires.len() == 2 => {
                    let name = if *gate == Unitary::cnot() {
                        "cx"
                    } else if *gate == Unitary::swap(2)? {
                        "swap"
                    } else if *gate == Unitary::z().controlled(2, 1)? {
                        "cz"
                    } else {
                        return Err(Error::Unsupported(
                            "non-native synthesized entangler".into(),
                        ));
                    };
                    output.line(
                        &format!("{name} q[{}], q[{}];\n", wires[0], wires[1]),
                        options.max_bytes,
                    )?;
                    output.emitted_gates += 1;
                }
                _ => {
                    return Err(Error::Unsupported(
                        "unsynthesized OpenQASM instruction".into(),
                    ));
                }
            }
        }
        output.numerical_error += output.emitted_gates as f64 * 64. * f64::EPSILON;
        if !output.numerical_error.is_finite()
            || output.numerical_error > options.synthesis.max_numerical_error
        {
            return Err(Error::InvalidValue(
                "OpenQASM reconstruction exceeds numerical budget".into(),
            ));
        }
        for (c, &q) in readout_wires.iter().enumerate() {
            output.line(&format!("c[{c}] = measure q[{q}];\n"), options.max_bytes)?;
        }
        Ok(output)
    }
    /// Exact exported text. Global phase is intentionally unspecified.
    pub fn program(&self) -> &str {
        &self.program
    }
    /// Number of logical qubits, without synthesized ancillas.
    pub fn qubits(&self) -> usize {
        self.qubits
    }
    /// Original source wire for each terminal classical bit, in measurement order.
    pub fn readout_wires(&self) -> &[usize] {
        &self.readout_wires
    }
    /// Gate count before Euler decomposition (resets/readouts excluded).
    pub fn native_gates(&self) -> usize {
        self.native_gates
    }
    /// Number of emitted primitive quantum gates (resets/readouts excluded).
    pub fn emitted_gates(&self) -> usize {
        self.emitted_gates
    }
    /// Host reconstruction diagnostic, distinct from SDK/backend precision error.
    pub fn numerical_error(&self) -> f64 {
        self.numerical_error
    }
    /// Convert a Burn big-endian amplitude index into PsiQDK's little-endian index.
    /// Wire identities themselves remain unchanged.
    pub fn psiqdk_index(&self, burn_index: usize) -> Result<usize> {
        if burn_index >= 1usize << self.qubits {
            return Err(Error::Basis);
        }
        Ok(burn_index.reverse_bits() >> (usize::BITS as usize - self.qubits))
    }
    /// Decode Workbench's weighted measurement values (0 or 2^physical_wire)
    /// into ordinary booleans in the declared terminal measurement order.
    pub fn decode_psiqdk_readouts(&self, weighted: &[u64]) -> Result<Vec<bool>> {
        if weighted.len() != self.readout_wires.len() {
            return Err(Error::Shape("PsiQDK readout width".into()));
        }
        weighted
            .iter()
            .zip(&self.readout_wires)
            .map(|(&value, &q)| {
                if value == 0 {
                    Ok(false)
                } else if value == 1u64 << q {
                    Ok(true)
                } else {
                    Err(Error::InvalidValue(
                        "PsiQDK weighted readout does not match wire".into(),
                    ))
                }
            })
            .collect()
    }
    fn rotation(&mut self, name: &str, wire: usize, angle: f64, limit: usize) -> Result<()> {
        self.line(&format!("{name}({angle:.17e}) q[{wire}];\n"), limit)?;
        self.emitted_gates += 1;
        Ok(())
    }
    fn line(&mut self, line: &str, limit: usize) -> Result<()> {
        let required = self
            .program
            .len()
            .checked_add(line.len())
            .ok_or(Error::Overflow)?;
        if required > limit {
            return Err(Error::ResourceLimit {
                resource: "OpenQASM bytes",
                required,
                limit,
            });
        }
        self.program.push_str(line);
        Ok(())
    }
}
fn euler(gate: &Unitary) -> (f64, f64, f64, f64) {
    let u = gate.matrix().entries();
    let co = u[0].norm();
    let si = u[2].norm();
    let (a, b, c) = if si < 1e-14 {
        (u[3].arg() - u[0].arg(), 0., 0.)
    } else if co < 1e-14 {
        (u[2].arg() - (-u[1]).arg(), std::f64::consts::PI, 0.)
    } else {
        (
            u[2].arg() - u[0].arg(),
            2. * si.atan2(co),
            (-u[1]).arg() - u[0].arg(),
        )
    };
    let (co, si) = ((b / 2.).cos(), (b / 2.).sin());
    let reconstruction = [
        C::from_polar(co, -(a + c) / 2.),
        -C::from_polar(si, -(a - c) / 2.),
        C::from_polar(si, (a - c) / 2.),
        C::from_polar(co, (a + c) / 2.),
    ];
    let overlap: C = reconstruction
        .iter()
        .zip(u)
        .map(|(x, y)| x.conj() * y)
        .sum();
    let phase = C::from_polar(1., overlap.arg());
    let residual = reconstruction
        .iter()
        .zip(u)
        .map(|(x, y)| (*x * phase - *y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    (a, b, c, residual)
}
