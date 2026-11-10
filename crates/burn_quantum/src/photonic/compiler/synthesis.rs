//! Bounded algebraically exact synthesis into local U(2) gates and CNOT.
use crate::{Circuit, Error, Matrix, Operation, Result, Unitary};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::f64::consts::PI;

/// Explicit host synthesis work and numerical reconstruction budgets.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SynthesisOptions {
    /// Maximum expanded native gate count, checked while building each instruction.
    pub max_native_gates: usize,
    /// Largest dense opaque unitary accepted by QR synthesis (at most four qubits).
    pub max_unitary_qubits: usize,
    /// Largest structured controlled predicate, including an X target when present.
    pub max_predicate_qubits: usize,
    /// Maximum accumulated Frobenius reconstruction residual plus roundoff allowance.
    /// This is a host synthesis diagnostic; backend execution error is separate.
    pub max_numerical_error: f64,
}
impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_native_gates: 32768,
            max_unitary_qubits: 4,
            max_predicate_qubits: 12,
            max_numerical_error: 1e-7,
        }
    }
}
/// Native circuit with preserved affine bindings and an explicit numerical error account.
///
/// QR/two-level elimination and phase-polynomial predicates introduce no intended
/// approximation. Global phase is immaterial and may change. The reconstruction
/// diagnostic includes a conservative floating-point allowance, not interval arithmetic
/// certification or a bound on subsequent noisy/finite-precision quantum execution.
#[derive(Clone, Debug, Serialize)]
pub struct GateSynthesis {
    circuit: Circuit,
    numerical_error: f64,
}
impl GateSynthesis {
    /// Synthesize arbitrary bounded dense qubit unitaries and structured multicontrols.
    pub fn compile(source: &Circuit, options: SynthesisOptions) -> Result<Self> {
        if options.max_native_gates == 0
            || options.max_unitary_qubits == 0
            || options.max_unitary_qubits > 4
            || options.max_predicate_qubits == 0
            || options.max_predicate_qubits > 20
            || !options.max_numerical_error.is_finite()
            || options.max_numerical_error <= 0.
        {
            return Err(Error::InvalidValue(
                "invalid native synthesis budgets".into(),
            ));
        }
        if source.register().dimensions().iter().any(|&d| d != 2) {
            return Err(Error::Unsupported(
                "native photonic synthesis requires qubits".into(),
            ));
        }
        let mut builder = Builder {
            ops: vec![],
            limit: options.max_native_gates,
        };
        let mut error = 0.;
        for (instruction, op) in source.operations().iter().enumerate() {
            let begin = builder.ops.len();
            match op {
                Operation::Rotate { .. } => builder.push(op.clone())?,
                Operation::Unitary { wires, gate }
                    if wires.len() == 1 || native_entangler(gate, wires.len())? =>
                {
                    builder.push(op.clone())?
                }
                Operation::Unitary { wires, gate } => {
                    if wires.len() > options.max_unitary_qubits {
                        return Err(Error::Unsupported(format!(
                            "instruction {instruction}: dense unitary exceeds {}-qubit synthesis budget",
                            options.max_unitary_qubits
                        )));
                    }
                    builder.qr(wires, gate)?;
                    error += reconstruction(&builder.ops[begin..], wires, gate)?;
                }
                Operation::ControlledX { controls, target } => {
                    check_predicate(controls.len() + 1, options)?;
                    builder.polarity(controls)?;
                    let controls: Vec<_> = controls.iter().map(|c| c.0).collect();
                    builder.mcx(&controls, *target)?;
                    builder.polarity(match op {
                        Operation::ControlledX { controls, .. } => controls,
                        _ => unreachable!(),
                    })?;
                }
                Operation::PhaseFlip { controls } => {
                    check_predicate(controls.len(), options)?;
                    builder.polarity(controls)?;
                    builder.phase(&controls.iter().map(|c| c.0).collect::<Vec<_>>(), PI)?;
                    builder.polarity(controls)?;
                }
            }
            // Finite arithmetic in constant gate construction, independent of tensor dtype.
            error += (builder.ops.len() - begin) as f64 * 64. * f64::EPSILON;
            if error > options.max_numerical_error {
                return Err(Error::InvalidValue(format!(
                    "instruction {instruction}: synthesis numerical error {error:e} exceeds budget"
                )));
            }
        }
        let mut circuit = Circuit::new(
            source.register().clone(),
            source.num_parameters(),
            source.num_inputs(),
        );
        for op in builder.ops {
            circuit.push(op)?;
        }
        Ok(Self {
            circuit,
            numerical_error: error,
        })
    }
    /// Executable local/CNOT/CZ/SWAP circuit; parameter and input IDs are unchanged.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// Accumulated host reconstruction residual and floating-point allowance.
    pub fn numerical_error(&self) -> f64 {
        self.numerical_error
    }
}
fn check_predicate(n: usize, options: SynthesisOptions) -> Result<()> {
    if n > options.max_predicate_qubits {
        return Err(Error::ResourceLimit {
            resource: "synthesis predicate qubits",
            required: n,
            limit: options.max_predicate_qubits,
        });
    }
    Ok(())
}
fn native_entangler(gate: &Unitary, n: usize) -> Result<bool> {
    Ok(n == 2
        && [
            Unitary::cnot(),
            Unitary::swap(2)?,
            Unitary::z().controlled(2, 1)?,
        ]
        .contains(gate))
}
struct Builder {
    ops: Vec<Operation>,
    limit: usize,
}
impl Builder {
    fn push(&mut self, op: Operation) -> Result<()> {
        if self.ops.len() >= self.limit {
            return Err(Error::ResourceLimit {
                resource: "synthesized native gates",
                required: self.ops.len() + 1,
                limit: self.limit,
            });
        }
        self.ops.push(op);
        Ok(())
    }
    fn local(&mut self, wire: usize, gate: Unitary) -> Result<()> {
        self.push(Operation::Unitary {
            wires: vec![wire],
            gate,
        })
    }
    fn cx(&mut self, a: usize, b: usize) -> Result<()> {
        self.push(Operation::Unitary {
            wires: vec![a, b],
            gate: Unitary::cnot(),
        })
    }
    fn polarity(&mut self, controls: &[(usize, usize)]) -> Result<()> {
        for &(wire, level) in controls {
            if level == 0 {
                self.local(wire, Unitary::x())?;
            }
        }
        Ok(())
    }
    // exp(i angle |11...1><11...1|), ignoring its physically irrelevant global phase.
    fn phase(&mut self, wires: &[usize], angle: f64) -> Result<()> {
        if wires.is_empty() || angle == 0. {
            return Ok(());
        }
        if wires.len() == 1 {
            return self.local(wires[0], Unitary::phase(angle)?);
        }
        let count = 1usize << wires.len();
        for mask in 1..count {
            let subset: Vec<_> = wires
                .iter()
                .enumerate()
                .filter_map(|(i, &w)| (mask & (1 << i) != 0).then_some(w))
                .collect();
            let (&target, controls) = subset.split_last().unwrap();
            for &control in controls {
                self.cx(control, target)?;
            }
            let sign = if subset.len() % 2 == 0 { -1. } else { 1. };
            self.local(target, rz(2. * sign * angle / count as f64)?)?;
            for &control in controls.iter().rev() {
                self.cx(control, target)?;
            }
        }
        Ok(())
    }
    fn mcx(&mut self, controls: &[usize], target: usize) -> Result<()> {
        match controls {
            [] => self.local(target, Unitary::x()),
            [control] => self.cx(*control, target),
            _ => {
                self.local(target, Unitary::h())?;
                let mut all = controls.to_vec();
                all.push(target);
                self.phase(&all, PI)?;
                self.local(target, Unitary::h())
            }
        }
    }
    // ABC=I and AXBXC=exp(-i alpha) U (Barenco et al.); the predicate phase
    // restores alpha. Every controlled X is itself synthesized into the native basis.
    fn controlled(
        &mut self,
        controls: &[(usize, usize)],
        target: usize,
        u: &Unitary,
    ) -> Result<()> {
        if controls.is_empty() {
            return self.local(target, u.clone());
        }
        self.polarity(controls)?;
        let ids: Vec<_> = controls.iter().map(|c| c.0).collect();
        let (a, b, c, phase) = euler(u)?;
        self.local(target, rz((c - a) / 2.)?)?;
        self.mcx(&ids, target)?;
        self.local(target, multiply(&ry(-b / 2.)?, &rz(-(a + c) / 2.)?)?)?;
        self.mcx(&ids, target)?;
        self.local(target, multiply(&rz(a)?, &ry(b / 2.)?)?)?;
        self.phase(&ids, phase)?;
        self.polarity(controls)
    }
    fn adjacent(&mut self, wires: &[usize], p: usize, q: usize, gate: &Unitary) -> Result<()> {
        let diff = p ^ q;
        debug_assert_eq!(diff.count_ones(), 1);
        let target = wires.len() - 1 - diff.trailing_zeros() as usize;
        let controls: Vec<_> = wires
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != target)
            .map(|(i, &w)| (w, (p >> (wires.len() - 1 - i)) & 1))
            .collect();
        let gate = if p & diff == 0 {
            gate.clone()
        } else {
            let e = gate.matrix().entries();
            Unitary::new(Matrix::new(2, vec![e[3], e[2], e[1], e[0]])?)?
        };
        if gate == Unitary::x() {
            self.polarity(&controls)?;
            self.mcx(
                &controls.iter().map(|x| x.0).collect::<Vec<_>>(),
                wires[target],
            )?;
            self.polarity(&controls)
        } else {
            self.controlled(&controls, wires[target], &gate)
        }
    }
    fn two_level(&mut self, wires: &[usize], p: usize, q: usize, gate: &Unitary) -> Result<()> {
        let mut path = vec![p];
        let mut current = p;
        while (current ^ q).count_ones() > 1 {
            current ^= 1usize << (current ^ q).trailing_zeros();
            path.push(current);
        }
        for pair in path.windows(2) {
            self.adjacent(wires, pair[0], pair[1], &Unitary::x())?;
        }
        self.adjacent(wires, current, q, gate)?;
        for pair in path.windows(2).rev() {
            self.adjacent(wires, pair[0], pair[1], &Unitary::x())?;
        }
        Ok(())
    }
    fn qr(&mut self, wires: &[usize], gate: &Unitary) -> Result<()> {
        let d = gate.matrix().dimension();
        let mut r = gate.matrix().entries().to_vec();
        let mut inverse = vec![];
        for col in 0..d {
            for row in (col + 1..d).rev() {
                let a = r[col * d + col];
                let b = r[row * d + col];
                if b.norm() == 0. {
                    continue;
                }
                let norm = a.norm().hypot(b.norm());
                let g = Unitary::new(Matrix::new(
                    2,
                    vec![a.conj() / norm, b.conj() / norm, -b / norm, a / norm],
                )?)?;
                let e = g.matrix().entries();
                for j in 0..d {
                    let x = r[col * d + j];
                    let y = r[row * d + j];
                    r[col * d + j] = e[0] * x + e[1] * y;
                    r[row * d + j] = e[2] * x + e[3] * y;
                }
                inverse.push((col, row, g.adjoint()));
            }
        }
        for index in 0..d {
            let controls: Vec<_> = wires
                .iter()
                .enumerate()
                .map(|(i, &w)| (w, (index >> (wires.len() - 1 - i)) & 1))
                .collect();
            self.polarity(&controls)?;
            self.phase(wires, r[index * d + index].arg())?;
            self.polarity(&controls)?;
        }
        for (p, q, g) in inverse.into_iter().rev() {
            self.two_level(wires, p, q, &g)?;
        }
        Ok(())
    }
}
fn rz(a: f64) -> Result<Unitary> {
    Unitary::new(Matrix::new(
        2,
        vec![
            C::from_polar(1., -a / 2.),
            0.0.into(),
            0.0.into(),
            C::from_polar(1., a / 2.),
        ],
    )?)
}
fn ry(a: f64) -> Result<Unitary> {
    let (c, s) = ((a / 2.).cos(), (a / 2.).sin());
    Unitary::new(Matrix::real(2, vec![c, -s, s, c])?)
}
fn multiply(a: &Unitary, b: &Unitary) -> Result<Unitary> {
    let a = a.matrix().entries();
    let b = b.matrix().entries();
    Unitary::new(Matrix::new(
        2,
        (0..4)
            .map(|i| (0..2).map(|k| a[(i / 2) * 2 + k] * b[k * 2 + i % 2]).sum())
            .collect(),
    )?)
}
fn euler(u: &Unitary) -> Result<(f64, f64, f64, f64)> {
    let e = u.matrix().entries();
    let co = e[0].norm();
    let si = e[2].norm();
    let (a, b, c) = if si < 1e-14 {
        (e[3].arg() - e[0].arg(), 0., 0.)
    } else if co < 1e-14 {
        (e[2].arg() - (-e[1]).arg(), PI, 0.)
    } else {
        (
            e[2].arg() - e[0].arg(),
            2. * si.atan2(co),
            (-e[1]).arg() - e[0].arg(),
        )
    };
    let r = multiply(&multiply(&rz(a)?, &ry(b)?)?, &rz(c)?)?;
    let overlap: C = r
        .matrix()
        .entries()
        .iter()
        .zip(e)
        .map(|(x, y)| x.conj() * y)
        .sum();
    Ok((a, b, c, overlap.arg()))
}
fn reconstruction(ops: &[Operation], wires: &[usize], desired: &Unitary) -> Result<f64> {
    let d = 1usize << wires.len();
    let mut out = vec![C::new(0., 0.); d * d];
    for i in 0..d {
        out[i * d + i] = 1.0.into();
    }
    for op in ops {
        let Operation::Unitary {
            wires: targets,
            gate,
        } = op
        else {
            unreachable!()
        };
        let positions: Vec<_> = targets
            .iter()
            .map(|w| wires.iter().position(|q| q == w).unwrap())
            .collect();
        let local = 1usize << positions.len();
        let mut next = vec![C::new(0., 0.); d * d];
        for row in 0..d {
            let mut r = 0;
            for &p in &positions {
                r = 2 * r + ((row >> (wires.len() - 1 - p)) & 1);
            }
            for s in 0..local {
                let mut source = row;
                for (j, &p) in positions.iter().enumerate() {
                    let bit = 1 << (wires.len() - 1 - p);
                    source = (source & !bit) | (((s >> (positions.len() - 1 - j)) & 1) * bit);
                }
                for col in 0..d {
                    next[row * d + col] +=
                        gate.matrix().entries()[r * local + s] * out[source * d + col];
                }
            }
        }
        out = next;
    }
    let desired = desired.matrix().entries();
    let overlap: C = out.iter().zip(desired).map(|(a, b)| a.conj() * b).sum();
    let phase = C::from_polar(1., overlap.arg());
    Ok(out
        .iter()
        .zip(desired)
        .map(|(a, b)| (*a * phase - *b).norm_sqr())
        .sum::<f64>()
        .sqrt())
}
