use super::*;
use crate::{Error, Operation, Result, Unitary, gate::Axis};
use std::f64::consts::{FRAC_PI_2, PI};

struct Builder {
    pattern: MeasurementPattern,
    limits: PatternCompileLimits,
}
impl Builder {
    fn push(&mut self, command: PatternCommand) -> Result<()> {
        let required = self
            .pattern
            .commands
            .len()
            .checked_add(1)
            .ok_or(Error::Overflow)?;
        if required > self.limits.max_commands {
            return Err(Error::ResourceLimit {
                resource: "pattern commands",
                required,
                limit: self.limits.max_commands,
            });
        }
        self.pattern.commands.push(command);
        Ok(())
    }
    fn j(&mut self, wire: usize, angle: Angle) -> Result<()> {
        let required = self.pattern.sites.checked_add(1).ok_or(Error::Overflow)?;
        if required > self.limits.max_sites {
            return Err(Error::ResourceLimit {
                resource: "pattern sites",
                required,
                limit: self.limits.max_sites,
            });
        }
        let old = self.pattern.outputs[wire];
        let new = PatternSite(self.pattern.sites);
        let bit = self.pattern.measurements;
        self.push(PatternCommand::PreparePlus { site: new })?;
        self.push(PatternCommand::Entangle { sites: [old, new] })?;
        self.push(PatternCommand::Measure {
            site: old,
            angle,
            bit,
        })?;
        self.push(PatternCommand::CorrectX { site: new, bit })?;
        self.pattern.sites = required;
        self.pattern.measurements += 1;
        self.pattern.outputs[wire] = new;
        Ok(())
    }
    fn rz(&mut self, wire: usize, angle: Angle) -> Result<()> {
        self.j(wire, angle)?;
        self.j(wire, Angle::Constant(0.))
    }
    fn rx(&mut self, wire: usize, angle: Angle) -> Result<()> {
        self.j(wire, Angle::Constant(0.))?;
        self.j(wire, angle)
    }
    fn ry(&mut self, wire: usize, angle: Angle) -> Result<()> {
        self.rz(wire, Angle::Constant(-FRAC_PI_2))?;
        self.rx(wire, angle)?;
        self.rz(wire, Angle::Constant(FRAC_PI_2))
    }
    fn cz(&mut self, a: usize, b: usize) -> Result<()> {
        self.push(PatternCommand::Entangle {
            sites: [self.pattern.outputs[a], self.pattern.outputs[b]],
        })
    }
    fn cnot(&mut self, a: usize, b: usize) -> Result<()> {
        self.j(b, Angle::Constant(0.))?;
        self.cz(a, b)?;
        self.j(b, Angle::Constant(0.))
    }
    fn local(&mut self, wire: usize, gate: &Unitary) -> Result<()> {
        if gate == &Unitary::h() {
            return self.j(wire, Angle::Constant(0.));
        }
        if gate == &Unitary::x() {
            return self.rx(wire, Angle::Constant(PI));
        }
        if gate == &Unitary::y() {
            return self.ry(wire, Angle::Constant(PI));
        }
        if gate == &Unitary::z() {
            return self.rz(wire, Angle::Constant(PI));
        }
        let u = gate.matrix().entries();
        // ZYZ Euler decomposition, with singular diagonal/antidiagonal cases.
        let cosine = u[0].norm();
        let sine = u[2].norm();
        let (a, b, c) = if sine < 1e-14 {
            (u[3].arg() - u[0].arg(), 0., 0.)
        } else if cosine < 1e-14 {
            (u[2].arg() - (-u[1]).arg(), PI, 0.)
        } else {
            (
                u[2].arg() - u[0].arg(),
                2. * sine.atan2(cosine),
                (-u[1]).arg() - u[0].arg(),
            )
        };
        if c != 0. {
            self.rz(wire, Angle::Constant(c))?;
        }
        if b != 0. {
            self.ry(wire, Angle::Constant(b))?;
        }
        if a != 0. {
            self.rz(wire, Angle::Constant(a))?;
        }
        Ok(())
    }
}
fn negate(angle: &Angle) -> Angle {
    match angle {
        Angle::Constant(x) => Angle::Constant(-x),
        Angle::Parameter {
            index,
            scale,
            offset,
        } => Angle::Parameter {
            index: *index,
            scale: -*scale,
            offset: -*offset,
        },
        Angle::Input {
            index,
            scale,
            offset,
        } => Angle::Input {
            index: *index,
            scale: -*scale,
            offset: -*offset,
        },
    }
}
pub(super) fn compile(
    source: &Circuit,
    limits: PatternCompileLimits,
) -> Result<MeasurementPattern> {
    if source.register().dimensions().iter().any(|&d| d != 2) {
        return Err(Error::Unsupported(
            "measurement compiler requires qubits".into(),
        ));
    }
    let n = source.register().wires();
    if n > limits.max_sites {
        return Err(Error::ResourceLimit {
            resource: "pattern sites",
            required: n,
            limit: limits.max_sites,
        });
    }
    let mut builder = Builder {
        pattern: MeasurementPattern {
            version: 1,
            source: source.clone(),
            commands: vec![],
            outputs: (0..n).map(PatternSite).collect(),
            sites: n,
            measurements: 0,
            synthesis_error: 0.,
        },
        limits,
    };
    for (index, op) in source.operations().iter().enumerate() {
        match op {
            Operation::Unitary { wires, gate } if wires.len() == 1 => {
                builder.local(wires[0], gate)?
            }
            Operation::Unitary { wires, gate } if wires.len() == 2 && gate == &Unitary::cnot() => {
                builder.cnot(wires[0], wires[1])?
            }
            Operation::Unitary { wires, gate }
                if wires.len() == 2 && gate == &Unitary::z().controlled(2, 1)? =>
            {
                builder.cz(wires[0], wires[1])?
            }
            Operation::Unitary { wires, gate }
                if wires.len() == 2 && gate == &Unitary::swap(2)? =>
            {
                builder.cnot(wires[0], wires[1])?;
                builder.cnot(wires[1], wires[0])?;
                builder.cnot(wires[0], wires[1])?;
            }
            Operation::Rotate { wire, rotation } => {
                let angle = if rotation.levels == [1, 0] && rotation.axis != Axis::X {
                    negate(&rotation.angle)
                } else {
                    rotation.angle.clone()
                };
                match rotation.axis {
                    Axis::X => builder.rx(*wire, angle)?,
                    Axis::Y => builder.ry(*wire, angle)?,
                    Axis::Z => builder.rz(*wire, angle)?,
                }
            }
            Operation::ControlledX { controls, target } if controls.is_empty() => {
                builder.rx(*target, Angle::Constant(PI))?
            }
            Operation::ControlledX { controls, target } if controls.len() == 1 => {
                let (control, level) = controls[0];
                if level == 0 {
                    builder.rx(control, Angle::Constant(PI))?;
                }
                builder.cnot(control, *target)?;
                if level == 0 {
                    builder.rx(control, Angle::Constant(PI))?;
                }
            }
            Operation::PhaseFlip { controls } if controls.len() <= 2 => {
                for &(wire, level) in controls {
                    if level == 0 {
                        builder.rx(wire, Angle::Constant(PI))?;
                    }
                }
                match controls.as_slice() {
                    [] => {} // A global minus sign does not change the quantum channel.
                    [(wire, _)] => builder.rz(*wire, Angle::Constant(PI))?,
                    [(a, _), (b, _)] => builder.cz(*a, *b)?,
                    _ => unreachable!(),
                }
                for &(wire, level) in controls {
                    if level == 0 {
                        builder.rx(wire, Angle::Constant(PI))?;
                    }
                }
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "measurement compiler cannot lower instruction {index}"
                )));
            }
        }
    }
    Ok(builder.pattern)
}
