use super::*;
use crate::simulation::TraceLimits;
const EPS: f64 = 5e-4;
fn invalid() -> Error {
    Error::InvalidValue("inconsistent visualization document".into())
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= EPS
}
fn finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}
fn transmission(values: [f64; 2]) -> bool {
    values
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}
impl VisualizationDocument {
    /// Validate an imported document against the interactive defaults before displaying it.
    /// This checks scientific consistency, not provenance or whether a claimed experiment was run.
    /// Without global amplitudes, local positivity and probability marginals are checked,
    /// but joint mixed-state coherences and global compatibility cannot be reconstructed.
    pub fn validate(&self) -> Result<()> {
        self.validate_with(TraceLimits::default())
    }
    /// Validate bounds, finite values, state identities and physical schedule references.
    pub fn validate_with(&self, limits: TraceLimits) -> Result<()> {
        let reg = self.circuit.register();
        limits.check(reg, self.trace.frames.len())?;
        if ![1, 2].contains(&self.schema_version)
            || self.trace.frames.is_empty()
            || self.title.len() > 4096
            || self.description.len() > 32768
            || self.trace.backend.len() > 4096
            || self.circuit.operations().len() + 1 > limits.max_frames
            || self.parameters.len() != self.circuit.num_parameters()
            || self.inputs.len() != self.circuit.num_inputs()
            || self.parameters.len() > 64
            || self.inputs.len() > 256
            || !finite(&self.parameters)
            || !finite(&self.inputs)
            || self.operation_labels.len() != self.circuit.operations().len()
            || self.operation_labels.iter().any(|s| s.len() > 4096)
        {
            return Err(invalid());
        }
        for op in self.circuit.operations() {
            if let burn_quantum::Operation::Unitary { gate, .. } = op
                && gate.matrix().entries().len() > limits.max_matrix_elements
            {
                return Err(invalid());
            }
        }
        match self.trace.kind {
            TraceKind::Circuit
                if self.trace.frames.len() != self.circuit.operations().len() + 1
                    || !self.optimization.is_empty() =>
            {
                return Err(invalid());
            }
            TraceKind::Optimization if self.optimization.len() != self.trace.frames.len() => {
                return Err(invalid());
            }
            _ => {}
        }
        if let Some(record) = &self.protocol {
            use burn_quantum::protocol::Instruction;
            let bits = record.program.validate(&self.circuit)?;
            if self.schema_version != 2
                || self.trace.kind != TraceKind::Protocol
                || !self.optimization.is_empty()
                || record.outcomes.len() != bits
                || record.outcomes.iter().any(|v| *v > 1)
                || record.program.instructions.len() + 1 != self.trace.frames.len()
                || record.probabilities.len() != self.trace.frames.len()
                || !close(record.probabilities[0], 1.)
            {
                return Err(invalid());
            }
            for (index, instruction) in record.program.instructions.iter().enumerate() {
                let previous = record.probabilities[index];
                let next = record.probabilities[index + 1];
                if !next.is_finite() || next <= 0. || next > 1. + EPS {
                    return Err(invalid());
                }
                let expected = match instruction {
                    Instruction::Measure { wire, bit } => {
                        let outcome = record.outcomes[*bit];
                        let before = self.trace.frames[index]
                            .subsystems
                            .get(*wire)
                            .ok_or_else(invalid)?;
                        let after = self.trace.frames[index + 1]
                            .subsystems
                            .get(*wire)
                            .ok_or_else(invalid)?;
                        if before.probabilities.len() != 2
                            || after.probabilities.len() != 2
                            || !close(after.probabilities[outcome], 1.)
                        {
                            return Err(invalid());
                        }
                        previous * before.probabilities[outcome]
                    }
                    _ => previous,
                };
                if !close(next, expected) {
                    return Err(invalid());
                }
            }
        } else if self.trace.kind == TraceKind::Protocol {
            return Err(invalid());
        }
        for (i, frame) in self.trace.frames.iter().enumerate() {
            if frame.step != i
                || frame.label.len() > 4096
                || frame.probabilities.len() != reg.size()
                || !finite(&frame.probabilities)
                || frame.probabilities.iter().any(|v| *v < 0.)
                || !close(frame.norm, 1.)
                || !close(frame.norm, frame.probabilities.iter().sum())
                || frame.subsystems.len() != reg.wires()
            {
                return Err(invalid());
            }
            if let Some(amplitudes) = &frame.amplitudes
                && (amplitudes.len() != reg.size()
                    || amplitudes
                        .iter()
                        .zip(&frame.probabilities)
                        .any(|(a, &p)| !close(a.re * a.re + a.im * a.im, p)))
            {
                return Err(invalid());
            }
            for (wire, subsystem) in frame.subsystems.iter().enumerate() {
                let d = reg.dimensions()[wire];
                if subsystem.wire != wire
                    || subsystem.dimension != d
                    || subsystem.probabilities.len() != d
                    || subsystem.density.len() != d * d
                {
                    return Err(invalid());
                }
                density(subsystem)?;
                let stride: usize = reg.dimensions()[wire + 1..].iter().product();
                let mut marginal = vec![0.; d];
                for (index, &p) in frame.probabilities.iter().enumerate() {
                    marginal[(index / stride) % d] += p;
                }
                if marginal
                    .iter()
                    .zip(&subsystem.probabilities)
                    .any(|(&a, &b)| !close(a, b))
                {
                    return Err(invalid());
                }
                if let Some(amplitudes) = &frame.amplitudes {
                    let mut reduced = vec![ComplexValue::default(); d * d];
                    for base in (0..reg.size()).filter(|i| (i / stride).is_multiple_of(d)) {
                        for r in 0..d {
                            for c in 0..d {
                                let a = amplitudes[base + r * stride];
                                let b = amplitudes[base + c * stride];
                                reduced[r * d + c].re += a.re * b.re + a.im * b.im;
                                reduced[r * d + c].im += a.im * b.re - a.re * b.im;
                            }
                        }
                    }
                    if reduced
                        .iter()
                        .zip(&subsystem.density)
                        .any(|(a, b)| !close(a.re, b.re) || !close(a.im, b.im))
                    {
                        return Err(invalid());
                    }
                }
            }
        }
        for (i, p) in self.optimization.iter().enumerate() {
            let frame = &self.trace.frames[i];
            if p.iteration != i
                || p.parameters.len() != self.parameters.len()
                || p.metric.len() != p.parameters.len() * p.parameters.len()
                || !finite(&p.parameters)
                || !finite(&p.metric)
                || !p.loss.is_finite()
                || p.probabilities != frame.probabilities
                || p.bloch != frame.subsystems.iter().map(|s| s.bloch).collect::<Vec<_>>()
                || (i > 0
                    && (p.loss_evaluations < self.optimization[i - 1].loss_evaluations
                        || p.fidelity_evaluations < self.optimization[i - 1].fidelity_evaluations))
            {
                return Err(invalid());
            }
        }
        if self
            .optimization
            .last()
            .is_some_and(|p| p.parameters != self.parameters)
        {
            return Err(invalid());
        }
        if let Some(fabric) = &self.fabric {
            fabric.validate()?;
        }
        Ok(())
    }
}
fn density(s: &SubsystemState) -> Result<()> {
    let d = s.dimension;
    let a = &s.density;
    if a.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        || !close((0..d).map(|i| a[i * d + i].re).sum(), 1.)
    {
        return Err(invalid());
    }
    for i in 0..d {
        for j in 0..d {
            if !close(a[i * d + j].re, a[j * d + i].re) || !close(a[i * d + j].im, -a[j * d + i].im)
            {
                return Err(invalid());
            }
        }
        if !close(s.probabilities[i], a[i * d + i].re) || s.probabilities[i] < -EPS {
            return Err(invalid());
        }
    }
    // Regularized Cholesky checks positivity at the declared display tolerance.
    let mut lower = vec![ComplexValue::default(); d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut value = a[i * d + j];
            for k in 0..j {
                let x = lower[i * d + k];
                let y = lower[j * d + k];
                value.re -= x.re * y.re + x.im * y.im;
                value.im -= x.im * y.re - x.re * y.im;
            }
            if i == j {
                let pivot = value.re + EPS;
                if !pivot.is_finite() || pivot <= 0. {
                    return Err(invalid());
                }
                lower[i * d + j].re = pivot.sqrt();
            } else {
                lower[i * d + j] = ComplexValue {
                    re: value.re / lower[j * d + j].re,
                    im: value.im / lower[j * d + j].re,
                };
            }
        }
    }
    let purity = a.iter().map(|v| v.re * v.re + v.im * v.im).sum();
    if !close(s.purity, purity) {
        return Err(invalid());
    }
    match (d, s.bloch) {
        (2, Some(b)) => {
            let expected = [2. * a[1].re, -2. * a[1].im, a[0].re - a[3].re];
            if b.iter().zip(expected).any(|(&x, y)| !close(x, y))
                || b.iter().map(|x| x * x).sum::<f64>() > 1. + EPS
            {
                return Err(invalid());
            }
        }
        (2, None) => return Err(invalid()),
        (_, Some(_)) => return Err(invalid()),
        _ => {}
    }
    Ok(())
}
impl FabricSnapshot {
    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty()
            || self.nodes.len() > 256
            || self.links.len() > 1024
            || self.routes.len() > 4096
            || self.fusions.len() > 2048
            || self.outputs.len() > 8192
            || self.photon_count > 8192
            || self.resource_count > 4096
            || self.finish_tick > 1_000_000
        {
            return Err(invalid());
        }
        let total_hops = self.routes.iter().try_fold(0usize, |n, route| {
            n.checked_add(route.hops.len()).ok_or(Error::Overflow)
        })?;
        if total_hops > 32768 {
            return Err(Error::ResourceLimit {
                resource: "visualization route hops",
                required: total_hops,
                limit: 32768,
            });
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if n.id != i
                || n.label.len() > 4096
                || n.kind.len() > 128
                || n.position.iter().any(|v| !v.is_finite() || v.abs() > 1e6)
                || !transmission(n.storage)
            {
                return Err(invalid());
            }
        }
        for (i, e) in self.links.iter().enumerate() {
            if e.id != i
                || e.from >= self.nodes.len()
                || e.to >= self.nodes.len()
                || e.from == e.to
                || e.ticks == 0
                || !transmission(e.transmission)
            {
                return Err(invalid());
            }
        }
        if !self.resources.is_empty() {
            if self.resources.len() != self.resource_count {
                return Err(invalid());
            }
            let mut owned = vec![false; self.photon_count];
            for r in &self.resources {
                if r.source >= self.nodes.len()
                    || r.photons.is_empty()
                    || r.photons.len() > 64
                    || r.edges.len() > 4096
                    || r.hadamards.len() > r.photons.len()
                {
                    return Err(invalid());
                }
                for &photon in &r.photons {
                    if photon >= owned.len() || owned[photon] {
                        return Err(invalid());
                    }
                    owned[photon] = true;
                }
                if r.edges.iter().enumerate().any(|(i, &(a, b))| {
                    a >= r.photons.len()
                        || b >= r.photons.len()
                        || a >= b
                        || r.edges[..i].contains(&(a, b))
                }) || r
                    .hadamards
                    .iter()
                    .enumerate()
                    .any(|(i, &w)| w >= r.photons.len() || r.hadamards[..i].contains(&w))
                {
                    return Err(invalid());
                }
                for route in self
                    .routes
                    .iter()
                    .filter(|route| r.photons.contains(&route.photon))
                {
                    if route.from != r.source {
                        return Err(invalid());
                    }
                }
            }
            if owned.iter().any(|v| !*v) {
                return Err(invalid());
            }
        }
        let mut consumed = vec![false; self.photon_count];
        let mut routed = vec![false; self.photon_count];
        for r in &self.routes {
            if r.photon >= self.photon_count
                || routed[r.photon]
                || r.from >= self.nodes.len()
                || r.to >= self.nodes.len()
                || r.release > r.arrival
                || r.arrival > self.finish_tick
                || r.hops.len() > 4096
                || !transmission(r.transmission)
            {
                return Err(invalid());
            }
            routed[r.photon] = true;
            let mut node = r.from;
            let mut time = r.release;
            let mut expected_transmission = [1.0; 2];
            for h in &r.hops {
                let Some(link) = self.links.get(h.link) else {
                    return Err(invalid());
                };
                if link.from != node
                    || h.departure < time
                    || h.arrival.checked_sub(h.departure) != Some(link.ticks)
                {
                    return Err(invalid());
                }
                for (rail, value) in expected_transmission.iter_mut().enumerate() {
                    *value *= self.nodes[node].storage[rail].powf((h.departure - time) as f64)
                        * link.transmission[rail];
                }
                node = link.to;
                time = h.arrival;
            }
            if node != r.to
                || time != r.arrival
                || expected_transmission
                    .iter()
                    .zip(r.transmission)
                    .any(|(&actual, supplied)| !close(actual, supplied))
            {
                return Err(invalid());
            }
        }
        for f in &self.fusions {
            if f.node >= self.nodes.len()
                || f.tick >= self.finish_tick
                || f.photons[0] == f.photons[1]
            {
                return Err(invalid());
            }
            for &p in &f.photons {
                if p >= self.photon_count || consumed[p] || !routed[p] {
                    return Err(invalid());
                }
                consumed[p] = true;
                let r = self.routes.iter().find(|r| r.photon == p).unwrap();
                if r.to != f.node || r.arrival > f.tick {
                    return Err(invalid());
                }
            }
        }
        if routed != consumed {
            return Err(invalid());
        }
        for &p in &self.outputs {
            if p >= self.photon_count || consumed[p] {
                return Err(invalid());
            }
            consumed[p] = true;
        }
        if consumed.iter().any(|p| !*p) {
            return Err(invalid());
        }
        Ok(())
    }
}
