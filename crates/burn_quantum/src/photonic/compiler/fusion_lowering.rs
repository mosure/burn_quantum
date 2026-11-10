//! A real, bounded resource/fusion lowering for heralded single-qubit gate teleportation.
use crate::photonic::{
    CompileOptions, DualRail, Fabric, FusionBranch, FusionOutcome, NodeId, NodeKind, Parity,
    RailTransmission, ResourceState, Route, StandardFusion,
};
use crate::{
    ComplexTensor, DenseSimulator, DensityMatrix, Error, Matrix, Result, StateVector, Unitary,
};
use num_complex::Complex64;
use serde::Serialize;

/// Ports for an injected input and a supplied two-photon Choi resource.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TeleportationPorts {
    /// Source at which the external input photon is available at tick zero.
    pub input: NodeId,
    /// Source supplying the ideal (I tensor U)|Phi+> resource at tick zero.
    pub resource: NodeId,
    /// Destructive standard fusion analyzer.
    pub analyzer: NodeId,
}
/// Compiled heralded gate teleportation with explicit input/resource provenance.
///
/// This injects arbitrary encoded density inputs, including entanglement with
/// reference sites. The resource is ideal and the local correction is ideal
/// with zero controller latency; no resource-factory success is implied.
/// All five physical fusion outcomes remain observable. In the lossless model,
/// successful branches implement the requested gate. With attenuation they retain
/// the physical noisy channel, including output vacuum and asymmetric filtering;
/// a successful herald does not guarantee an undamaged logical output.
#[derive(Clone, Debug, Serialize)]
pub struct HeraldedGatePlan {
    gate: Unitary,
    fabric: Fabric,
    ports: TeleportationPorts,
    options: CompileOptions,
    routes: [Route; 2],
    transmissions: [RailTransmission; 3],
    finish: u64,
}
impl HeraldedGatePlan {
    /// Route an injected input and one half of a Choi resource to the analyzer.
    /// The other resource photon remains at its source as the teleported output.
    pub fn compile(
        gate: Unitary,
        fabric: &Fabric,
        ports: TeleportationPorts,
        options: CompileOptions,
    ) -> Result<Self> {
        DualRail::unitary(&gate)?;
        for (resource, required, limit) in [
            ("teleportation photons", 3, options.max_photons),
            ("teleportation fusions", 1, options.max_fusions),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        if fabric.node(ports.input)?.kind != NodeKind::Source
            || fabric.node(ports.resource)?.kind != NodeKind::Source
            || fabric.node(ports.analyzer)?.kind != NodeKind::Fusion
        {
            return Err(Error::InvalidValue(
                "teleportation requires source/source/analyzer ports".into(),
            ));
        }
        let mut reservations = vec![];
        let first = fabric.route_reserved(
            ports.input,
            ports.analyzer,
            0,
            options.max_ticks,
            &mut reservations,
        )?;
        let second = fabric.route_reserved(
            ports.resource,
            ports.analyzer,
            0,
            options.max_ticks,
            &mut reservations,
        )?;
        let tick = first.arrival.max(second.arrival);
        let finish = tick
            .checked_add(1)
            .filter(|t| *t <= options.max_ticks)
            .ok_or_else(|| Error::Unsupported("teleportation exceeds scheduling horizon".into()))?;
        let at = fabric.node(ports.analyzer)?.storage;
        let transmissions = [
            first
                .transmission
                .then(at.over_ticks(tick - first.arrival))
                .then(options.detector_transmission),
            second
                .transmission
                .then(at.over_ticks(tick - second.arrival))
                .then(options.detector_transmission),
            fabric.node(ports.resource)?.storage.over_ticks(finish),
        ];
        Ok(Self {
            gate,
            fabric: fabric.clone(),
            ports,
            options,
            routes: [first, second],
            transmissions,
            finish,
        })
    }
    /// Requested logical one-qubit unitary, supplied through the Choi resource.
    pub fn gate(&self) -> &Unitary {
        &self.gate
    }
    /// Fabric snapshot retained for provenance.
    pub fn fabric(&self) -> &Fabric {
        &self.fabric
    }
    /// Input and ideal-resource source ports.
    pub fn ports(&self) -> TeleportationPorts {
        self.ports
    }
    /// Scheduled routes for the input and measured resource half, in that order.
    pub fn routes(&self) -> &[Route; 2] {
        &self.routes
    }
    /// Input, measured resource half and output resource half attenuation.
    pub fn transmissions(&self) -> &[RailTransmission; 3] {
        &self.transmissions
    }
    /// Completion tick, including analyzer occupancy.
    pub fn finish_tick(&self) -> u64 {
        self.finish
    }
    /// Execute the physical instrument, preserving input reference sites and order.
    ///
    /// `input_site` is replaced by the output photon in the returned register.
    /// Other input sites are untouched reference systems. Constant gate/resource
    /// parameters are host values; input tensor gradients are retained.
    pub fn execute(
        &self,
        engine: &DenseSimulator,
        input: &DensityMatrix,
        input_site: usize,
    ) -> Result<Vec<FusionBranch>> {
        DualRail::validate(input.register())?;
        input.register().local_size(&[input_site])?;
        let n = input.register().size();
        let total = n.checked_mul(9).ok_or(Error::Overflow)?;
        engine.limits.check_state(
            input.batch_size(),
            total.checked_mul(total).ok_or(Error::Overflow)?,
        )?;
        let wires = input.register().wires();
        let resource = ResourceState::bell_pair().circuit()?;
        let initial = StateVector::zero(
            DualRail::register(2)?,
            1,
            &input.entries().device(),
            input.entries().dtype(),
            engine.limits,
        )?;
        let state = resource.run_from(engine, initial, None, None)?;
        let resource = DensityMatrix::from_pure(&state, engine.limits)?.apply(
            engine,
            &[1],
            &DualRail::unitary(&self.gate)?,
        )?;
        let b = input.batch_size();
        let ar = input.entries().real().reshape([b, n, 1, n, 1]);
        let ai = input.entries().imag().reshape([b, n, 1, n, 1]);
        let br = resource.entries().real().reshape([1, 1, 9, 1, 9]);
        let bi = resource.entries().imag().reshape([1, 1, 9, 1, 9]);
        let entries = ComplexTensor::new(
            (ar.clone() * br.clone() - ai.clone() * bi.clone()).reshape([b, total, total]),
            (ar * bi + ai * br).reshape([b, total, total]),
        )?;
        let mut joint =
            DensityMatrix::from_entries(DualRail::register(wires + 2)?, entries, engine.limits)?;
        for (site, transmission) in [input_site, wires, wires + 1]
            .into_iter()
            .zip(self.transmissions)
        {
            if transmission != RailTransmission::IDEAL {
                joint = joint.channel(engine, &[site], &transmission.channel()?)?;
            }
        }
        let mut branches = StandardFusion::branches(engine, &joint, [input_site, wires])?;
        let order: Vec<_> = (0..wires)
            .map(|site| {
                if site == input_site {
                    wires - 1
                } else if site < input_site {
                    site
                } else {
                    site - 1
                }
            })
            .collect();
        for branch in &mut branches {
            let mut state = branch
                .state
                .take()
                .expect("teleportation retains its output photon");
            if let FusionOutcome::Success { xx } = branch.outcomes[0] {
                let correction = conjugated_correction(&self.gate, xx)?;
                state = state.apply(engine, &[wires - 1], &DualRail::unitary(&correction)?)?;
            }
            branch.state = Some(state.reduced(&order, engine.limits)?);
        }
        Ok(branches)
    }
}
fn conjugated_correction(gate: &Unitary, xx: Parity) -> Result<Unitary> {
    // Psi+ -> U X|psi>, Psi- -> U XZ|psi>. Y equals XZ up to global phase.
    let p = if xx == Parity::Even {
        Unitary::x()
    } else {
        Unitary::y()
    };
    let u = gate.matrix().entries();
    let mut c = vec![Complex64::new(0., 0.); 4];
    for row in 0..2 {
        for col in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    c[row * 2 + col] +=
                        u[row * 2 + a] * p.matrix().entries()[a * 2 + b] * u[col * 2 + b].conj();
                }
            }
        }
    }
    Unitary::new(Matrix::new(2, c)?)
}
