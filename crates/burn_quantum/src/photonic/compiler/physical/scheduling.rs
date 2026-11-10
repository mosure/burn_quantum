//! Checked native gate/resource placement, finite calendars and wavelength routes.
use super::*;
impl PhysicalCircuit {
    /// Compile resource births, routes, destructive fusions and feedback delays.
    pub fn compile(
        source: &Circuit,
        fabric: &Fabric,
        placement: PhysicalPlacement,
        options: CompileOptions,
    ) -> Result<Self> {
        Self::compile_impl(source, fabric, placement, options, None, None, None)
    }
    /// Compile against finite packet buffers and switch configuration times.
    /// `fusion_ticks` contains one causal consumption tick per synthesized fusion.
    /// Fixed timing makes resource feasibility auditable; an infeasible plan is rejected
    /// atomically rather than changing requested physical capacities or birth times.
    pub fn compile_finite(
        source: &Circuit,
        fabric: &Fabric,
        placement: PhysicalPlacement,
        options: CompileOptions,
        constraints: FabricConstraints,
        fusion_ticks: &[u64],
    ) -> Result<Self> {
        Self::compile_impl(
            source,
            fabric,
            placement,
            options,
            Some((constraints, fusion_ticks)),
            None,
            None,
        )
    }
    /// Compile with actual heralded generation windows and noisy optical/source models.
    /// With finite constraints, consumption ticks refer to the synthesized gate sequence.
    pub fn compile_calibrated(
        source: &Circuit,
        fabric: &Fabric,
        placement: PhysicalPlacement,
        options: CompileOptions,
        calibration: PhysicalCalibration,
        finite: Option<(FabricConstraints, &[u64])>,
    ) -> Result<Self> {
        Self::compile_impl(
            source,
            fabric,
            placement,
            options,
            finite,
            Some(calibration),
            None,
        )
    }
    /// Compile coherent independent rails with matched wavelengths, bounded skew,
    /// finite optical-mode capacity and a calibrated physical executor.
    pub fn compile_rails(
        source: &Circuit,
        fabric: &Fabric,
        placement: PhysicalPlacement,
        options: CompileOptions,
        calibration: PhysicalCalibration,
        rails: PhysicalRailOptions,
    ) -> Result<Self> {
        Self::compile_impl(
            source,
            fabric,
            placement,
            options,
            None,
            Some(calibration),
            Some(rails),
        )
    }
    fn compile_impl(
        source: &Circuit,
        fabric: &Fabric,
        placement: PhysicalPlacement,
        options: CompileOptions,
        finite: Option<(FabricConstraints, &[u64])>,
        calibration: Option<PhysicalCalibration>,
        rail_options: Option<PhysicalRailOptions>,
    ) -> Result<Self> {
        if source.register().dimensions().iter().any(|&d| d != 2) {
            return Err(Error::Unsupported(
                "physical compiler requires qubits".into(),
            ));
        }
        let n = source.register().wires();
        if placement.inputs.len() != n {
            return Err(Error::Shape(
                "physical input placement differs from logical register".into(),
            ));
        }
        for &port in placement
            .inputs
            .iter()
            .chain(std::iter::once(&placement.resource))
        {
            if fabric.node(port)?.kind != NodeKind::Source {
                return Err(Error::InvalidValue(
                    "physical input/resource port is not a source".into(),
                ));
            }
        }
        if fabric.node(placement.analyzer)?.kind != NodeKind::Fusion {
            return Err(Error::InvalidValue(
                "physical analyzer port is not a fusion node".into(),
            ));
        }
        let synthesized = GateSynthesis::compile(
            source,
            SynthesisOptions {
                max_native_gates: options.max_fusions,
                ..Default::default()
            },
        )?;
        let operations = synthesized.circuit().operations().to_vec();
        let fusions = operations.iter().map(|op| targets(op).len()).sum::<usize>();
        if finite
            .as_ref()
            .is_some_and(|(_, ticks)| ticks.len() != fusions)
        {
            return Err(Error::Shape(
                "one consumption tick per synthesized physical fusion is required".into(),
            ));
        }
        if rail_options
            .as_ref()
            .is_some_and(|r| r.fusion_ticks.len() != fusions)
        {
            return Err(Error::Shape(
                "one rail consumption tick per synthesized physical fusion is required".into(),
            ));
        }
        let supplied = if let Some(model) = &calibration {
            operations.iter().try_fold(0usize, |sum, op| {
                let k = targets(op).len();
                sum.checked_add(model.resource(k)?.factory().max_photons())
                    .and_then(|x| x.checked_add(k * model.analyzer().supplied_ancillas()))
                    .ok_or(Error::Overflow)
            })?
        } else {
            fusions.checked_mul(2).ok_or(Error::Overflow)?
        };
        let photons = n.checked_add(supplied).ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("physical fusions", fusions, options.max_fusions),
            ("physical photons", photons, options.max_photons),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        let mut locations = placement.inputs.clone();
        let mut births = vec![0u64; n];
        let mut physical_births = births.clone();
        let constraints = finite.as_ref().map(|(c, _)| c.clone());
        let mut packets = constraints
            .clone()
            .map(|c| PacketScheduler::new(fabric.clone(), c))
            .transpose()?;
        let mut fusion_index = 0;
        let mut rail_scheduler = rail_options
            .as_ref()
            .map(|r| RailPacketScheduler::new(fabric.clone(), r.constraints.clone()))
            .transpose()?;
        let mut cursor = 0u64;
        let mut gates = vec![];
        let mut reservations = vec![];
        for operation in operations {
            let wires = targets(&operation);
            if let Some(model) = &calibration {
                cursor = cursor
                    .checked_add(model.resource(wires.len())?.factory().worst_case_ticks())
                    .filter(|&t| t <= options.max_ticks)
                    .ok_or_else(|| {
                        Error::Unsupported("resource generation exceeds physical horizon".into())
                    })?;
            }
            let start = cursor;
            let mut input = vec![];
            let mut resource_measured = vec![];
            let mut routes = vec![];
            let mut rail_routes = vec![];
            for &wire in &wires {
                let (a, b, tick) = if let Some(scheduler) = &mut rail_scheduler {
                    let config = rail_options.as_ref().unwrap();
                    let tick = config.fusion_ticks[fusion_index];
                    if tick < cursor || tick >= options.max_ticks {
                        return Err(Error::InvalidValue(
                            "rail consumption outside causal horizon".into(),
                        ));
                    }
                    let [a, b] = scheduler.pair_at(
                        [
                            PacketRequest {
                                from: locations[wire],
                                to: placement.analyzer,
                                birth: physical_births[wire],
                                release: cursor,
                                consume: tick,
                            },
                            PacketRequest {
                                from: placement.resource,
                                to: placement.analyzer,
                                birth: start,
                                release: cursor,
                                consume: tick,
                            },
                        ],
                        config.wavelengths,
                    )?;
                    let packets = (a.packet.clone(), b.packet.clone(), tick);
                    rail_routes.extend([a, b]);
                    packets
                } else if let Some(scheduler) = &mut packets {
                    let tick = finite.as_ref().unwrap().1[fusion_index];
                    if tick < cursor || tick >= options.max_ticks {
                        return Err(Error::InvalidValue(
                            "physical consumption outside causal horizon".into(),
                        ));
                    }
                    let [a, b] = scheduler.pair_at([
                        PacketRequest {
                            from: locations[wire],
                            to: placement.analyzer,
                            birth: physical_births[wire],
                            release: cursor,
                            consume: tick,
                        },
                        PacketRequest {
                            from: placement.resource,
                            to: placement.analyzer,
                            birth: start,
                            release: cursor,
                            consume: tick,
                        },
                    ])?;
                    (a, b, tick)
                } else {
                    let a = fabric.route_reserved(
                        locations[wire],
                        placement.analyzer,
                        cursor,
                        options.max_ticks,
                        &mut reservations,
                    )?;
                    let b = fabric.route_reserved(
                        placement.resource,
                        placement.analyzer,
                        cursor,
                        options.max_ticks,
                        &mut reservations,
                    )?;
                    let tick = a.arrival.max(b.arrival);
                    (a, b, tick)
                };
                fusion_index += 1;
                let analyzer_storage = fabric.node(placement.analyzer)?.storage;
                input.push(
                    fabric
                        .node(locations[wire])?
                        .storage
                        .over_ticks(cursor - births[wire])
                        .then(a.transmission)
                        .then(analyzer_storage.over_ticks(tick - a.arrival))
                        .then(options.detector_transmission),
                );
                resource_measured.push(
                    fabric
                        .node(placement.resource)?
                        .storage
                        .over_ticks(cursor - start)
                        .then(b.transmission)
                        .then(analyzer_storage.over_ticks(tick - b.arrival))
                        .then(options.detector_transmission),
                );
                routes.extend([a, b]);
                cursor = tick
                    .checked_add(1)
                    .filter(|&t| t <= options.max_ticks)
                    .ok_or_else(|| Error::Unsupported("physical gate exceeds horizon".into()))?;
            }
            cursor = cursor
                .checked_add(placement.feedback_ticks)
                .filter(|&t| t <= options.max_ticks)
                .ok_or_else(|| Error::Unsupported("physical feedback exceeds horizon".into()))?;
            let resource_outputs = vec![
                fabric
                    .node(placement.resource)?
                    .storage
                    .over_ticks(cursor - start);
                wires.len()
            ];
            for &wire in &wires {
                locations[wire] = placement.resource;
                births[wire] = cursor;
                physical_births[wire] = start;
            }
            gates.push(PhysicalGate {
                operation,
                wires,
                input,
                resource_measured,
                resource_outputs,
                routes,
                rail_routes,
                start,
                finish: cursor,
            });
        }
        if let Some(packets) = &mut packets {
            for (&location, &birth) in locations.iter().zip(&physical_births) {
                packets.store_output(location, birth, cursor)?;
            }
        }
        if let Some(scheduler) = &mut rail_scheduler {
            for (&location, &birth) in locations.iter().zip(&physical_births) {
                scheduler.store_output(
                    location,
                    birth,
                    cursor,
                    rail_options.as_ref().unwrap().wavelengths,
                )?;
            }
        }
        let final_storage = locations
            .iter()
            .zip(births)
            .map(|(&port, birth)| Ok(fabric.node(port)?.storage.over_ticks(cursor - birth)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            version: 1,
            source: source.clone(),
            fabric: fabric.clone(),
            placement,
            options,
            gates,
            final_storage,
            synthesis_error: synthesized.numerical_error(),
            constraints,
            calibration,
            rail_options,
            finish: cursor,
        })
    }
}

fn targets(op: &Operation) -> Vec<usize> {
    match op {
        Operation::Unitary { wires, .. } => wires.clone(),
        Operation::Rotate { wire, .. } => vec![*wire],
        _ => unreachable!("normalized physical gate"),
    }
}
