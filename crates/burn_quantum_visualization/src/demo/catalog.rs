//! Configurable algorithm catalog. Preflight allocates only the small circuit definition.
use super::*;
use burn_quantum::algorithms::{self, AlgorithmCircuit};
/// Portable settings. `size` and `variant` have experiment-specific labels; `depth` is QAOA layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DemoConfig {
    /// Qubits, input bits, precision bits, controls or independent transfers, as labeled by the catalog.
    pub size: usize,
    /// Variational ansatz layers (QAOA only).
    pub depth: usize,
    /// Discrete oracle/direction/error/branch choice.
    pub variant: usize,
}
impl DemoConfig {
    /// Modest but nontrivial portable defaults; larger sizes are opt-in and budget checked.
    pub fn for_demo(kind: DemoKind) -> Self {
        Self {
            size: match kind {
                DemoKind::Ghz => 6,
                DemoKind::Toffoli => 2,
                DemoKind::Teleportation => 1,
                DemoKind::BitFlip => 3,
                DemoKind::Qft | DemoKind::DeutschJozsa | DemoKind::Qpe => 4,
                DemoKind::Grover => 5,
                DemoKind::Qaoa => 6,
                _ => 1,
            },
            depth: 2,
            variant: 0,
        }
    }
    /// Inclusive selectable size range and increment. Actual retained-state limits apply as well.
    pub fn range(kind: DemoKind) -> (usize, usize, usize) {
        match kind {
            DemoKind::Ghz => (2, 16, 1),
            DemoKind::Teleportation => (1, 5, 1),
            DemoKind::Toffoli => (2, 15, 1),
            DemoKind::Qft => (2, 16, 1),
            DemoKind::DeutschJozsa | DemoKind::Qpe => (2, 15, 1),
            DemoKind::Grover => (2, 10, 1),
            DemoKind::Qaoa => (3, 14, 1),
            DemoKind::BitFlip => (3, 9, 2),
            _ => (1, 1, 1),
        }
    }
    /// Human-readable meaning of size.
    pub fn size_label(kind: DemoKind) -> &'static str {
        match kind {
            DemoKind::Teleportation => "transfers",
            DemoKind::Toffoli => "controls",
            DemoKind::DeutschJozsa => "input bits",
            DemoKind::Qpe => "precision bits",
            _ => "qubits",
        }
    }
    /// Number of useful discrete settings for this catalog entry.
    pub fn variants(kind: DemoKind) -> usize {
        match kind {
            DemoKind::Teleportation => 4,
            DemoKind::Toffoli
            | DemoKind::Qft
            | DemoKind::DeutschJozsa
            | DemoKind::Qpe
            | DemoKind::Grover => 2,
            DemoKind::BitFlip => 4,
            _ => 1,
        }
    }
    /// Label for the selected variant, independent of simulation output.
    pub fn variant_label(self, kind: DemoKind) -> String {
        match kind {
            DemoKind::Teleportation => format!("Branch {:02b} / transfer", 3 - self.variant.min(3)),
            DemoKind::Toffoli => if self.variant == 0 {
                "Controls |+>"
            } else {
                "Controls |1>"
            }
            .into(),
            DemoKind::Qft => if self.variant == 0 {
                "Forward QFT · input 5 mod N"
            } else {
                "Inverse QFT · input 5 mod N"
            }
            .into(),
            DemoKind::DeutschJozsa => if self.variant == 0 {
                "Balanced parity oracle"
            } else {
                "Constant-one oracle"
            }
            .into(),
            DemoKind::Qpe => if self.variant == 0 {
                "Eigenphase 3/8"
            } else {
                "Eigenphase 1/3"
            }
            .into(),
            DemoKind::Grover => if self.variant == 0 {
                "Near-optimal iterations"
            } else {
                "One extra iteration"
            }
            .into(),
            DemoKind::BitFlip => [
                "X error on q0",
                "X error on last qubit",
                "No injected error",
                "Two X errors (compare distance)",
            ][self.variant.min(3)]
            .into(),
            _ => "Default".into(),
        }
    }
    /// Validate settings without constructing a tensor or exponential state.
    pub fn validate(self, kind: DemoKind) -> Result<()> {
        let (min, max, step) = Self::range(kind);
        if self.size < min
            || self.size > max
            || !(self.size - min).is_multiple_of(step)
            || self.depth == 0
            || self.depth > 6
            || self.variant >= Self::variants(kind)
        {
            return Err(burn_quantum::Error::InvalidValue(
                "unsupported catalog configuration".into(),
            ));
        }
        Ok(())
    }
}
/// Exact dimensions and timeline storage estimate, excluding transient backend workspaces and JSON overhead.
#[derive(Clone, Debug, Serialize)]
pub struct Preview {
    /// Number of quantum wires.
    pub wires: usize,
    /// Pure-state amplitudes per frame.
    pub dimension: usize,
    /// Executable steps, including measurements.
    pub steps: usize,
    /// Approximate host numeric bytes for amplitudes, probabilities and local density/marginals.
    pub snapshot_bytes: usize,
}
fn algorithm(kind: DemoKind, config: DemoConfig) -> Result<Option<AlgorithmCircuit>> {
    config.validate(kind)?;
    let n = config.size;
    Ok(Some(match kind {
        DemoKind::Teleportation => algorithms::teleportation(n, 1.1, 0.7)?,
        DemoKind::Toffoli => algorithms::controlled_x_demo(n, config.variant == 0)?,
        DemoKind::Qft => algorithms::qft(n, 5 % (1 << n), config.variant == 1)?,
        DemoKind::DeutschJozsa => {
            algorithms::deutsch_jozsa(n, if config.variant == 0 { (1 << n) - 1 } else { 0 }, true)?
        }
        DemoKind::Qpe => algorithms::phase_estimation(
            n,
            if config.variant == 0 {
                3. / 8.
            } else {
                1. / 3.
            },
        )?,
        DemoKind::Grover => algorithms::grover(
            n,
            (1 << n) - 3,
            algorithms::grover_iterations(n)? + config.variant,
        )?,
        DemoKind::Qaoa => algorithms::qaoa_maxcut(
            n,
            &(0..n).map(|w| (w, (w + 1) % n)).collect::<Vec<_>>(),
            &vec![(0.7, 0.35); config.depth],
        )?,
        DemoKind::BitFlip => algorithms::bit_flip_code(
            n,
            &match config.variant {
                0 => vec![0],
                1 => vec![n - 1],
                2 => vec![],
                _ => vec![0, 1],
            },
            1.1,
            0.7,
        )?,
        DemoKind::Ghz => {
            let mut circuit = Circuit::new(Register::qubits(n)?, 0, 0);
            let mut labels = vec![];
            circuit.unitary(vec![0], Unitary::h())?;
            labels.push("H q0".into());
            for w in 1..n {
                circuit.push(Operation::ControlledX {
                    controls: vec![(w - 1, 1)],
                    target: w,
                })?;
                labels.push(format!("CX q{} → q{w}", w - 1));
            }
            AlgorithmCircuit {
                circuit,
                labels,
                parameters: vec![],
                protocol: None,
            }
        }
        DemoKind::MixedRegister => {
            let mut circuit = Circuit::new(Register::new(vec![2, 2, 2, 2, 3, 3])?, 0, 0);
            let mut labels = vec![];
            for (wires, gate, label) in [
                (vec![0], Unitary::h(), "H q0"),
                (vec![0, 1], Unitary::cnot(), "Bell pair q0,q1"),
                (vec![2], Unitary::h(), "Pure |+> q2"),
                (vec![3], Unitary::x(), "Pure |1> q3"),
                (vec![4], Unitary::fourier(3)?, "F3 q4"),
                (vec![4, 5], Unitary::sum(3)?, "Qutrit Bell pair q4,q5"),
            ] {
                circuit.unitary(wires, gate)?;
                labels.push(label.into());
            }
            AlgorithmCircuit {
                circuit,
                labels,
                parameters: vec![],
                protocol: None,
            }
        }
        _ => return Ok(None),
    }))
}
fn preview_algorithm(a: &AlgorithmCircuit, limits: TraceLimits) -> Result<Preview> {
    let steps = a
        .protocol
        .as_ref()
        .map(|p| p.instructions.len())
        .unwrap_or(a.circuit.operations().len());
    limits.check(a.circuit.register(), steps + 1)?;
    let dimension = a.circuit.register().size();
    let local: usize = a
        .circuit
        .register()
        .dimensions()
        .iter()
        .map(|d| 16 * d * d + 8 * d + 32)
        .sum();
    Ok(Preview {
        wires: a.circuit.register().wires(),
        dimension,
        steps,
        snapshot_bytes: (24 * dimension + local) * (steps + 1),
    })
}
/// Check an algorithm configuration before launching a simulation. None marks a fixed-size foundational demo.
pub fn preview(kind: DemoKind, config: DemoConfig, limits: TraceLimits) -> Result<Option<Preview>> {
    algorithm(kind, config)?
        .map(|a| preview_algorithm(&a, limits))
        .transpose()
}
/// Run a configurable experiment on the chosen Burn device; no device guessing or silent reduction in size.
pub async fn build_configured(
    kind: DemoKind,
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    let Some(a) = algorithm(kind, config)? else {
        return super::build_basic(kind, device, limits).await;
    };
    let info = preview_algorithm(&a, limits)?;
    let (trace, protocol) = if a.protocol.is_some() {
        let pair = 3 - config.variant;
        let outcomes = (0..config.size)
            .flat_map(|_| [(pair >> 1) & 1, pair & 1])
            .collect();
        let (trace, record) =
            crate::simulation::simulate_protocol(&a, outcomes, device, limits).await?;
        (trace, Some(record))
    } else {
        (
            simulate(&a.circuit, &a.parameters, &[], &a.labels, device, limits).await?,
            None,
        )
    };
    let readout = readout(kind, config, trace.frames.last().unwrap());
    let size_description = if kind == DemoKind::MixedRegister {
        "4 qubits + 2 qutrits".into()
    } else {
        format!("{} {}", config.size, DemoConfig::size_label(kind))
    };
    let document = VisualizationDocument {
        schema_version: 2,
        title: kind.title().into(),
        description: format!(
            "{}\n{} · {} steps · {} basis states. {}\n{}",
            kind.description(),
            size_description,
            info.steps,
            info.dimension,
            config.variant_label(kind),
            readout
        ),
        circuit: a.circuit,
        parameters: a.parameters,
        inputs: vec![],
        operation_labels: a.labels,
        trace,
        fabric: None,
        optimization: vec![],
        protocol,
    };
    document.validate_with(limits)?;
    Ok(document)
}
fn readout(kind: DemoKind, c: DemoConfig, frame: &TimelineFrame) -> String {
    let p = &frame.probabilities;
    match kind {
        DemoKind::Grover=>format!("Marked state {}: success {:.3}%.",(1<<c.size)-3,100.*p[(1<<c.size)-3]),
        DemoKind::DeutschJozsa=>format!("P(input register = 0) = {:.6}. Zero indicates balanced; one indicates constant.",p[0]+p[1]),
        DemoKind::Qpe=>{let (i,mass)=p.chunks(2).enumerate().map(|(i,v)|(i,v.iter().sum::<f64>())).max_by(|a,b|a.1.total_cmp(&b.1)).unwrap();format!("Peak estimate {i}/{} = {:.5}, probability {:.3}%.",1<<c.size,i as f64/(1<<c.size)as f64,mass*100.)},
        DemoKind::Qaoa=>{let edges=(0..c.size).map(|w|(w,(w+1)%c.size)).collect::<Vec<_>>();let value:f64=p.iter().enumerate().map(|(i,&prob)|prob*algorithms::cut_value(c.size,&edges,i).unwrap()as f64).sum();format!("Ring MaxCut expectation {value:.4} / {} edges; {} fixed-angle layers. This is an ansatz evaluation, not an optimized solution.",c.size,c.depth)},
        DemoKind::Teleportation=>"Frames follow the explicitly selected measurement branch. All outcomes are validated separately; branch choice is not a deterministic measurement claim.".into(),
        DemoKind::BitFlip=>format!("Decoded q0 Bloch vector {:?}; remaining wires retain syndrome information.",frame.subsystems[0].bloch.unwrap().map(|v|(v*1000.).round()/1000.)),
        _=>"Read joint probabilities in Focus; local Bloch vectors alone do not describe entanglement.".into(),
    }
}
