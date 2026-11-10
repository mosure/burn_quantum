//! Compile two small Bell resources into a lossy fabric and destructively fuse them.
//! Every attempt is counted; successful outputs are entangled up to known Pauli signs.
use burn::tensor::DType;
use burn_quantum::{photonic::*, *};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let ideal = RailTransmission::IDEAL;
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Switch,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: ideal,
            },
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(0.9)?,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(0.8)?,
            },
            FabricLink {
                from: NodeId(2),
                to: NodeId(3),
                ticks: 2,
                transmission: ideal,
            },
        ],
    )?;
    let mut program = FusionProgram::new();
    let alice = program.add_resource(ResourceState::bell_pair(), NodeId(0))?;
    let bob = program.add_resource(ResourceState::bell_pair(), NodeId(1))?;
    program.fuse(alice[1], bob[0], NodeId(3))?;
    let plan = program.compile(&fabric, CompileOptions::default())?;
    let result = DenseFusionSimulator::default().execute(&plan, 1, &device, DType::F32)?;
    let mut success = 0.;
    let mut failure = 0.;
    let mut loss = 0.;
    let mut fidelity_min = 1.0f64;
    for branch in result.branches() {
        let probability = values(branch.probability())?[0];
        match branch.outcomes()[0] {
            FusionOutcome::Success { xx } => {
                success += probability;
                let mut rho = branch
                    .conditional_state(1e-8)?
                    .ok_or("missing output state")?;
                let engine = DenseSimulator::default();
                // For Phi+ resources, successful outputs are Psi+ / Psi-.
                rho = rho.apply(&engine, &[1], &DualRail::unitary(&Unitary::x())?)?;
                if xx == Parity::Odd {
                    rho = rho.apply(&engine, &[1], &DualRail::unitary(&Unitary::z())?)?;
                }
                let re = values(rho.entries().real())?;
                // Independent overlap with (|00>+|11>)/sqrt(2), indices 0 and 4.
                let fidelity = (re[0] + re[4] + re[36] + re[40]) / 2.;
                fidelity_min = fidelity_min.min(fidelity);
            }
            FusionOutcome::Failure { .. } => failure += probability,
            FusionOutcome::Loss => loss += probability,
            _ => return Err("unsupported future analyzer outcome".into()),
        }
    }
    let shots = result.sample(4096, 23, 4096)?;
    let sampled_success = shots[0]
        .iter()
        .filter(|shot| {
            matches!(
                result.branches()[shot.branch].outcomes()[0],
                FusionOutcome::Success { .. }
            )
        })
        .count() as f64
        / 4096.;
    ensure(
        (success - 0.36).abs() < 2e-5
            && (failure - 0.36).abs() < 2e-5
            && (loss - 0.28).abs() < 2e-5,
        "fusion probabilities disagree with 0.5 eta_a eta_b",
    )?;
    ensure(
        fidelity_min > 1. - 2e-5,
        "conditional Bell pair lost its coherence",
    )?;
    ensure(
        (sampled_success - success).abs() < 0.025,
        "sampled success disagrees with absolute branch weights",
    )?;
    ensure(
        plan.routes()[0].route.arrival == 3
            && plan.routes()[1].route.arrival == 5
            && plan.finish_tick() == 6,
        "routing ignored the shared-link reservation",
    )?;
    Report::new(
        "fbqc_bell_swap",
        23,
        1,
        &[
            ("supplied_photons", 4.),
            ("fusion_count", 1.),
            ("finish_tick", plan.finish_tick() as f64),
            ("success_probability", success),
            ("failure_probability", failure),
            ("loss_probability", loss),
            ("total_probability", values(result.total_probability())?[0]),
            ("conditional_bell_fidelity", fidelity_min),
            ("sample_fusion_success", sampled_success),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn routes_loss_and_fuses_resources_with_correct_unconditional_statistics() {
        super::run().unwrap();
    }
}
