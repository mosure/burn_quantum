//! Single-photon dual-rail code space and a restricted logical circuit frontend.
use crate::{Circuit, Error, Matrix, Observable, Operation, Register, Result, Unitary};
use num_complex::Complex64;

/// A dual-rail site: levels 0=`|10>`, 1=`|01>`, 2=`|00>` (vacuum).
///
/// The two occupied levels represent one photon across two optical modes.
/// Multiphoton occupation, spectral labels and mode mixing between sites require
/// a Fock-space engine; ordinary qutrit gates are not automatically optical gates.
pub struct DualRail;
impl DualRail {
    /// The vacuum level; photon loss never becomes logical zero.
    pub const VACUUM: usize = 2;
    /// Build a register of encoded sites, rejecting overflow before allocation.
    pub fn register(sites: usize) -> Result<Register> {
        if sites >= usize::BITS as usize {
            return Err(Error::Overflow);
        }
        Register::new(vec![3; sites])
    }
    /// Validate an all-dual-rail register.
    pub fn validate(register: &Register) -> Result<()> {
        if register.dimensions().iter().any(|&d| d != 3) {
            return Err(Error::InvalidDimensions(
                "dual-rail sites require dimension three".into(),
            ));
        }
        Ok(())
    }
    /// Embed a one-qubit unitary as U on the code and identity on vacuum.
    pub fn unitary(gate: &Unitary) -> Result<Unitary> {
        if gate.matrix().dimension() != 2 {
            return Err(Error::Shape(
                "dual-rail embedding requires a one-qubit unitary".into(),
            ));
        }
        let u = gate.matrix().entries();
        let z = Complex64::new(0., 0.);
        Unitary::new(Matrix::new(
            3,
            vec![u[0], u[1], z, u[2], u[3], z, z, z, 1.0.into()],
        )?)
    }
    /// Logical Z with value zero on vacuum; this is an unconditional readout.
    pub fn z(site: usize) -> Observable {
        Observable::new(
            vec![site],
            Matrix::real(3, vec![1., 0., 0., 0., -1., 0., 0., 0., 0.]).unwrap(),
        )
        .unwrap()
    }
    /// Photon-survival projector, retaining its probability rather than postselecting.
    pub fn survival(site: usize) -> Observable {
        Observable::new(
            vec![site],
            Matrix::real(3, vec![1., 0., 0., 0., 1., 0., 0., 0., 0.]).unwrap(),
        )
        .unwrap()
    }
    /// Lower a qubit circuit containing only local unitaries/rotations to the code.
    ///
    /// Parameter/input indices and affine expressions are preserved. This is an
    /// encoding pass, not synthesis of an entangling FBQC network or an optical
    /// component decomposition. Unsupported instructions fail with their index.
    pub fn encode_circuit(logical: &Circuit) -> Result<Circuit> {
        if logical.register().dimensions().iter().any(|&d| d != 2) {
            return Err(Error::InvalidDimensions("frontend requires qubits".into()));
        }
        let mut encoded = Circuit::new(
            Self::register(logical.register().wires())?,
            logical.num_parameters(),
            logical.num_inputs(),
        );
        for (index, operation) in logical.operations().iter().enumerate() {
            match operation {
                Operation::Unitary { wires, gate } if wires.len() == 1 => {
                    encoded.unitary(wires.clone(), Self::unitary(gate)?)?;
                }
                Operation::Rotate { wire, rotation } => {
                    encoded.rotate(*wire, rotation.clone())?;
                }
                _ => {
                    return Err(Error::Unsupported(format!(
                        "photonic local frontend cannot lower instruction {index}; use an explicit resource/fusion program"
                    )));
                }
            }
        }
        Ok(encoded)
    }
}
