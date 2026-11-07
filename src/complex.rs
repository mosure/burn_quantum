//! Complex numbers represented by a pair of ordinary differentiable tensors.
use crate::{Error, Result};
use burn::tensor::{DType, Device, Tensor};

/// Split-complex tensor with matching shape, dtype, and device.
#[derive(Clone, Debug)]
pub struct ComplexTensor<const D: usize> {
    pub(crate) re: Tensor<D>,
    pub(crate) im: Tensor<D>,
}

impl<const D: usize> ComplexTensor<D> {
    /// Validate metadata without reading values back from the device.
    pub fn new(re: Tensor<D>, im: Tensor<D>) -> Result<Self> {
        if re.dims() != im.dims() {
            return Err(Error::Shape("real/imaginary shapes differ".into()));
        }
        compatible(&re, &im)?;
        if !matches!(re.dtype(), DType::F32 | DType::F64) {
            return Err(Error::Unsupported(
                "quantum tensors require F32 or F64".into(),
            ));
        }
        Ok(Self { re, im })
    }
    /// Real component, retaining its computation graph.
    pub fn real(&self) -> Tensor<D> {
        self.re.clone()
    }
    /// Imaginary component, retaining its computation graph.
    pub fn imag(&self) -> Tensor<D> {
        self.im.clone()
    }
    /// Component dimensions.
    pub fn dims(&self) -> [usize; D] {
        self.re.dims()
    }
    /// Hardware device and tensor execution context.
    pub fn device(&self) -> Device {
        let real = self.re.device();
        if real.is_autodiff() {
            real
        } else {
            self.im.device()
        }
    }
    /// Component floating-point type.
    pub fn dtype(&self) -> DType {
        self.re.dtype()
    }
    /// Complex conjugation.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }
    /// Elementwise squared modulus.
    pub fn abs_squared(&self) -> Tensor<D> {
        self.re.clone().square() + self.im.clone().square()
    }
    /// Move both components to a new device.
    pub fn to_device(&self, device: &Device) -> Self {
        Self {
            re: self.re.clone().to_device(device),
            im: self.im.clone().to_device(device),
        }
    }
}

pub(crate) fn compatible<const D: usize, const E: usize>(
    a: &Tensor<D>,
    b: &Tensor<E>,
) -> Result<()> {
    let ad = a.device();
    let bd = b.device();
    if ad != bd || a.dtype() != b.dtype() {
        return Err(Error::TensorContext);
    }
    #[cfg(feature = "autodiff")]
    if ad.is_autodiff()
        && bd.is_autodiff()
        && ad.gradient_checkpointing_strategy() != bd.gradient_checkpointing_strategy()
    {
        return Err(Error::TensorContext);
    }
    Ok(())
}

pub(crate) fn read<const D: usize>(value: Tensor<D>) -> Result<Vec<f64>> {
    value
        .into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| Error::Readback(format!("{e:?}")))
}
