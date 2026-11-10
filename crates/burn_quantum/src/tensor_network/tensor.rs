use super::MpsLimits;
use crate::{ComplexTensor, Error, Matrix, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};
use nalgebra::DMatrix;
use num_complex::Complex64;

pub(super) fn data<const D: usize>(x: &ComplexTensor<D>) -> Result<Vec<Complex64>> {
    let re = crate::complex::read(x.real())?;
    let im = crate::complex::read(x.imag())?;
    Ok(re
        .into_iter()
        .zip(im)
        .map(|(re, im)| Complex64::new(re, im))
        .collect())
}
pub(super) fn from<const D: usize>(
    values: &[Complex64],
    shape: [usize; D],
    device: &Device,
    dtype: DType,
) -> Result<ComplexTensor<D>> {
    ComplexTensor::new(
        Tensor::from_data(
            TensorData::new(values.iter().map(|x| x.re).collect(), shape),
            (device, dtype),
        ),
        Tensor::from_data(
            TensorData::new(values.iter().map(|x| x.im).collect(), shape),
            (device, dtype),
        ),
    )
}
pub(super) fn reshape<const D: usize, const E: usize>(
    x: &ComplexTensor<D>,
    shape: [usize; E],
) -> ComplexTensor<E> {
    ComplexTensor {
        re: x.re.clone().reshape(shape),
        im: x.im.clone().reshape(shape),
    }
}
pub(super) fn mm(a: &ComplexTensor<2>, b: &ComplexTensor<2>) -> ComplexTensor<2> {
    let m = crate::precision::matmul2;
    ComplexTensor {
        re: m(a.re.clone(), b.re.clone()) - m(a.im.clone(), b.im.clone()),
        im: m(a.re.clone(), b.im.clone()) + m(a.im.clone(), b.re.clone()),
    }
}
pub(super) fn adjoint(a: &ComplexTensor<2>) -> ComplexTensor<2> {
    ComplexTensor {
        re: a.re.clone().transpose(),
        im: -a.im.clone().transpose(),
    }
}
pub(super) fn scale<const D: usize>(a: &ComplexTensor<D>, value: f64) -> ComplexTensor<D> {
    // Flatten in logical order before multiplying. Burn 0.22.1 CUDA fusion can
    // misindex a scalar multiply on a permuted [bond, physical, 1] center (the
    // Bell/T regression loses its entire complex |11> branch). This native
    // reshape makes that layout contiguous without readback or AD detachment.
    // Each intermediate retains the same element bound; the normalization and
    // truncation math are unchanged.
    let shape = a.dims();
    let elements = shape.iter().product::<usize>();
    ComplexTensor {
        re: a
            .re
            .clone()
            .reshape([elements])
            .mul_scalar(value)
            .reshape(shape),
        im: a
            .im
            .clone()
            .reshape([elements])
            .mul_scalar(value)
            .reshape(shape),
    }
}
pub(super) struct Svd {
    pub u: DMatrix<Complex64>,
    pub s: Vec<f64>,
    pub v: DMatrix<Complex64>,
}
pub(super) fn svd(x: &ComplexTensor<2>, limits: MpsLimits) -> Result<Svd> {
    let [m, n] = x.dims();
    limits.tensor(crate::error::checked_product([m, n])?)?;
    if x.re.is_require_grad() || x.im.is_require_grad() {
        return Err(Error::Unsupported(
            "MPS SVD has no reverse-mode derivative; use parameter shift".into(),
        ));
    }
    let values = data(x)?;
    if values
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(Error::InvalidValue("nonfinite MPS contraction".into()));
    }
    let a = DMatrix::from_row_slice(m, n, &values);
    let svd = nalgebra::linalg::SVD::try_new(a, true, true, 1e-14, limits.svd_iterations)
        .ok_or_else(|| Error::InvalidValue("MPS complex SVD did not converge".into()))?;
    Ok(Svd {
        u: svd.u.unwrap(),
        s: svd.singular_values.as_slice().to_vec(),
        v: svd.v_t.unwrap(),
    })
}
pub(super) fn rows(a: &DMatrix<Complex64>) -> Vec<Complex64> {
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| a[(i, j)]))
        .collect()
}
/// Exact canonical-center transfer. No rank truncation or renormalization.
pub(super) fn move_right(
    a: &ComplexTensor<3>,
    b: &ComplexTensor<3>,
    limits: MpsLimits,
) -> Result<(ComplexTensor<3>, ComplexTensor<3>)> {
    let [l, d, r] = a.dims();
    let [bl, e, t] = b.dims();
    if r != bl {
        return Err(Error::Shape("MPS bond mismatch".into()));
    }
    let s = svd(&reshape(a, [l * d, r]), limits)?;
    let k = s.s.len();
    let mut v = s.v;
    for i in 0..k {
        for j in 0..r {
            v[(i, j)] *= s.s[i];
        }
    }
    limits.tensor(crate::error::checked_product([k, e, t])?)?;
    let left = from(&rows(&s.u), [l, d, k], &a.device(), a.dtype())?;
    let c = from(&rows(&v), [k, r], &a.device(), a.dtype())?;
    Ok((left, reshape(&mm(&c, &reshape(b, [r, e * t])), [k, e, t])))
}
pub(super) fn move_left(
    a: &ComplexTensor<3>,
    b: &ComplexTensor<3>,
    limits: MpsLimits,
) -> Result<(ComplexTensor<3>, ComplexTensor<3>)> {
    let [l, d, r] = a.dims();
    let [bl, e, t] = b.dims();
    if r != bl {
        return Err(Error::Shape("MPS bond mismatch".into()));
    }
    let s = svd(&reshape(b, [r, e * t]), limits)?;
    let k = s.s.len();
    let mut u = s.u;
    for i in 0..r {
        for j in 0..k {
            u[(i, j)] *= s.s[j];
        }
    }
    limits.tensor(crate::error::checked_product([l, d, k])?)?;
    let right = from(&rows(&s.v), [k, e, t], &b.device(), b.dtype())?;
    let c = from(&rows(&u), [r, k], &b.device(), b.dtype())?;
    Ok((reshape(&mm(&reshape(a, [l * d, r]), &c), [l, d, k]), right))
}

/// One shared streaming product-observable contraction for local and remote MPS.
pub(super) fn environment_step(
    env: &ComplexTensor<2>,
    site: &ComplexTensor<3>,
    operator: Option<&Matrix>,
    limits: MpsLimits,
) -> Result<ComplexTensor<2>> {
    let [l, d, r] = site.dims();
    limits.tensor(crate::error::checked_product([r, r])?)?;
    if env.dims() != [l, l] || operator.is_some_and(|m| m.dimension() != d) {
        return Err(Error::Shape("MPS observable environment mismatch".into()));
    }
    let mut out = ComplexTensor {
        re: Tensor::zeros([r, r], (&site.device(), site.dtype())),
        im: Tensor::zeros([r, r], (&site.device(), site.dtype())),
    };
    for a in 0..d {
        let range = if operator.is_some() { 0..d } else { a..a + 1 };
        for b in range {
            let z: Complex64 = operator.map_or(1.0.into(), |m| m.entries()[a * d + b]);
            if z.norm_sqr() == 0. {
                continue;
            }
            let sa = ComplexTensor {
                re: site.re.clone().slice_dim(1, a..a + 1).reshape([l, r]),
                im: site.im.clone().slice_dim(1, a..a + 1).reshape([l, r]),
            };
            let sb = ComplexTensor {
                re: site.re.clone().slice_dim(1, b..b + 1).reshape([l, r]),
                im: site.im.clone().slice_dim(1, b..b + 1).reshape([l, r]),
            };
            let part = mm(&mm(&adjoint(&sa), env), &sb);
            out.re = out.re + part.re.clone().mul_scalar(z.re) - part.im.clone().mul_scalar(z.im);
            out.im = out.im + part.re.mul_scalar(z.im) + part.im.mul_scalar(z.re);
        }
    }
    Ok(out)
}
