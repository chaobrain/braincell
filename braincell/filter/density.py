# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Reusable preference densities for continuous morphology sampling."""

from dataclasses import dataclass
import warnings

import brainunit as u
import numpy as np

__all__ = ["exponential", "gaussian", "spatial_gaussian"]


def _dimensionless_mantissa(value: object, *, name: str) -> np.ndarray:
    if isinstance(value, u.Quantity):
        if not u.get_unit(value).is_unitless:
            raise TypeError(f"{name} must be dimensionless after normalization.")
        return np.asarray(u.get_mantissa(value), dtype=float)
    return np.asarray(value, dtype=float)


def _positive_scale(value: object, *, name: str) -> object:
    if isinstance(value, u.Quantity):
        mantissa = np.asarray(u.get_mantissa(value), dtype=float)
    else:
        mantissa = np.asarray(value, dtype=float)
    if mantissa.ndim != 0 or not np.isfinite(mantissa).all() or float(mantissa) <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return value


def _field(context: object, name: str) -> object:
    # `name` is always a frozen dataclass's own `field`, and exponential() /
    # gaussian() -- the only two construction sites -- already applied the
    # non-empty-string check before storing it.
    try:
        return getattr(context, name)
    except AttributeError as exc:
        raise ValueError(f"Spatial context has no density field {name!r}.") from exc


@dataclass(frozen=True)
class _ExponentialDensity:
    field: str
    scale: object
    sign: float

    def _log_density(self, context: object) -> np.ndarray:
        value = _field(context, self.field)
        return self.sign * _dimensionless_mantissa(value / self.scale, name="exponential exponent")

    def __call__(self, context: object) -> np.ndarray:
        return np.exp(np.clip(self._log_density(context), -700.0, 700.0))


@dataclass(frozen=True)
class _GaussianDensity:
    field: str
    center: object
    sigma: object

    def _log_density(self, context: object) -> np.ndarray:
        value = _field(context, self.field)
        z = _dimensionless_mantissa((value - self.center) / self.sigma, name="Gaussian argument")
        return -0.5 * z * z

    def __call__(self, context: object) -> np.ndarray:
        return np.exp(self._log_density(context))


@dataclass(frozen=True)
class _SpatialGaussianDensity:
    center: object
    sigma: object

    def _log_density(self, context: object) -> np.ndarray:
        position = getattr(context, "position")
        delta = (position - self.center) / self.sigma
        values = _dimensionless_mantissa(delta, name="spatial Gaussian argument")
        if values.shape[-1:] != (3,):
            raise ValueError("Spatial context position must end in a three-dimensional coordinate axis.")
        return -0.5 * np.sum(values * values, axis=-1)

    def __call__(self, context: object) -> np.ndarray:
        return np.exp(self._log_density(context))


def exponential(
    field: str,
    scale: object,
    direction: str = "increasing",
) -> object:
    """Build an exponential preference density over a context field.

    Parameters
    ----------
    field : str
        Name of a numeric :class:`~braincell.filter.SamplingContext` field.
    scale : scalar or Quantity
        Positive scale with units compatible with the selected field.
    direction : {'increasing', 'decreasing'}, optional
        Whether preference grows or decays as the field increases.

    Returns
    -------
    callable
        A callable accepting one sampling context.
    """
    warnings.warn(
        "density.exponential(field, ...) is deprecated; write a callable using braincell.filter.metric instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(field, str) or not field:
        raise TypeError("field must be a non-empty string.")
    _positive_scale(scale, name="scale")
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'.")
    return _ExponentialDensity(field, scale, 1.0 if direction == "increasing" else -1.0)


def gaussian(field: str, center: object, sigma: object) -> object:
    """Build a Gaussian preference density over a context field.

    Parameters
    ----------
    field : str
        Name of a numeric :class:`~braincell.filter.SamplingContext` field.
    center : scalar or Quantity
        Center in units compatible with the selected field.
    sigma : scalar or Quantity
        Finite positive width in units compatible with the selected field.

    Returns
    -------
    callable
        A callable accepting one sampling context.
    """
    warnings.warn(
        "density.gaussian(field, ...) is deprecated; write a callable using braincell.filter.metric instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(field, str) or not field:
        raise TypeError("field must be a non-empty string.")
    _positive_scale(sigma, name="sigma")
    return _GaussianDensity(field, center, sigma)


def spatial_gaussian(center: object, sigma: object) -> object:
    """Build an isotropic Gaussian density over 3-D morphology positions.

    Parameters
    ----------
    center : Quantity
        Three-dimensional center coordinate.
    sigma : Quantity
        Finite positive spatial width.

    Returns
    -------
    callable
        A callable accepting one sampling context. Evaluation requires full
        point geometry on the morphology.
    """
    if not isinstance(center, u.Quantity):
        raise TypeError("center must be a three-dimensional quantity.")
    center_values = np.asarray(u.get_mantissa(center), dtype=float)
    if center_values.shape != (3,) or not np.all(np.isfinite(center_values)):
        raise ValueError("center must be a finite three-dimensional coordinate.")
    if not isinstance(sigma, u.Quantity):
        raise TypeError("sigma must be a length quantity.")
    _positive_scale(sigma, name="sigma")
    _dimensionless_mantissa(center / sigma, name="spatial Gaussian center")
    return _SpatialGaussianDensity(center, sigma)
