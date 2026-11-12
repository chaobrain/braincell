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

import brainunit as u
import numpy as np

from braincell._deprecation import removed_attributes

__all__ = ["spatial_gaussian"]

#: The two field helpers braincell 0.3 removed without a deprecation period
#: (API-15), with what to write instead; raised as ``AttributeError``
#: (``ImportError`` on a ``from braincell.filter.density import ...`` line).
__getattr__ = removed_attributes(
    __name__,
    {
        "exponential": (
            "braincell.filter.density.exponential() was removed in braincell 0.3. Write the density as a "
            "callable built from braincell.filter.metric: exponential('branch_x', 0.2) is "
            "lambda context: np.exp(metric.branch_x(context) / 0.2), and direction='decreasing' negates the exponent."
        ),
        "gaussian": (
            "braincell.filter.density.gaussian() was removed in braincell 0.3. Write the density as a "
            "callable built from braincell.filter.metric: gaussian('branch_x', 0.5, 0.1) is "
            "lambda context: np.exp(-0.5 * ((metric.branch_x(context) - 0.5) / 0.1) ** 2)."
        ),
    },
)


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
