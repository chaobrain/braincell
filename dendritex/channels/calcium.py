# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent calcium channels.

"""

from __future__ import annotations

from typing import Union, Callable, Optional

import brainstate as bst
import brainunit as bu

from .._base import Channel, IonInfo, State4Integral
from ..ions import Calcium

__all__ = [
  'CalciumChannel',

  'ICaN_IS2008',
  'ICaT_HM1992',
  'ICaT_HP1992',
  'ICaHT_HM1992',
  'ICaHT_Re1993',
  'ICaL_IS2008',
  "ICav12_Ma2020",
  "ICav13_Ma2020",
  "ICav23_Ma2020",
  "ICav31_Ma2020",
  'ICaGrc_Ma2020',
]


class CalciumChannel(Channel):
  """Base class for Calcium ion channels."""

  __module__ = 'dendritex.channels'

  root_type = Calcium

  def before_integral(self, V, Ca: IonInfo):
    pass

  def after_integral(self, V, Ca: IonInfo):
    pass

  def compute_derivative(self, V, Ca: IonInfo):
    pass

  def current(self, V, Ca: IonInfo):
    raise NotImplementedError

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    pass

  def reset_state(self, V, Ca: IonInfo, batch_size: int = None):
    pass


class ICaN_IS2008(CalciumChannel):
  r"""The calcium-activated non-selective cation channel model
  proposed by (Inoue & Strowbridge, 2008) [2]_.

  The dynamics of the calcium-activated non-selective cation channel model [1]_ [2]_ is given by:

  .. math::

      \begin{aligned}
      I_{CAN} &=g_{\mathrm{max}} M\left([Ca^{2+}]_{i}\right) p \left(V-E\right)\\
      &M\left([Ca^{2+}]_{i}\right) ={[Ca^{2+}]_{i} \over 0.2+[Ca^{2+}]_{i}} \\
      &{dp \over dt} = {\phi \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1.0 \over 1 + \exp(-(V + 43) / 5.2)} \\
      &\tau_{p} = {2.7 \over \exp(-(V + 55) / 15) + \exp((V + 55) / 15)} + 1.6
      \end{aligned}

  where :math:`\phi` is the temperature factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature factor.

  References
  ----------

  .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.
  .. [2] Inoue T, Strowbridge BW (2008) Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons.
         J Neurophysiol 99: 187–199.
  """
  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      E: Union[bst.typing.ArrayLike, Callable] = 10. * bu.mV,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.E = bst.init.param(E, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    V = V.to_decimal(bu.mV)
    self.p.value = 1.0 / (1 + bu.math.exp(-(V + 43.) / 5.2))

  def compute_derivative(self, V, Ca):
    V = V.to_decimal(bu.mV)
    phi_p = 1.0 / (1 + bu.math.exp(-(V + 43.) / 5.2))
    p_inf = 2.7 / (bu.math.exp(-(V + 55.) / 15.) + bu.math.exp((V + 55.) / 15.)) + 1.6
    self.p.derivative = self.phi * (phi_p - self.p.value) / p_inf / bu.ms

  def current(self, V, Ca):
    M = Ca.C / (Ca.C + 0.2 * bu.mM)
    g = self.g_max * M * self.p.value
    return g * (self.E - V)


class _ICa_p2q_ss(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with steady-state format.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 3.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      mode: Optional[bst.mixin.Mode] = None,
      name: Optional[str] = None
  ):
    super().__init__(
      size,
      name=name,
      mode=mode,
    )

    # parameters
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if batch_size is not None:
      assert self.p.value.shape[0] == batch_size
      assert self.q.value.shape[0] == batch_size

  def compute_derivative(self, V, Ca):
    self.p.derivative = self.phi_p * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / bu.ms
    self.q.derivative = self.phi_q * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / bu.ms

  def current(self, V, Ca):
    return self.g_max * self.p.value * self.p.value * self.q.value * (Ca.E - V)

  def f_p_inf(self, V):
    raise NotImplementedError

  def f_p_tau(self, V):
    raise NotImplementedError

  def f_q_inf(self, V):
    raise NotImplementedError

  def f_q_tau(self, V):
    raise NotImplementedError


class _ICa_p2q_markov(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with first-order Markov chain.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= \phi_p (\alpha_p(V)(1-p) - \beta_p(V)p) \\
      {dq \over dt} &= \phi_q (\alpha_q(V)(1-q) - \beta_q(V)q) \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 3.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    alpha, beta = self.f_p_alpha(V), self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha, beta = self.f_q_alpha(V), self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)

  def compute_derivative(self, V, Ca):
    p = self.p.value
    q = self.q.value
    self.p.derivative = self.phi_p * (self.f_p_alpha(V) * (1 - p) - self.f_p_beta(V) * p) / bu.ms
    self.q.derivative = self.phi_q * (self.f_q_alpha(V) * (1 - q) - self.f_q_beta(V) * q) / bu.ms

  def current(self, V, Ca):
    return self.g_max * self.p.value * self.p.value * self.q.value * (Ca.E - V)

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError

  def f_q_alpha(self, V):
    raise NotImplementedError

  def f_q_beta(self, V):
    raise NotImplementedError


class ICaT_HM1992(_ICa_p2q_ss):
  r"""
  The low-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+59-V_{sh}) / 6.2]} \\
      &\tau_{p} = 0.612 + {1 \over \exp [-(V+132.-V_{sh}) / 16.7]+\exp [(V+16.8-V_{sh}) / 18.2]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+83-V_{sh}) / 4]} \\
      & \begin{array}{l} \tau_{q} = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
          \tau_{q} = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array}

  where :math:`\phi_p = 3.55^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """
  __module__ = 'dendritex.channels'

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 3.55,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -3. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1 + bu.math.exp(-(V + 59.) / 6.2))

  def f_p_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (bu.math.exp(-(V + 132.) / 16.7) +
                 bu.math.exp((V + 16.8) / 18.2)) + 0.612

  def f_q_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp((V + 83.) / 4.0))

  def f_q_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return bu.math.where(V >= -80.,
                         bu.math.exp(-(V + 22.) / 10.5) + 28.,
                         bu.math.exp((V + 467.) / 66.6))


class ICaT_HP1992(_ICa_p2q_ss):
  r"""The low-threshold T-type calcium current model for thalamic
  reticular nucleus proposed by (Huguenard & Prince, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model in thalamic
  reticular nucleus neurons [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+52-V_{sh}) / 7.4]}  \\
      &\tau_{p} = 3+{1 \over \exp [(V+27-V_{sh}) / 10]+\exp [-(V+102-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+80-V_{sh}) / 5]} \\
      & \tau_q = 85+ {1 \over \exp [(V+48-V_{sh}) / 4]+\exp [-(V+407-V_{sh}) / 50]}

  where :math:`\phi_p = 5^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, Prince DA (1992) A novel T-type current underlies
         prolonged Ca2+- dependent burst firing in GABAergic neurons of rat
         thalamic reticular nucleus. J Neurosci 12: 3804–3817.

  See Also
  --------
  ICa_p2q_form
  """
  __module__ = 'dendritex.channels'

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 5.,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1.75 * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -3. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp(-(V + 52.) / 7.4))

  def f_p_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 3. + 1. / (bu.math.exp((V + 27.) / 10.) +
                      bu.math.exp(-(V + 102.) / 15.))

  def f_q_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp((V + 80.) / 5.))

  def f_q_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 85. + 1. / (bu.math.exp((V + 48.) / 4.) +
                       bu.math.exp(-(V + 407.) / 50.))


class ICaHT_HM1992(_ICa_p2q_ss):
  r"""The high-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The high-threshold T-type calcium current model is adopted from [1]_.
  Its dynamics is given by

  .. math::

      \begin{aligned}
      I_{\mathrm{Ca/HT}} &= g_{\mathrm{max}} p^2 q (V-E_{Ca})
      \\
      {dp \over dt} &= {\phi_{p} \cdot (p_{\infty} - p) \over \tau_{p}} \\
      &\tau_{p} =\frac{1}{\exp \left(\frac{V+132-V_{sh}}{-16.7}\right)+\exp \left(\frac{V+16.8-V_{sh}}{18.2}\right)}+0.612 \\
      & p_{\infty} = {1 \over 1+exp[-(V+59-V_{sh}) / 6.2]}
      \\
      {dq \over dt} &= {\phi_{q} \cdot (q_{\infty} - h) \over \tau_{q}} \\
      & \begin{array}{l} \tau_q = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
      \tau_q = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array} \\
      &q_{\infty}  = {1 \over 1+exp[(V+83 -V_{shift})/4]}
      \end{aligned}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : bst.typing.ArrayLike, Callable
    The maximum conductance.
  V_sh : bst.typing.ArrayLike, Callable
    The membrane potential shift.

  References
  ----------
  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """
  __module__ = 'dendritex.channels'

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 3.55,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 25. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=T_base_p ** ((T - 24) / 10),
      phi_q=T_base_q ** ((T - 24) / 10),
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp(-(V + 59.) / 6.2))

  def f_p_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (bu.math.exp(-(V + 132.) / 16.7) +
                 bu.math.exp((V + 16.8) / 18.2)) + 0.612

  def f_q_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp((V + 83.) / 4.))

  def f_q_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return bu.math.where(V >= -80.,
                         bu.math.exp(-(V + 22.) / 10.5) + 28.,
                         bu.math.exp((V + 467.) / 66.6))


class ICaHT_Re1993(_ICa_p2q_markov):
  r"""The high-threshold T-type calcium current model proposed by (Reuveni, et al., 1993) [1]_.

  HVA Calcium current was described for neocortical neurons by Sayer et al. (1990).
  Its dynamics is given by (the rate functions are measured under 36 Celsius):

  .. math::

     \begin{aligned}
      I_{L} &=\bar{g}_{L} q^{2} r\left(V-E_{\mathrm{Ca}}\right) \\
      \frac{\mathrm{d} q}{\mathrm{~d} t} &= \phi_p (\alpha_{q}(V)(1-q)-\beta_{q}(V) q) \\
      \frac{\mathrm{d} r}{\mathrm{~d} t} &= \phi_q (\alpha_{r}(V)(1-r)-\beta_{r}(V) r) \\
      \alpha_{q} &=\frac{0.055(-27-V+V_{sh})}{\exp [(-27-V+V_{sh}) / 3.8]-1} \\
      \beta_{q} &=0.94 \exp [(-75-V+V_{sh}) / 17] \\
      \alpha_{r} &=0.000457 \exp [(-13-V+V_{sh}) / 50] \\
      \beta_{r} &=\frac{0.0065}{\exp [(-15-V+V_{sh}) / 28]+1},
      \end{aligned}

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
    If `None`, :math:`\phi_p = \mathrm{T_base_p}^{\frac{T-23}{10}}`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.
    If `None`, :math:`\phi_q = \mathrm{T_base_q}^{\frac{T-23}{10}}`.

  References
  ----------
  .. [1] Reuveni, I., et al. "Stepwise repolarization from Ca2+ plateaus
         in neocortical pyramidal cells: evidence for nonhomogeneous
         distribution of HVA Ca2+ channels in dendrites." Journal of
         Neuroscience 13.11 (1993): 4609-4621.

  """
  __module__ = 'dendritex.channels'

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 2.3,
      T_base_q: bst.typing.ArrayLike = 2.3,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 23.) / 10.) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 23.) / 10.) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = (- V + self.V_sh).to_decimal(bu.mV)
    temp = -27 + V
    return 0.055 * temp / (bu.math.exp(temp / 3.8) - 1)

  def f_p_beta(self, V):
    V = (- V + self.V_sh).to_decimal(bu.mV)
    return 0.94 * bu.math.exp((-75. + V) / 17.)

  def f_q_alpha(self, V):
    V = (- V + self.V_sh).to_decimal(bu.mV)
    return 0.000457 * bu.math.exp((-13. + V) / 50.)

  def f_q_beta(self, V):
    V = (- V + self.V_sh).to_decimal(bu.mV)
    return 0.0065 / (bu.math.exp((-15. + V) / 28.) + 1.)


class ICaL_IS2008(_ICa_p2q_ss):
  r"""The L-type calcium channel model proposed by (Inoue & Strowbridge, 2008) [1]_.

  The L-type calcium channel model is adopted from (Inoue, et, al., 2008) [1]_.
  Its dynamics is given by:

  .. math::

      I_{CaL} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      & p_{\infty} = {1 \over 1+\exp [-(V+10-V_{sh}) / 4.]} \\
      & \tau_{p} = 0.4+{0.7 \over \exp [(V+5-V_{sh}) / 15]+\exp [-(V+5-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      & q_{\infty} = {1 \over 1+\exp [(V+25-V_{sh}) / 2]} \\
      & \tau_q = 300 + {100 \over \exp [(V+40-V_{sh}) / 9.5]+\exp [-(V+40-V_{sh}) / 9.5]}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float
    The temperature.
  T_base_p : float
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float
    The maximum conductance.
  V_sh : float
    The membrane potential shift.

  References
  ----------

  .. [1] Inoue, Tsuyoshi, and Ben W. Strowbridge. "Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons." Journal of
         neurophysiology 99, no. 1 (2008): 187-199.

  See Also
  --------
  ICa_p2q_form
  """
  __module__ = 'dendritex.channels'

  def __init__(
      self,
      size: bst.typing.Size,
      T: Union[bst.typing.ArrayLike, Callable] = 36.,
      T_base_p: Union[bst.typing.ArrayLike, Callable] = 3.55,
      T_base_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=T_base_p ** ((T - 24) / 10),
      phi_q=T_base_q ** ((T - 24) / 10),
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1 + bu.math.exp(-(V + 10.) / 4.))

  def f_p_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 0.4 + .7 / (bu.math.exp(-(V + 5.) / 15.) + bu.math.exp((V + 5.) / 15.))

  def f_q_inf(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 1. / (1. + bu.math.exp((V + 25.) / 2.))

  def f_q_tau(self, V):
    V = (V - self.V_sh).to_decimal(bu.mV)
    return 300. + 100. / (bu.math.exp((V + 40) / 9.5) + bu.math.exp(-(V + 40) / 9.5))
  

class ICav12_Ma2020(CalciumChannel):
  r'''
  : model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"
  : also added Calcium dependent inactivation
  '''

  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 0 * (bu.mS / bu.cm**2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0 * bu.mV,
      T_base: bst.typing.ArrayLike = 3 ,
      T: bst.typing.ArrayLike = 22.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi = bst.init.param( 1., self.varshape, allow_none=False)

    self.kf = 0.0005
    self.VDI =  0.17
  


  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.m = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.h = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.n = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    self.m.value =  self.f_m_inf(V)
    self.h.value =  self.f_h_inf(V)
    self.n.value =  self.f_n_inf(V,Ca)

  def compute_derivative(self, V, Ca):
    self.m.derivative = self.phi * (self.f_m_inf(V) - self.m.value) / self.f_m_tau(V) / bu.ms
    self.h.derivative = self.phi * (self.f_h_inf(V) - self.h.value) / self.f_h_tau(V) / bu.ms
    self.n.derivative = self.phi * (self.f_n_inf(V,Ca) - self.n.value) / self.f_n_tau(V) / bu.ms

  def f_m_inf(self, V):
    V = V / bu.mV
    return 1/(1 + bu.math.exp((V + 8.9)/(-6.7)))
  def f_h_inf(self, V):
    V = V / bu.mV
    return self.VDI/(1 + bu.math.exp((V +55)/8)) + (1-self.VDI)
  
  def f_n_inf(self, V, Ca):
    V = V / bu.mV
    return bu.math.ones_like(V)*self.kf/(self.kf + Ca.C/bu.mM)
  
  def f_m_tau(self, V):
    V = V / bu.mV
    mA = 39800*(V + 8.124)/(bu.math.exp((V + 8.124)/9.005) - 1)
    mB = 990*bu.math.exp(V/31.4)
    return 1/(mA + mB)
  
  def f_h_tau(self, V):
    return 44.3
  
  def f_n_tau(self, V):
    return  0.5
    

  def current(self, V, Ca: IonInfo):
    return self.g_max * self.m.value * self.h.value * self.n.value * (Ca.E - V)
  
class ICav13_Ma2020(CalciumChannel):
  r'''
  : model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"
  : also added Calcium dependent inactivation
  '''
  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 0 * (bu.mS / bu.cm **2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0 * bu.mV,
      T_base: bst.typing.ArrayLike = 3 ,
      T: bst.typing.ArrayLike = 22.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi = bst.init.param( 1., self.varshape, allow_none=False)

    self.kf = 0.0005
    self.VDI = 1
   


  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.m = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.h = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.n = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    self.m.value =  self.f_m_inf(V)
    self.h.value =  self.f_h_inf(V)
    self.n.value =  self.f_n_inf(V,Ca)

  def compute_derivative(self, V, Ca):
    self.m.derivative = self.phi * (self.f_m_inf(V) - self.m.value) / self.f_m_tau(V) / bu.ms
    self.h.derivative = self.phi * (self.f_h_inf(V) - self.h.value) / self.f_h_tau(V) / bu.ms
    self.n.derivative = self.phi * (self.f_n_inf(V,Ca) - self.n.value) / self.f_n_tau(V) / bu.ms



  def f_m_inf(self, V):
    V = V / bu.mV
    return 1.0/((bu.math.exp ( (V - (-40.0))/(-5))) + 1.0)
  def f_h_inf(self, V):
    V = V / bu.mV
    return self.VDI/( (bu.math.exp ( (V - (-37))/(5))) + 1.0) + (1-self.VDI)
  
  def f_n_inf(self, V, Ca):
    V = V / bu.mV
    return bu.math.ones_like(V)*self.kf/(self.kf + Ca.C/bu.mM) 
  
  def f_m_tau(self, V):
    V = V / bu.mV
    mA = (39800*( V + 67.24))/( bu.math.exp ( (V + 67.24)/15.005) - 1.0)
    mB = 3500* bu.math.exp(V/31.4) 
    return 1/(mA + mB)
  
  def f_h_tau(self, V):
    return 44.3
  
  def f_n_tau(self, V):
    return  0.5
  

  def current(self, V, Ca: IonInfo):
    return self.g_max * self.m.value * self.h.value * self.n.value * (Ca.E - V)
  

class ICav23_Ma2020(CalciumChannel):
  r'''
  TITLE Ca R-type channel with medium threshold for activation
  : used in distal dendritic regions, together with calH.mod, to help
  : the generation of Ca++ spikes in these regions
  : uses channel conductance (not permeability)
  : written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu
  : From car to Cav2_3
  '''

  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 0 * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0 * bu.mV,
      T_base: bst.typing.ArrayLike = 3 ,
      T: bst.typing.ArrayLike = 22.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi = bst.init.param( 1., self.varshape, allow_none=False)

    self.eca = 140 * bu.mV
  


  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.m = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.h = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    

  def reset_state(self, V, Ca, batch_size=None):
    self.m.value =  self.f_m_inf(V)
    self.h.value =  self.f_h_inf(V)
  

  def compute_derivative(self, V, Ca):
    self.m.derivative = self.phi * (self.f_m_inf(V) - self.m.value) / self.f_m_tau(V) / bu.ms
    self.h.derivative = self.phi * (self.f_h_inf(V) - self.h.value) / self.f_h_tau(V) / bu.ms


  def current(self, V, Ca: IonInfo):
      return self.g_max * self.m.value **3 * self.h.value *  (self.eca - V)

  def f_m_inf(self, V):
    V = V / bu.mV
    return 1 / (1 + bu.math.exp((V+48.5)/(-3))) 
  def f_h_inf(self, V):
    V = V / bu.mV
    return 1/ (1 + bu.math.exp((V+53)/ 1.)) 
  
  def f_m_tau(self, V):
    return 50.
  
  def f_h_tau(self, V):
    return 5.
  
  

class ICav31_Ma2020(CalciumChannel):
  r'''
  TITLE Low threshold calcium current Cerebellum Purkinje Cell Model

  COMMENT

  Kinetics adapted to fit the Cav3.1 Iftinca et al 2006, Temperature dependence of T-type Calcium channel gating, NEUROSCIENCE

  Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*

  *Article available as Open Access

  PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513

  Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
  Contact: Haroon Anwar (anwar@oist.jp)

  '''
  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2.5e-4 * (bu.cm / bu.second),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0 * bu.mV,
      T_base: bst.typing.ArrayLike = 3  ,
      T: bst.typing.ArrayLike = 22.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi = bst.init.param(T_base ** ((T - 37) / 10), self.varshape, allow_none=False)

    self.v0_m_inf = -52
    self.v0_h_inf = -72
    self.k_m_inf = -5
    self.k_h_inf = 7

    self.C_tau_m = 1
    self.A_tau_m = 1.0
    self.v0_tau_m1 = -40
    self.v0_tau_m2 = -102
    self.k_tau_m1 = 9
    self.k_tau_m2 = -18

    self.C_tau_h = 15
    self.A_tau_h = 1.0
    self.v0_tau_h1 = -32
    self.k_tau_h1 = 7



  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    self.p.value =  self.f_p_inf(V)
    self.q.value =  self.f_q_inf(V)

  def compute_derivative(self, V, Ca):
    self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / bu.ms
    self.q.derivative = self.phi * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / bu.ms

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1.0 / ( 1 + bu.math.exp((V  - self.v0_m_inf)/self.k_m_inf) )
  
  def f_q_inf(self, V):
    V = V / bu.mV
    return  1.0 / ( 1 + bu.math.exp((V - self.v0_h_inf)/self.k_h_inf) )
  
  def f_p_tau(self, V):
    V = V / bu.mV
    return bu.math.where(V<=-90, 1, (self.C_tau_m + self.A_tau_m / (bu.math.exp((V - self.v0_tau_m1)/ self.k_tau_m1) + bu.math.exp((V - self.v0_tau_m2)/self.k_tau_m2))) )
  def f_p_tau(self, V):
    V = V / bu.mV
    return ( self.C_tau_h + self.A_tau_h / bu.math.exp((V - self.v0_tau_h1)/self.k_tau_h1) )
  
  def ghk(self, V, Ca: IonInfo): 
    E = (1e-3) * V
    zeta = (2 * bu.faraday_constant * E )/( bu.gas_constant  * (273.15 + self.T) * bu.kelvin)
    zeta = zeta.to_decimal()
    ci = Ca.C
    co = 2 * bu.mM  #co = Ca.C0 for Calciumdetailed
    g_1 = (1e-6) * bu.faraday_constant * (ci - co * bu.math.exp(-zeta)) * (1 + zeta/2)
    print(g_1)
    g_2= (1e-6) * (zeta*bu.faraday_constant) * (ci - co*bu.math.exp(-zeta)) / (1-bu.math.exp(-zeta))
    return bu.math.where(bu.math.abs((1-bu.math.exp(-zeta))) <= 1e-6,g_1,g_2)
  
  def current(self, V, Ca: IonInfo):
    i = (1e3)*self.g_max * self.p.value ** 2 * self.q.value * self.ghk(V,Ca)
    print(self.ghk(V,Ca))
    return (1e3)*self.g_max * self.p.value ** 2 * self.q.value * self.ghk(V,Ca)
  


class ICaGrc_Ma2020(CalciumChannel):
  r'''
  TITLE Cerebellum Granule Cell Model

  COMMENT
          CaHVA channel
    
    Author: E.D'Angelo, T.Nieus, A. Fontana
    Last revised: 8.5.2000
  '''

  __module__ = 'dendritex.channels'

  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 0.46 * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0 * bu.mV,
      T_base: bst.typing.ArrayLike = 3 ,
      T: bst.typing.ArrayLike = 22.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi = bst.init.param( T_base ** ((T - 20) / 10), self.varshape, allow_none=False)

    self.eca = 129.33 * bu.mV

    self.Aalpha_s = 0.04944
    self.Kalpha_s = 15.87301587302
    self.V0alpha_s = -29.06

    self.Abeta_s = 0.08298
    self.Kbeta_s = -25.641
    self.V0beta_s = -18.66

    self.Aalpha_u = 0.0013
    self.Kalpha_u = -18.183
    self.V0alpha_u = -48

    self.Abeta_u = 0.0013
    self.Kbeta_u = 83.33
    self.V0beta_u = -48



  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.m = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.h = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    

  def reset_state(self, V, Ca, batch_size=None):
    self.m.value =  self.f_m_inf(V)
    self.h.value =  self.f_h_inf(V)
  

  def compute_derivative(self, V, Ca):
    self.m.derivative = self.phi * (self.f_m_inf(V) - self.m.value) / self.f_m_tau(V) / bu.ms
    self.h.derivative = self.phi * (self.f_h_inf(V) - self.h.value) / self.f_h_tau(V) / bu.ms


  def current(self, V, Ca: IonInfo):
      return self.g_max * self.m.value ** 2 * self.h.value *  (self.eca - V)

  def f_m_inf(self, V):
    return self.alpha_m(V)/(self.alpha_m(V) + self.beta_m(V)) 

  def f_h_inf(self, V):
    return  self.alpha_h(V)/(self.alpha_h(V) + self.beta_h(V)) 
  
  def f_m_tau(self, V):
    return 1./(self.alpha_m(V) + self.beta_m(V)) 
  
  def f_h_tau(self, V):
    return 1./(self.alpha_h(V) + self.beta_h(V)) 
  
  def alpha_m(self, V) :
    V = (V - self.V_sh) / bu.mV
    return self.Aalpha_s * bu.math.exp((V - self.V0alpha_s) / self.Kalpha_s)

  
  def beta_m(self, V) :
    V = (V - self.V_sh) / bu.mV
    return self.Abeta_s * bu.math.exp((V - self.V0beta_s) / self.Kbeta_s)

  
  def alpha_h(self, V) :
    V = (V - self.V_sh) / bu.mV
    return self.Aalpha_u * bu.math.exp((V - self.V0alpha_u) / self.Kalpha_u)

  
  def beta_h(self, V) :
    V = (V - self.V_sh) / bu.mV
    return self.Abeta_u * bu.math.exp((V - self.V0beta_u) / self.Kbeta_u)

 
 
