# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent sodium channel.

"""

from typing import Union, Callable, Optional

import brainstate
import brainunit as u

from braincell._base import Channel, IonInfo
from braincell._integrator import get_integrator
from braincell._integrator_protocol import DiffEqState, IndependentIntegration
from braincell.ion import Sodium

__all__ = [
    'SodiumChannel',
    'INa_p3q_markov',
    'INa_Ba2002',
    'INa_TM1991',
    'INa_HH1952',
    'INa_Rsg',
]


class SodiumChannel(Channel):
    """
    Base class for sodium channel dynamics.

    This class provides a template for implementing sodium channel models.
    It defines methods that should be overridden by subclasses to implement
    specific sodium channel behaviors.
    """

    __module__ = 'braincell.channel'

    root_type = Sodium

    def pre_integral(self, V, Na: IonInfo):
        """
        Perform any necessary operations before the integration step.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.
        """
        pass

    def post_integral(self, V, Na: IonInfo):
        """
        Perform any necessary operations after the integration step.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.
        """
        pass

    def compute_derivative(self, V, Na: IonInfo):
        """
        Compute the derivative of the channel state variables.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.
        """
        pass

    def current(self, V, Na: IonInfo):
        """
        Calculate the sodium current through the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.

        Raises:
        NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def init_state(self, V, Na: IonInfo, batch_size: int = None):
        """
        Initialize the state variables of the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.
        batch_size : int, optional
            Size of the batch for vectorized operations.
        """
        pass

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        """
        Reset the state variables of the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        Na : IonInfo
            Information about sodium ions.
        batch_size : int, optional
            Size of the batch for vectorized operations.
        """
        pass


class INa_p3q_markov(SodiumChannel):
    r"""
    The sodium current model of :math:`p^3q` current which described with first-order Markov chain.

    The general model can be used to model the dynamics with:

    .. math::

      \begin{aligned}
      I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
      \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
      \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
      \end{aligned}

    where :math:`\phi` is a temperature-dependent factor.

    Parameters
    ----------
    g_max : float, ArrayType, Callable, Initializer
      The maximal conductance density (:math:`mS/cm^2`).
    phi : float, ArrayType, Callable, Initializer
      The temperature-dependent factor.
    name: str
      The name of the object.

    """

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 90. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        self.phi = brainstate.init.param(phi, self.varshape, allow_none=False)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)

    def init_state(self, V, Na: IonInfo, batch_size=None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
        self.q = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, Na: IonInfo, batch_size=None):
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        self.p.value = alpha / (alpha + beta)
        alpha = self.f_q_alpha(V)
        beta = self.f_q_beta(V)
        self.q.value = alpha / (alpha + beta)

    def compute_derivative(self, V, Na: IonInfo):
        p = self.p.value
        q = self.q.value
        self.p.derivative = self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p) / u.ms
        self.q.derivative = self.phi * (self.f_q_alpha(V) * (1. - q) - self.f_q_beta(V) * q) / u.ms

    def current(self, V, Na: IonInfo):
        return self.g_max * self.p.value ** 3 * self.q.value * (Na.E - V)

    def f_p_alpha(self, V):
        raise NotImplementedError

    def f_p_beta(self, V):
        raise NotImplementedError

    def f_q_alpha(self, V):
        raise NotImplementedError

    def f_q_beta(self, V):
        raise NotImplementedError


class INa_Ba2002(INa_p3q_markov):
    r"""
    The sodium current model.

    The sodium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
    It's dynamics is given by:

    .. math::

      \begin{aligned}
      I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
      \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &=\frac{0.32\left(V-V_{sh}-13\right)}{1-\exp \left(-\left(V-V_{sh}-13\right) / 4\right)} \\
      \beta_{p} &=\frac{-0.28\left(V-V_{sh}-40\right)}{1-\exp \left(\left(V-V_{sh}-40\right) / 5\right)} \\
      \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
      \alpha_q &=0.128 \exp \left(-\left(V-V_{sh}-17\right) / 18\right) \\
      \beta_q &= \frac{4}{1+\exp \left(-\left(V-V_{sh}-40\right) / 5\right)}
      \end{aligned}

    where :math:`\phi` is a temperature-dependent factor, which is given by
    :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

    Parameters
    ----------
    g_max : float, ArrayType, Callable, Initializer
      The maximal conductance density (:math:`mS/cm^2`).
    T : float, ArrayType
      The temperature (Celsius, :math:`^{\circ}C`).
    V_sh : float, ArrayType, Callable, Initializer
      The shift of the membrane potential to spike.

    References
    ----------

    .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
           and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

    See Also
    --------
    INa_TM1991
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: brainstate.typing.Size,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(36.),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 90. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -50. * u.mV,
        name: Optional[str] = None,
    ):
        T = u.kelvin2celsius(T)
        super().__init__(
            size,
            name=name,
            phi=3 ** ((T - 36) / 10),
            g_max=g_max,
        )
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        temp = V - 13.
        return 0.32 * temp / (1. - u.math.exp(-temp / 4.))

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        temp = V - 40.
        return -0.28 * temp / (1. - u.math.exp(temp / 5.))

    def f_q_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.128 * u.math.exp(-(V - 17.) / 18.)

    def f_q_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 4. / (1. + u.math.exp(-(V - 40.) / 5.))


class INa_TM1991(INa_p3q_markov):
    r"""
    The sodium current model described by (Traub and Miles, 1991) [1]_.

    The dynamics of this sodium current model is given by:

    .. math::

       \begin{split}
       \begin{aligned}
          I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
          \frac {dm} {dt} &= \phi(\alpha_m (1-x)  - \beta_m) \\
          &\alpha_m(V) = 0.32 \frac{(13 - V + V_{sh})}{\exp((13 - V +V_{sh}) / 4) - 1.}  \\
          &\beta_m(V) = 0.28 \frac{(V - V_{sh} - 40)}{(\exp((V - V_{sh} - 40) / 5) - 1)}  \\
          \frac {dh} {dt} &= \phi(\alpha_h (1-x)  - \beta_h) \\
          &\alpha_h(V) = 0.128 * \exp((17 - V + V_{sh}) / 18)  \\
          &\beta_h(V) = 4. / (1 + \exp(-(V - V_{sh} - 40) / 5)) \\
       \end{aligned}
       \end{split}

    where :math:`V_{sh}` is the membrane shift (default -63 mV), and
    :math:`\phi` is the temperature-dependent factor (default 1.).

    Parameters
    ----------
    size: int, tuple of int
      The size of the simulation target.
    name: str
      The name of the object.
    g_max : float, ArrayType, Callable, Initializer
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh: float, ArrayType, Callable, Initializer
      The membrane shift.

    References
    ----------
    .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
           Vol. 777. Cambridge University Press, 1991.

    See Also
    --------
    INa_Ba2002
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -63. * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(
            size,
            name=name,
            phi=phi,
            g_max=g_max,
        )
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V):
        V = (self.V_sh - V).to_decimal(u.mV)
        temp = 13 + V
        return 0.32 * 4 / u.math.exprel(temp / 4)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        temp = V - 40
        return 0.28 * 5 / u.math.exprel(temp / 5)

    def f_q_alpha(self, V):
        V = (- V + self.V_sh).to_decimal(u.mV)
        return 0.128 * u.math.exp((17 + V) / 18)

    def f_q_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 4. / (1 + u.math.exp(-(V - 40) / 5))


class INa_HH1952(INa_p3q_markov):
    r"""
    The sodium current model described by Hodgkin–Huxley model [1]_.

    The dynamics of this sodium current model is given by:

    .. math::

       \begin{split}
       \begin{aligned}
          I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
          \frac {dm} {dt} &= \phi (\alpha_m (1-x)  - \beta_m) \\
          &\alpha_m(V) = \frac {0.1(V-V_{sh}-5)}{1-\exp(\frac{-(V -V_{sh} -5)} {10})}  \\
          &\beta_m(V) = 4.0 \exp(\frac{-(V -V_{sh}+ 20)} {18})  \\
          \frac {dh} {dt} &= \phi (\alpha_h (1-x)  - \beta_h) \\
          &\alpha_h(V) = 0.07 \exp(\frac{-(V-V_{sh}+20)}{20})  \\
          &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V -V_{sh}-10)} {10})} \\
       \end{aligned}
       \end{split}

    where :math:`V_{sh}` is the membrane shift (default -45 mV), and
    :math:`\phi` is the temperature-dependent factor (default 1.).

    Parameters
    ----------
    size: int, tuple of int
      The size of the simulation target.
    name: str
      The name of the object.
    g_max : float, ArrayType, Callable, Initializer
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh: float, ArrayType, Callable, Initializer
      The membrane shift.

    References
    ----------
    .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
           membrane current and its application to conduction and excitation in
           nerve." The Journal of physiology 117.4 (1952): 500.

    See Also
    --------
    IK_HH1952
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -45. * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size, name=name, phi=phi, g_max=g_max)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V):
        temp = (V - self.V_sh).to_decimal(u.mV) - 5
        return 1. / u.math.exprel(-temp / 10)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 4.0 * u.math.exp(-(V + 20) / 18)

    def f_q_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.07 * u.math.exp(-(V + 20) / 20.)

    def f_q_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1 / (1 + u.math.exp(-(V - 10) / 10))


class INa_Rsg(SodiumChannel, IndependentIntegration):
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: brainstate.typing.Size,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 15. * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
        solver: str = 'backward_euler',
    ):
        super().__init__(size=size, name=name)
        IndependentIntegration.__init__(self, solver=solver)

        T = u.kelvin2celsius(T)
        self.phi = brainstate.init.param(3 ** ((T - 22) / 10), self.varshape, allow_none=False)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)

        self.Con = 0.005
        self.Coff = 0.5
        self.Oon = 0.75
        self.Ooff = 0.005
        self.alpha = 150.
        self.beta = 3.
        self.gamma = 150.
        self.delta = 40.
        self.epsilon = 1.75
        self.zeta = 0.03

        self.x1 = 20.
        self.x2 = -20.
        self.x3 = 1e12
        self.x4 = -1e12
        self.x5 = 1e12
        self.x6 = -25.
        self.vshifta = 0.
        self.vshifti = 0.
        self.vshiftk = 0.

        self.alfac = (self.Oon / self.Con) ** (1 / 4)
        self.btfac = (self.Ooff / self.Coff) ** (1 / 4)

    def init_state(self, V, Na: IonInfo, batch_size=None):
        state_names = ["C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B", ]
        for name in state_names:
            state = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
            setattr(self, name, state)

        self.state_names = state_names
        self.redundant_state = "I6"

        self.state_pairs = [
            ("C1", "C2", "f01", "b01"),
            ("C2", "C3", "f02", "b02"),
            ("C3", "C4", "f03", "b03"),
            ("C4", "C5", "f04", "b04"),
            ("C5", "O", "f0O", "b0O"),
            ("O", "B", "fip", "bip"),
            ("O", "I6", "fin", "bin"),
            ("I1", "I2", "f11", "b11"),
            ("I2", "I3", "f12", "b12"),
            ("I3", "I4", "f13", "b13"),
            ("I4", "I5", "f14", "b14"),
            ("I5", "I6", "f1n", "b1n"),
            ("C1", "I1", "fi1", "bi1"),
            ("C2", "I2", "fi2", "bi2"),
            ("C3", "I3", "fi3", "bi3"),
            ("C4", "I4", "fi4", "bi4"),
            ("C5", "I5", "fi5", "bi5"),
        ]

    def reset_state(self, V, Na: IonInfo, batch_size=None):

        state_names = ["C1", "C2", "C3", "C4", "C5", "O", "B", "I1", "I2", "I3", "I4", "I5"]
        for name in state_names:
            state = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
            setattr(self, name, state)

    def pre_integral(self, V, Na: IonInfo):
        # state_value = u.math.clip(u.math.stack([getattr(self, name).value for name in self.state_names]), 0., 1.,)
        # state_sum = state_value.sum(axis=0, keepdims=True)
        # state_value = u.math.where(state_sum > 1., state_value / state_sum , state_value)
        pass

    def compute_derivative(self, V, Na: IonInfo):
        state_value = u.math.stack([getattr(self, name).value for name in self.state_names])
        state_dict = {name: state_value[i] for i, name in enumerate(self.state_names)}
        state_dict[self.redundant_state] = 1.0 - u.math.sum(state_value, axis=0)

        derivative_dict = {name: u.math.zeros_like(st) for name, st in state_dict.items()}
       
        for src, dst, f_rate, b_rate in self.state_pairs:
            f = getattr(self, f_rate)(V)
            b = getattr(self, b_rate)(V)
            derivative_dict[src] += -state_dict[src] * f + state_dict[dst] * b 
            derivative_dict[dst] += state_dict[src] * f - state_dict[dst] * b  

        for name in self.state_names:
            getattr(self, name).derivative = derivative_dict[name] /u.ms


    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    f01 = lambda self, V: 4 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f02 = lambda self, V: 3 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f03 = lambda self, V: 2 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f04 = lambda self, V: 1 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f0O = lambda self, V: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fip = lambda self, V: self.epsilon * u.math.exp((V / u.mV) / self.x5) * self.phi
    f11 = lambda self, V: 4 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f12 = lambda self, V: 3 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f13 = lambda self, V: 2 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f14 = lambda self, V: 1 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f1n = lambda self, V: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fi1 = lambda self, V: self.Con * self.phi
    fi2 = lambda self, V: self.Con * self.alfac * self.phi
    fi3 = lambda self, V: self.Con * self.alfac ** 2 * self.phi
    fi4 = lambda self, V: self.Con * self.alfac ** 3 * self.phi
    fi5 = lambda self, V: self.Con * self.alfac ** 4 * self.phi
    fin = lambda self, V: self.Oon * self.phi

    b01 = lambda self, V: 1 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b02 = lambda self, V: 2 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b03 = lambda self, V: 3 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b04 = lambda self, V: 4 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b0O = lambda self, V: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bip = lambda self, V: self.zeta * u.math.exp(V / u.mV / self.x6) * self.phi
    b11 = lambda self, V: 1 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b12 = lambda self, V: 2 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b13 = lambda self, V: 3 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b14 = lambda self, V: 4 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b1n = lambda self, V: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bi1 = lambda self, V: self.Coff * self.phi
    bi2 = lambda self, V: self.Coff * self.btfac * self.phi
    bi3 = lambda self, V: self.Coff * self.btfac ** 2 * self.phi
    bi4 = lambda self, V: self.Coff * self.btfac ** 3 * self.phi
    bi5 = lambda self, V: self.Coff * self.btfac ** 4 * self.phi
    bin = lambda self, V: self.Ooff * self.phi
