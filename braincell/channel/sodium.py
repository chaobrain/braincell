# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent sodium channel.

"""

from typing import Union, Callable, Optional

import brainstate
import brainunit as u
import jax


from braincell._base import Channel, IonInfo
from braincell._integrator import get_integrator
from braincell._integrator_integrator import get_integrator
from braincell._integrator_protocol import DiffEqState, IndependentIntegration, IndependentIntegration
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
        super().__init__(
            size,
            name=name,
            phi=phi,
            g_max=g_max,
        )
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


class INa_Rsg(SodiumChannel, IndependentIntegration, IndependentIntegration): 
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: brainstate.typing.Size,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 15. * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
        solver: str = 'implicit_euler'
    ):
        super().__init__(size=size, name=name, )

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

        self.solver = get_integrator(solver)

        self.solver = get_integrator(solver)

        state_names = ["C1", "C2", "C3", "C4", "C5",  "I1", "I2", "I3", "I4", "I5", "O", "B",]
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
            ("C5", "O",  "f0O", "b0O"),
            ("O",  "B",  "fip", "bip"),
            ("O",  "I6", "fin", "bin"),
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

        state_value =u.math.stack([getattr(self, name).value for name in self.state_names])
        state_dict = {name: state_value[i] for i, name in enumerate(self.state_names)}
        state_dict[self.redundant_state] = 1.0 - u.math.sum(state_value, axis=0) 
        
        for src, dst, f_rate, b_rate in self.state_pairs:

            f = getattr(self, f_rate)(V)  
            b = getattr(self, b_rate)(V)  
            getattr(self, src).derivative += (-state_dict[src] * f + state_dict[dst] * b) / u.ms
            getattr(self, dst).derivative += ( state_dict[src] * f - state_dict[dst] * b) / u.ms

    def update(self, V, Na: IonInfo):
                
        # --- 1. Time step ---
        dt = u.get_magnitude(brainstate.environ.get_dt())  # scalar

        # --- 2. Stack current state values into tensor ---
        state_value = u.math.stack([getattr(self, name).value for name in self.state_names])
        # shape = (S, P, N)
        # S = number of non-redundant states
        # P = number of populations
        # N = number of compartments/segments

        # --- 3. Define ODE function to compute derivatives ---
        def ode_fn(S, V):
            """
            Compute time derivatives of channel states for given voltage V.

            Parameters
            ----------
            S : array, shape (S, P, N)
                Current state values
            V : array, shape (P, N)
                Membrane voltage

            Returns
            -------
            derivs_stacked : array, shape (S, P, N)
                Time derivatives of non-redundant states

            Notes
            -----
            - Redundant state is computed automatically: I = 1 - sum(other states)
            - Each state derivative is computed using forward/backward rates from self.state_pairs
            """
            # --- 3a. Create dict of states including redundant ---
            state_dict = {name: S[i] for i, name in enumerate(self.state_names)}
            state_dict[self.redundant_state] = 1.0 - u.math.sum(S, axis=0)  # shape (P, N)

            # --- 3b. Initialize derivatives dict ---
            derivs = {name: u.math.zeros_like(state_dict[name]) for name in state_dict}  # shape (P, N)

            # --- 3c. Compute derivatives from state transitions ---
            for from_state, to_state, forward_rate_fn, backward_rate_fn in self.state_pairs:
                f = getattr(self, forward_rate_fn)(V)   # forward rate, shape (P, N)
                b = getattr(self, backward_rate_fn)(V)  # backward rate, shape (P, N)

                derivs[from_state] += -state_dict[from_state] * f + state_dict[to_state] * b
                derivs[to_state]   +=  state_dict[from_state] * f - state_dict[to_state] * b

            # --- 3d. Stack derivatives for solver ---
            return u.math.stack([derivs[name] for name in self.state_names], axis=0)  # shape (S, P, N)

        # --- 4. Solve linear ODE system using backward Euler ---
        S_next = backward_euler_solver(ode_fn, state_value, dt, V)  # shape (S, P, N)

        # --- 5. Unpack updated states back into channel object ---
        for i, name in enumerate(self.state_names):
            getattr(self, name).value = S_next[i]  # shape (P, N)

    def current(self, V, Na: IonInfo):
        #jax.debug.print('O = {}',self.O.value)
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

def backward_euler_solver(ode_fn, S, dt, *args):
    """
    Solve a linear ODE system using the Backward Euler (implicit Euler) method for batched multi-dimensional states.

    Parameters
    ----------
    ode_fn : callable
        Linear ODE function: dS/dt = f(S, *args)
        Input: S with shape (State, batch_dims...)
        Output: f(S) with the same shape as S
        Note: Since the system is linear, the Jacobian J = df/dS is constant and represents the coefficient matrix.
    S : array_like
        Current state, shape = (State, batch_dims...)
    dt : float
        Time step for the backward Euler update
    *args :
        Additional arguments passed to ode_fn

    Returns
    -------
    S_next : array_like
        Updated state after one backward Euler step, shape = (State, batch_dims...)

    Notes
    -----
    For linear systems, f(S) = J @ S + c. The implicit Euler step is:
        S_{n+1} = S_n + dt * f(S_{n+1})

    This can be rewritten as a linear system for S_{n+1}:
        (I - dt * J) @ S_{n+1} = S_n + dt * (f(S_n) - J @ S_n)

    In this implementation:
    - J = Jacobian of ode_fn w.r.t. S, shape = (State, State, batch_dims...)
    - b = f(S) - J @ S, shape = (State, batch_dims...)  # residual / constant term
    - lhs = I - dt * J, shape = (State, State, batch_dims...)
    - rhs = S + dt * b, shape = (State, batch_dims...)
    - The linear system is solved for each batch element independently.
    
    All batch dimensions of S are automatically handled using reshape + solve.
    """

    # --- 1. Compute Jacobian w.r.t. state dimension (axis 0) ---
    jac_fn = jax.jacfwd(ode_fn, argnums=0)  # df/dS, shape = (State, State, batch_dims...)
    mapped_fn = jac_fn
    
    # --- 2. Map Jacobian computation over all batch dimensions ---
    for i in range(S.ndim - 1):
        # vmap along each batch axis to handle high-dimensional batches
        # in_axes and out_axes set to propagate batch dimensions correctly
        mapped_fn = jax.vmap(mapped_fn, in_axes=(i+1, i), out_axes=i+2)
    J = mapped_fn(S, *args)  # shape = (State, State, batch_dims...)

    # --- 3. Compute residual term ---
    dS = ode_fn(S, *args)                     # f(S), shape = (State, batch_dims...)
    b = dS - u.math.einsum('ij...,j...->i...', J, S)  # residual: f(S) - J @ S, same shape as S

    # --- 4. Form Backward Euler linear system ---
    # (I - dt * J) @ S_{n+1} = S_n + dt * b
    I = u.math.eye(S.shape[0])[..., None, None]  # identity matrix, shape = (State, State, 1, 1, ...)
    lhs = I - dt * J                             # left-hand side matrix, shape = (State, State, batch_dims...)
    rhs = S + dt * b                             # right-hand side, shape = (State, batch_dims...)

    # --- 5. Reshape for batch solve ---
    # reshape lhs to (batch_size, State, State) and rhs to (batch_size, State)
    lhs_reshaped = lhs.reshape(S.shape[0], S.shape[0], -1).transpose(2, 0, 1)
    rhs_reshaped = rhs.reshape(S.shape[0], -1).T

    # --- 6. Solve linear system for each batch ---
    S_next = u.math.linalg.solve(lhs_reshaped, rhs_reshaped[..., None])[..., 0]  # shape = (batch_size, State)
    S_next = S_next.T.reshape(S.shape)  # reshape back to original shape: (State, batch_dims...)

    return S_next