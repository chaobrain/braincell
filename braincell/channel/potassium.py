# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent potassium channel.

"""

from typing import Union, Callable, Optional, Sequence

import brainstate
import brainunit as u

from braincell._base import Channel, IonInfo
from braincell._protocol import DiffEqState
from braincell.ion import Potassium

__all__ = [
    'PotassiumChannel',

    'IK_p4_markov',
    'IKDR_Ba2002',
    'IK_TM1991',
    'IK_HH1952',

    'IKA_p4q_ss',
    'IKA1_HM1992',
    'IKA2_HM1992',

    'IKK2_pq_ss',
    'IKK2A_HM1992',
    'IKK2B_HM1992',
    'IKNI_Ya1989',
    'IK_Leak',
    'IKv11_Ak2007',
    'IKv34_Ma2020',
    'IKv43_Ma2020',
    'IKM_Grc_Ma2020',

]


class PotassiumChannel(Channel):
    """Base class for potassium channel dynamics.

    This class provides a template for implementing various potassium channel models.
    It defines methods that should be overridden by subclasses to implement
    specific potassium channel behaviors.
    """

    __module__ = 'braincell.channel'

    root_type = Potassium

    def pre_integral(self, V, K: IonInfo):
        """Perform any necessary operations before integration.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        """
        pass

    def post_integral(self, V, K: IonInfo):
        """Perform any necessary operations after integration.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        """
        pass

    def compute_derivative(self, V, K: IonInfo):
        """Compute the derivative of the channel state variables.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        """
        pass

    def current(self, V, K: IonInfo):
        """Calculate the potassium current through the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        """Initialize the state variables of the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        batch_size : int, optional
            The batch size for initialization.
        """
        pass

    def reset_state(self, V, K: IonInfo, batch_size: int = None):
        """Reset the state variables of the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        batch_size : int, optional
            The batch size for resetting.
        """
        pass


class IK_p4_markov(PotassiumChannel):
    r"""The delayed rectifier potassium channel of :math:`p^4`
    current which described with first-order Markov chain.

    This general potassium current model should have the form of

    .. math::

        \begin{aligned}
        I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
        \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p)
        \end{aligned}

    where :math:`\phi` is a temperature-dependent factor.

    Parameters
    ----------
    size: int, sequence of int
      The object size.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    phi : brainstate.typing.ArrayLike, Callable
      The temperature-dependent factor.
    name: Optional[str]
      The object name.

    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.phi = brainstate.init.param(phi, self.varshape, allow_none=False)

    def init_state(self, V, K: IonInfo, batch_size=None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size: int = None):
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        self.p.value = alpha / (alpha + beta)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size

    def compute_derivative(self, V, K: IonInfo):
        p = self.p.value
        dp = self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p) / u.ms
        self.p.derivative = dp

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value ** 4 * (K.E - V)

    def f_p_alpha(self, V):
        raise NotImplementedError

    def f_p_beta(self, V):
        raise NotImplementedError


class IKDR_Ba2002(IK_p4_markov):
    r"""The delayed rectifier potassium channel current.

    The potassium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
    It's dynamics is given by:

    .. math::

        \begin{aligned}
        I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
        \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
        \alpha_{p} &=\frac{0.032\left(V-V_{sh}-15\right)}{1-\exp \left(-\left(V-V_{sh}-15\right) / 5\right)} \\
        \beta_p &= 0.5 \exp \left(-\left(V-V_{sh}-10\right) / 40\right)
        \end{aligned}

    where :math:`\phi` is a temperature-dependent factor, which is given by
    :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

    Parameters
    ----------
    size: int, sequence of int
      The object size.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    T_base : float, ArrayType
      The brainpy_object of temperature factor.
    T : brainstate.typing.ArrayLike, Callable
      The temperature (Celsius, :math:`^{\circ}C`).
    V_sh : brainstate.typing.ArrayLike, Callable
      The shift of the membrane potential to spike.
    name: Optional[str]
      The object name.

    References
    ----------
    .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
           and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -50. * u.mV,
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(36.),
        phi: Optional[Union[brainstate.typing.ArrayLike, Callable]] = None,
        name: Optional[str] = None,
    ):
        T = u.kelvin2celsius(T)
        phi = T_base ** ((T - 36) / 10) if phi is None else phi
        super().__init__(
            size,
            name=name,
            g_max=g_max,
            phi=phi,
        )

        # parameters
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.T_base = brainstate.init.param(T_base, self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        tmp = V - 15.
        return 0.032 * 5. / u.math.exprel(-tmp / 5.)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.5 * u.math.exp(-(V - 10.) / 40.)


class IK_TM1991(IK_p4_markov):
    r"""The potassium channel described by (Traub and Miles, 1991) [1]_.

    The dynamics of this channel is given by:

    .. math::

       \begin{aligned}
        I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
        \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
        \alpha_{p} &= 0.032 \frac{(15 - V + V_{sh})}{(\exp((15 - V + V_{sh}) / 5) - 1.)} \\
        \beta_p &= 0.5 * \exp((10 - V + V_{sh}) / 40)
        \end{aligned}

    where :math:`V_{sh}` is the membrane shift (default -63 mV), and
    :math:`\phi` is the temperature-dependent factor (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    name: Optional[str]
      The object name.

    References
    ----------
    .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
           Vol. 777. Cambridge University Press, 1991.

    See Also
    --------
    INa_TM1991
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        V_sh: Union[int, brainstate.typing.ArrayLike, Callable] = -60. * u.mV,
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
        c = 15 + (- V + self.V_sh).to_decimal(u.mV)
        return 0.032 * 5 / u.math.exprel(c / 5)

    def f_p_beta(self, V):
        V = (self.V_sh - V).to_decimal(u.mV)
        return 0.5 * u.math.exp((10 + V) / 40)


class IK_HH1952(IK_p4_markov):
    r"""The potassium channel described by Hodgkin–Huxley model [1]_.

    The dynamics of this channel is given by:

    .. math::

       \begin{aligned}
        I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
        \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
        \alpha_{p} &= \frac{0.01 (V -V_{sh} + 10)}{1-\exp \left(-\left(V-V_{sh}+ 10\right) / 10\right)} \\
        \beta_p &= 0.125 \exp \left(-\left(V-V_{sh}+20\right) / 80\right)
        \end{aligned}

    where :math:`V_{sh}` is the membrane shift (default -45 mV), and
    :math:`\phi` is the temperature-dependent factor (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    name: Optional[str]
      The object name.

    References
    ----------
    .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
           membrane current and its application to conduction and excitation in
           nerve." The Journal of physiology 117.4 (1952): 500.

    See Also
    --------
    INa_HH1952
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        V_sh: Union[int, brainstate.typing.ArrayLike, Callable] = -45. * u.mV,
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
        V = (V - self.V_sh).to_decimal(u.mV)
        temp = -(V + 10) / 10
        return 0.1 / u.math.exprel(temp)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.125 * u.math.exp(-(V + 20) / 80)


class IKA_p4q_ss(PotassiumChannel):
    r"""
    The rapidly inactivating Potassium channel of :math:`p^4q`
    current which described with steady-state format.

    This model is developed according to the average behavior of
    rapidly inactivating Potassium channel in Thalamus relay neuron [2]_ [3]_.

    .. math::

       &IA = g_{\mathrm{max}} p^4 q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.phi_p = brainstate.init.param(phi_p, self.varshape, allow_none=False)
        self.phi_q = brainstate.init.param(phi_q, self.varshape, allow_none=False)

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi_p * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / u.ms
        self.q.derivative = self.phi_q * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / u.ms

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value ** 4 * self.q.value * (K.E - V)

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
        self.q = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        self.q.value = self.f_q_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size
            assert self.q.value.shape[0] == batch_size

    def f_p_inf(self, V):
        raise NotImplementedError

    def f_p_tau(self, V):
        raise NotImplementedError

    def f_q_inf(self, V):
        raise NotImplementedError

    def f_q_tau(self, V):
        raise NotImplementedError


class IKA1_HM1992(IKA_p4q_ss):
    r"""The rapidly inactivating Potassium channel (IA1) model proposed by (Huguenard & McCormick, 1992) [2]_.

    This model is developed according to the average behavior of
    rapidly inactivating Potassium channel in Thalamus relay neuron [2]_ [1]_.

    .. math::

       &IA = g_{\mathrm{max}} p^4 q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 60)/8.5]} \\
       &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
       &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
       &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
            \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh : float, ArrayType, Callable, Initializer
      The membrane potential shift.
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.

    See Also
    --------
    IKA2_HM1992
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 30. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            size,
            name=name,
            g_max=g_max,
            phi_p=phi_p,
            phi_q=phi_q,
        )

        # parameters
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V + 60.) / 8.5))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (u.math.exp((V + 35.8) / 19.7) +
                     u.math.exp(-(V + 79.7) / 12.7)) + 0.37

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp((V + 78.) / 6.))

    def f_q_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V < -63,
            1. / (u.math.exp((V + 46.) / 5.) +
                  u.math.exp(-(V + 238.) / 37.5)),
            19.
        )


class IKA2_HM1992(IKA_p4q_ss):
    r"""The rapidly inactivating Potassium channel (IA2) model proposed by (Huguenard & McCormick, 1992) [2]_.

    This model is developed according to the average behavior of
    rapidly inactivating Potassium channel in Thalamus relay neuron [2]_ [1]_.

    .. math::

       &IA = g_{\mathrm{max}} p^4 q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 36)/20.]} \\
       &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
       &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
       &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
            \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh : float, ArrayType, Callable, Initializer
      The membrane potential shift.
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.

    See Also
    --------
    IKA1_HM1992
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 20. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            size,
            name=name,
            g_max=g_max,
            phi_q=phi_q,
            phi_p=phi_p,
        )

        # parameters
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V + 36.) / 20.))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (u.math.exp((V + 35.8) / 19.7) +
                     u.math.exp(-(V + 79.7) / 12.7)) + 0.37

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp((V + 78.) / 6.))

    def f_q_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V < -63,
            1. / (u.math.exp((V + 46.) / 5.) +
                  u.math.exp(-(V + 238.) / 37.5)),
            19.
        )


class IKK2_pq_ss(PotassiumChannel):
    r"""The slowly inactivating Potassium channel of :math:`pq`
    current which described with steady-state format.

    The dynamics of the model is given as [2]_ [3]_.

    .. math::

       &IK2 = g_{\mathrm{max}} p q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.

    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )
        # parameters
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.phi_p = brainstate.init.param(phi_p, self.varshape, allow_none=False)
        self.phi_q = brainstate.init.param(phi_q, self.varshape, allow_none=False)

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi_p * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V)
        self.q.derivative = self.phi_q * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V)

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value * self.q.value * (K.E - V)

    def init_state(self, V, Ca: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
        self.q = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        self.q.value = self.f_q_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size
            assert self.q.value.shape[0] == batch_size

    def f_p_inf(self, V):
        raise NotImplementedError

    def f_p_tau(self, V):
        raise NotImplementedError

    def f_q_inf(self, V):
        raise NotImplementedError

    def f_q_tau(self, V):
        raise NotImplementedError


class IKK2A_HM1992(IKK2_pq_ss):
    r"""The slowly inactivating Potassium channel (IK2a) model proposed by (Huguenard & McCormick, 1992) [2]_.

    The dynamics of the model is given as [2]_ [3]_.

    .. math::

       &IK2 = g_{\mathrm{max}} p q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
       &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
          \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
       &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
       & \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) + \exp((V -V_{sh}+130)/-7.1)} + 120 \\

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh : float, ArrayType, Callable, Initializer
      The membrane potential shift.
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.

    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS * u.cm ** -2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            size,
            name=name,
            phi_p=phi_p,
            phi_q=phi_q,
            g_max=g_max,
        )

        # parameters
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V + 43.) / 17.))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (u.math.exp((V - 81.) / 25.6) +
                     u.math.exp(-(V + 132) / 18.)) + 9.9

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp((V + 58.) / 10.6))

    def f_q_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (u.math.exp((V - 1329.) / 200.) +
                     u.math.exp(-(V + 130.) / 7.1))


class IKK2B_HM1992(IKK2_pq_ss):
    r"""The slowly inactivating Potassium channel (IK2b) model proposed by (Huguenard & McCormick, 1992) [2]_.

    The dynamics of the model is given as [2]_ [3]_.

    .. math::

       &IK2 = g_{\mathrm{max}} p q (E-V) \\
       &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
       &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
       &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
       \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
       &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
       &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
       &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) +
                        \exp((V -V_{sh}+130)/-7.1)} + 120 \quad V<(-70+V_{sh})\, mV  \\
            \tau_{q} = 8.9  \quad V \geq (-70 + V_{sh})\, mV \end{array}

    where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh : float, ArrayType, Callable, Initializer
      The membrane potential shift.
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    phi_q : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`q`.

    References
    ----------
    .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neuron." Journal of neurophysiology 68.4 (1992): 1373-1383.
    .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
           TEA-sensitive K current in acutely isolated rat thalamic relay
           neuron." Journal of neurophysiology 66.4 (1991): 1316-1328.

    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS * u.cm ** -2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            size,
            name=name,
            phi_p=phi_p,
            phi_q=phi_q,
            g_max=g_max,
        )

        # parameters
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V + 43.) / 17.))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (u.math.exp((V - 81.) / 25.6) +
                     u.math.exp(-(V + 132) / 18.)) + 9.9

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp((V + 58.) / 10.6))

    def f_q_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V < -70,
            1. / (u.math.exp((V - 1329.) / 200.) +
                  u.math.exp(-(V + 130.) / 7.1)),
            8.9
        )


class IKNI_Ya1989(PotassiumChannel):
    r"""A slow non-inactivating K+ current described by Yamada et al. (1989) [1]_.

    This slow potassium current can effectively account for spike-frequency adaptation.

    .. math::

      \begin{aligned}
      &I_{M}=\bar{g}_{M} p\left(V-E_{K}\right) \\
      &\frac{\mathrm{d} p}{\mathrm{~d} t}=\left(p_{\infty}(V)-p\right) / \tau_{p}(V) \\
      &p_{\infty}(V)=\frac{1}{1+\exp [-(V-V_{sh}+35) / 10]} \\
      &\tau_{p}(V)=\frac{\tau_{\max }}{3.3 \exp [(V-V_{sh}+35) / 20]+\exp [-(V-V_{sh}+35) / 20]}
      \end{aligned}

    where :math:`\bar{g}_{M}` was :math:`0.004 \mathrm{mS} / \mathrm{cm}^{2}` and
    :math:`\tau_{\max }=4 \mathrm{~s}`, unless stated otherwise.

    Parameters
    ----------
    size: int, sequence of int
      The geometry size.
    name: Optional[str]
      The object name.
    g_max : brainstate.typing.ArrayLike, Callable
      The maximal conductance density (:math:`mS/cm^2`).
    V_sh : float, ArrayType, Callable, Initializer
      The membrane potential shift.
    phi_p : optional, float, ArrayType, Callable, Initializer
      The temperature factor for channel :math:`p`.
    tau_max: float, ArrayType, Callable, Initializer
      The :math:`tau_{\max}` parameter.

    References
    ----------
    .. [1] Yamada, Walter M. "Multiple channel and calcium dynamics." Methods in neuronal modeling (1989): 97-133.

    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.004 * (u.mS * u.cm ** -2),
        phi_p: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        phi_q: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        tau_max: Union[brainstate.typing.ArrayLike, Callable] = 4e3 * u.ms,
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.tau_max = brainstate.init.param(tau_max, self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)
        self.phi_p = brainstate.init.param(phi_p, self.varshape, allow_none=False)
        self.phi_q = brainstate.init.param(phi_q, self.varshape, allow_none=False)

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi_p * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V)

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value * (K.E - V)

    def init_state(self, V, Ca: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V + 35.) / 10.))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        temp = V + 35.
        return self.tau_max / (3.3 * u.math.exp(temp / 20.) + u.math.exp(-temp / 20.))


class IK_Leak(PotassiumChannel):
    """The potassium leak channel current.

    Parameters
    ----------
    g_max : float
      The potassium leakage conductance which is modulated by both
      acetylcholine and norepinephrine.
    """
    __module__ = 'braincell.channel'

    root_type = Potassium

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[int, brainstate.typing.ArrayLike, Callable] = 0.005 * (u.mS * u.cm ** -2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )
        self.g_max = brainstate.init.param(g_max, self.varshape)

    def reset_state(self, V, K: IonInfo, batch_size: int = None):
        pass

    def compute_derivative(self, V, K: IonInfo):
        pass

    def current(self, V, K: IonInfo):
        return self.g_max * (K.E - V)


class IKv11_Ak2007(PotassiumChannel):
    r"""
    TITLE Voltage-gated low threshold potassium current from Kv1 subunits

    COMMENT

    NEURON implementation of a potassium channel from Kv1.1 subunits
    Kinetical scheme: Hodgkin-Huxley m^4, no inactivation

    Experimental data taken from:
    Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
    Vhalf = -28.8 +- 2.3 mV; k = 8.1+- 0.9 mV

    The voltage dependency of the rate constants was approximated by:

    alpha = ca * exp(-(v+cva)/cka)
    beta = cb * exp(-(v+cvb)/ckb)

    Parameters ca, cva, cka, cb, cvb, ckb
    were determined from least square-fits to experimental data of G/Gmax(v) and tau(v).
    Values are defined in the CONSTANT block.
    Model includes calculation of Kv gating current

    Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

    Laboratory for Neuronal Circuit Dynamics
    RIKEN Brain Science Institute, Wako City, Japan
    http://www.neurodynamics.brain.riken.jp

    Date of Implementation: April 2007
    Contact: akemann@brain.riken.jp
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4. * (u.mS / u.cm ** 2),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.,
        gunit: Union[brainstate.typing.ArrayLike, Callable] = 16. * 1e-9 * u.mS,
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        T_base: brainstate.typing.ArrayLike = 2.7,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):

        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.gateCurrent = brainstate.init.param(gateCurrent, self.varshape, allow_none=False)
        self.gunit = brainstate.init.param(gunit, self.varshape, allow_none=False)
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.T_base = brainstate.init.param(T_base, self.varshape, allow_none=False)
        self.phi = brainstate.init.param(T_base ** ((T - 22) / 10), self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

        self.e0 = 1.60217646e-19 * u.coulomb
        self.q10 = 2.7
        self.ca = 0.12889
        self.cva = 45
        self.cka = -33.90877
        self.cb = 0.12889
        self.cvb = 45
        self.ckb = 12.42101
        self.zn = 2.7978

    def init_state(self, V, K: IonInfo, batch_size=None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size: int = None):
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        self.p.value = alpha / (alpha + beta)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi * (
            self.f_p_alpha(V) * (1. - self.p.value) - self.f_p_beta(V) * self.p.value) / u.ms

    def current(self, V, K: IonInfo):
        if self.gateCurrent == 0:
            ik = self.g_max * self.p.value ** 4 * (K.E - V)
        else:
            ngateFlip = self.phi * (self.f_p_alpha(V) * (1. - self.p.value) - self.f_p_beta(V) * self.p.value) / u.ms
            igate = (
                        1e12) * self.g_max / self.gunit * 1e6 * self.e0 * 4 * self.zn * ngateFlip  # NONSPECIFIC_CURRENT igate

            ik = -igate + self.g_max * self.p.value ** 4 * (K.E - V)
        return ik

    def f_p_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.ca * u.math.exp(- (V + self.cva) / self.cka)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.cb * u.math.exp(-(V + self.cvb) / self.ckb)


class IKv34_Ma2020(PotassiumChannel):
    r"""
    : HH TEA-sensitive Purkinje potassium current
    : Created 8/5/02 - nwg
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -11. * u.mV,
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.T_base = brainstate.init.param(T_base, self.varshape, allow_none=False)
        self.phi = brainstate.init.param(T_base ** ((T - 37) / 10), self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

        self.mivh = -24
        self.mik = 15.4
        self.mty0 = .00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1

        self.hiy0 = .31
        self.hiA = .69
        self.hivh = -5.802
        self.hik = 11.2

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / u.ms
        self.q.derivative = self.phi * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / u.ms

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value ** 3 * self.q.value * (K.E - V)

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
        self.q = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        self.q.value = self.f_q_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size
            assert self.q.value.shape[0] == batch_size

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1. / (1. + u.math.exp(-(V - self.mivh) / self.mik))

    def f_p_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        mtau_func = u.math.where(
            V < -35,
            (3.4225e-5 + .00498 * u.math.exp(-V / -28.29)) * 3,
            (self.mty0 + 1. / (u.math.exp((V + self.mtvh1) / self.mtk1) +
                               u.math.exp((V + self.mtvh2) / self.mtk2)))
        )
        return 1000 * mtau_func

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.hiy0 + self.hiA / (1 + u.math.exp((V - self.hivh) / self.hik))

    def f_q_tau(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        htau_func = u.math.where(
            V > 0,
            (.0012 + .0023 * u.math.exp(-.141 * V)),
            (1.2202e-05 + .012 * u.math.exp(-((V - (-56.3)) / 49.6) ** 2))
        )
        return 1000 * htau_func


class IKv43_Ma2020(PotassiumChannel):
    r"""
    TITLE Cerebellum Granule Cell Model

    COMMENT
          KA channel

      Author: E.D'Angelo, T.Nieus, A. Fontana
      Last revised: Egidio 3.12.2003
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.T_base = brainstate.init.param(T_base, self.varshape, allow_none=False)
        self.phi = brainstate.init.param(T_base ** ((T - 25.5) / 10), self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.32708
        self.V0alpha_a = -9.17203
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175
        self.V0beta_a = -18.27914

        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433
        self.V0alpha_b = -111.33209
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123
        self.V0beta_b = -49.9537

        self.V0_ainf = -38
        self.K_ainf = -17

        self.V0_binf = -78.8
        self.K_binf = 8.4

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / u.ms
        self.q.derivative = self.phi * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / u.ms

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value ** 3 * self.q.value * (K.E - V)

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))
        self.q = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        self.q.value = self.f_q_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size
            assert self.q.value.shape[0] == batch_size

    def sigm(self, x, y):
        return 1 / (u.math.exp(x / y) + 1)

    def f_p_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Aalpha_a * self.sigm(V - self.V0alpha_a, self.Kalpha_a)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Abeta_a / (u.math.exp((V - self.V0beta_a) / self.Kbeta_a))

    def f_q_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Aalpha_b * self.sigm(V - self.V0alpha_b, self.Kalpha_b)

    def f_q_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Abeta_b * self.sigm(V - self.V0beta_b, self.Kbeta_b)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1 / (1 + u.math.exp((V - self.V0_ainf) / self.K_ainf))

    def f_p_tau(self, V):
        return 1. / (self.f_p_alpha(V) + self.f_p_beta(V))

    def f_q_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1 / (1 + u.math.exp((V - self.V0_binf) / self.K_binf))

    def f_q_tau(self, V):
        return 1. / (self.f_q_alpha(V) + self.f_q_beta(V))


class IKM_Grc_Ma2020(PotassiumChannel):
    r"""
    TITLE Cerebellum Granule Cell Model

    COMMENT
            KM channel

      Author: A. Fontana
      CoAuthor: T.Nieus Last revised: 20.11.99
    """
    __module__ = 'braincell.channel'

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)
        self.T_base = brainstate.init.param(T_base, self.varshape, allow_none=False)
        self.phi = brainstate.init.param(T_base ** ((T - 22) / 10), self.varshape, allow_none=False)
        self.V_sh = brainstate.init.param(V_sh, self.varshape, allow_none=False)

        self.ek = -84.69 * u.mV

        self.Aalpha_n = 0.0033
        self.Kalpha_n = 40

        self.V0alpha_n = -30
        self.Abeta_n = 0.0033
        self.Kbeta_n = -20

        self.V0beta_n = -30
        self.V0_ninf = -35
        self.B_ninf = 6

    def compute_derivative(self, V, K: IonInfo):
        self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / u.ms

    def current(self, V, K: IonInfo):
        return self.g_max * self.p.value * (self.ek - V)

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        self.p = DiffEqState(brainstate.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.p.value = self.f_p_inf(V)
        if isinstance(batch_size, int):
            assert self.p.value.shape[0] == batch_size

    def f_p_alpha(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp((V - self.V0alpha_n) / self.Kalpha_n)

    def f_p_beta(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return self.Abeta_n * u.math.exp((V - self.V0beta_n) / self.Kbeta_n)

    def f_p_inf(self, V):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1 / (1 + u.math.exp(-(V - self.V0_ninf) / self.B_ninf))

    def f_p_tau(self, V):
        return 1. / (self.f_p_alpha(V) + self.f_p_beta(V))
