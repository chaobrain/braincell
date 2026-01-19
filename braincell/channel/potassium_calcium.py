# -*- coding: utf-8 -*-


"""
This module implements calcium-dependent potassium channel.
"""

from typing import Union, Callable, Optional

import brainstate
import braintools
import brainunit as u
import jax

from braincell._base import IonInfo, Channel
from braincell._integrator import DiffEqState
from braincell.ion import Calcium, Potassium

__all__ = [
    'KCaChannel',
    'IAHP_De1994',
    'IKca3_1_Ma2020',
    'IKca2_2_Ma2020',
    'IKca1_1_Ma2020',

]


class KCaChannel(Channel):
    """
    A base class for calcium-dependent potassium channels.

    This class defines the interface for calcium-dependent potassium channels,
    inheriting from the base Channel class. It provides a structure for
    implementing specific types of KCa channels.
    """

    __module__ = 'braincell.channel'

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]

    def pre_integral(self, V, K: IonInfo, Ca: IonInfo):
        """
        Perform any necessary computations before the integration step.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.
        """
        pass

    def post_integral(self, V, K: IonInfo, Ca: IonInfo):
        """
        Perform any necessary computations after the integration step.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.
        """
        pass

    def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
        """
        Compute the derivative of the channel's state variables.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.
        """
        pass

    def current(self, V, K: IonInfo, Ca: IonInfo):
        """
        Calculate the current through the channel.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.

        Returns
        ----------
        array_like
            The calculated current through the channel.

        Raises:
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        """
        Initialize the state variables of the channel.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.
        batch_size : int, optional
            The batch size for initialization.
        """
        pass

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        """
        Reset the state variables of the channel.

        Parameters
        ----------
        V : array_like
            Membrane potential.
        K : IonInfo
            Information about potassium ions.
        Ca : IonInfo
            Information about calcium ions.
        batch_size : int, optional
            The batch size for resetting.
        """
        pass


class IAHP_De1994(KCaChannel):
    r"""The calcium-dependent potassium current model proposed by (Destexhe, et al., 1994) [1]_.

    Both in vivo (Contreras et al. 1993; Mulle et al. 1986) and in
    vitro recordings (Avanzini et al. 1989) show the presence of a
    marked after-hyper-polarization (AHP) after each burst of the RE
    cell. This slow AHP is mediated by a slow :math:`Ca^{2+}`-dependent K+
    current (Bal and McCormick 1993). (Destexhe, et al., 1994) adopted a
    modified version of a model of :math:`I_{KCa}` introduced previously (Yamada et al.
    1989) that requires the binding of :math:`nCa^{2+}` to open the channel

    $$
    \begin{aligned}
    (\text{closed}) + n \mathrm{Ca}_{i}^{2+} \underset{\beta}{\stackrel{\alpha}{\rightleftharpoons}} (\text{open})
    \end{aligned}
    $$

    where :math:`Ca_i^{2+}` is the intracellular calcium and :math:`\alpha` and
    :math:`\beta` are rate constants. The ionic current is then given by

    .. math::

        \begin{aligned}
        I_{AHP} &= g_{\mathrm{max}} p^2 (V - E_K) \\
        {dp \over dt} &= \phi {p_{\infty}(V, [Ca^{2+}]_i) - p \over \tau_p(V, [Ca^{2+}]_i)} \\
        p_{\infty} &=\frac{\alpha[Ca^{2+}]_i^n}{\left(\alpha[Ca^{2+}]_i^n + \beta\right)} \\
        \tau_p &=\frac{1}{\left(\alpha[Ca^{2+}]_i +\beta\right)}
        \end{aligned}

    where :math:`E` is the reversal potential, :math:`g_{max}` is the maximum conductance,
    :math:`[Ca^{2+}]_i` is the intracellular Calcium concentration.
    The values :math:`n=2, \alpha=48 \mathrm{~ms}^{-1} \mathrm{mM}^{-2}` and
    :math:`\beta=0.03 \mathrm{~ms}^{-1}` yielded AHPs very similar to those RE cells
    recorded in vivo and in vitro.

    Parameters
    ----------
    g_max : float
      The maximal conductance density (:math:`mS/cm^2`).

    References
    ----------

    .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
           thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.

    """
    __module__ = 'braincell.channel'

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]

    def __init__(
        self,
        size: brainstate.typing.Size,
        n: Union[brainstate.typing.ArrayLike, Callable] = 2,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 48.,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.09,
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.n = braintools.init.param(n, self.varshape, allow_none=False)
        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.phi = braintools.init.param(phi, self.varshape, allow_none=False)

    def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
        C2 = self.alpha * u.math.power(Ca.C / u.mM, self.n)
        C3 = C2 + self.beta
        self.p.derivative = self.phi * (C2 / C3 - self.p.value) * C3 / u.ms

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * self.p.value * self.p.value * (K.E - V)

    def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        self.p = DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size))

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        C2 = self.alpha * u.math.power(Ca.C / u.mM, self.n)
        C3 = C2 + self.beta
        if batch_size is None:
            self.p.value = u.math.broadcast_to(C2 / C3, self.varshape)
        else:
            self.p.value = u.math.broadcast_to(C2 / C3, (batch_size,) + self.varshape)
            assert self.p.value.shape[0] == batch_size


class IKca3_1_Ma2020(KCaChannel):
    r'''
      TITLE Calcium dependent potassium channel
    : Implemented in Rubin and Cleland (2006) J Neurophysiology
    : Parameters from Bhalla and Bower (1993) J Neurophysiology
    : Adapted from /usr/local/neuron/demo/release/nachan.mod - squid
    :   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]
    '''
    __module__ = 'braincell.channel'

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120. * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
        self.phi = braintools.init.param(T_base ** ((T - 37) / 10), self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

        self.p_beta = 0.05

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * self.p.value * (K.E - V)

    def p_tau(self, V, Ca):
        return 1 / (self.p_alpha(V, Ca) + self.p_beta)

    def p_inf(self, V, Ca):
        return self.p_alpha(V, Ca) / (self.p_alpha(V, Ca) + self.p_beta)

    def p_alpha(self, V, Ca):
        V = V / u.mV
        return self.p_vdep(V) * self.p_concdep(Ca)

    def p_vdep(self, V):
        return u.math.exp((V + 70.) / 27.)

    def p_concdep(self, Ca):
        # concdep_1 = 500 * (0.015 - Ca.C / u.mM) / (u.math.exp((0.015 - Ca.C / u.mM) / 0.0013) - 1)
        concdep_1 = 500 * 0.0013 / u.math.exprel((0.015 - Ca.C / u.mM) / 0.0013)
        with jax.ensure_compile_time_eval():
            concdep_2 = 500 * 0.005 / (u.math.exp(0.005 / 0.0013) - 1)
        return u.math.where(Ca.C / u.mM < 0.01, concdep_1, concdep_2)

    def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        self.p = DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size))
        self.reset_state(V, K, Ca)

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        self.p.value = self.p_inf(V, Ca)

    def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
        self.p.derivative = self.phi * (self.p_inf(V, Ca) - self.p.value) / self.p_tau(V, Ca) / u.ms


class IKca2_2_Ma2020(KCaChannel):
    r'''
    TITLE SK2 multi-state model Cerebellum Golgi Cell Model

    COMMENT

    Author:Sergio Solinas, Lia Forti, Egidio DAngelo
    Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
    Last revised: May 2007

    Published in:
                Sergio M. Solinas, Lia Forti, Elisabetta Cesana,
                Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
                Computational reconstruction of pacemaking and intrinsic
                electroresponsiveness in cerebellar golgi cells
                Frontiers in Cellular Neuroscience 2:2
    '''
    __module__ = 'braincell.channel'

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 38. * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.,
        diff: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
        self.phi = braintools.init.param(T_base ** ((T - 23) / 10), self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.diff = braintools.init.param(diff, self.varshape, allow_none=False)

        self.invc1 = 80e-3  # (/ms)
        self.invc2 = 80e-3  # (/ms)
        self.invc3 = 200e-3  # (/ms)

        self.invo1 = 1  # (/ms)
        self.invo2 = 100e-3  # (/ms)
        self.diro1 = 160e-3  # (/ms)
        self.diro2 = 1.2  # (/ms)

        self.dirc2 = 200  # (/ms-mM)
        self.dirc3 = 160  # (/ms-mM)
        self.dirc4 = 80  # (/ms-mM)

    def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):

        self.C1 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.C2 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.C3 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.C4 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.O1 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.O2 = DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size))
        self.normalize_states([self.C1, self.C2, self.C3, self.C4, self.O1, self.O2])

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        self.normalize_states([self.C1, self.C2, self.C3, self.C4, self.O1, self.O2])

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * (self.O1.value + self.O2.value) * (K.E - V)

    def pre_integral(self, V, K: IonInfo, Ca: IonInfo):
        self.normalize_states([self.C1, self.C2, self.C3, self.C4, self.O1, self.O2])

    def normalize_states(self, states):
        total = 0.
        for state in states:
            state.value = u.math.maximum(state.value, 0)
            total = total + state.value
        for state in states:
            state.value = state.value / total

    def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):

        self.C1.derivative = (self.C2.value * self.invc1_t(Ca) - self.C1.value * self.dirc2_t_ca(Ca)) / u.ms
        self.C2.derivative = (self.C3.value * self.invc2_t(Ca) + self.C1.value * self.dirc2_t_ca(Ca) - self.C2.value * (
            self.invc1_t(Ca) + self.dirc3_t_ca(Ca))) / u.ms
        self.C3.derivative = (self.C4.value * self.invc3_t(Ca) + self.O1.value * self.invo1_t(Ca) - self.C3.value * (
            self.dirc4_t_ca(Ca) + self.diro1_t(Ca))) / u.ms
        self.C4.derivative = (self.C3.value * self.dirc4_t_ca(Ca) + self.O2.value * self.invo2_t(Ca) - self.C4.value * (
            self.invc3_t(Ca) + self.diro2_t(Ca))) / u.ms
        self.O1.derivative = (self.C3.value * self.diro1_t(Ca) - self.O1.value * self.invo1_t(Ca)) / u.ms
        self.O2.derivative = (self.C4.value * self.diro2_t(Ca) - self.O2.value * self.invo2_t(Ca)) / u.ms

    dirc2_t_ca = lambda self, Ca: self.dirc2_t * (Ca.C / u.mM) / self.diff
    dirc3_t_ca = lambda self, Ca: self.dirc3_t * (Ca.C / u.mM) / self.diff
    dirc4_t_ca = lambda self, Ca: self.dirc4_t * (Ca.C / u.mM) / self.diff

    invc1_t = lambda self, Ca: self.invc1 * self.phi
    invc2_t = lambda self, Ca: self.invc2 * self.phi
    invc3_t = lambda self, Ca: self.invc3 * self.phi
    invo1_t = lambda self, Ca: self.invo1 * self.phi
    invo2_t = lambda self, Ca: self.invo2 * self.phi
    diro1_t = lambda self, Ca: self.diro1 * self.phi
    diro2_t = lambda self, Ca: self.diro2 * self.phi
    dirc2_t = lambda self, Ca: self.dirc2 * self.phi
    dirc3_t = lambda self, Ca: self.dirc3 * self.phi
    dirc4_t = lambda self, Ca: self.dirc4 * self.phi


class IKca1_1_Ma2020(KCaChannel):
    r'''
    TITLE Large conductance Ca2+ activated K+ channel mslo

    COMMENT

    Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).

    Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channel with models of Ca2+ buffering in Purkinje cell. Cerebellum*

    *Article available as Open Access

    PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513


    Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
    Contact: Sungho Hong (shhong@oist.jp)
    '''
    __module__ = 'braincell.channel'

    root_type = brainstate.mixin.JointTypes[Potassium, Calcium]

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        # parameters
        T = u.kelvin2celsius(T)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
        self.phi = braintools.init.param(T_base ** ((T - 23) / 10), self.varshape, allow_none=False)

        self.Qo = 0.73
        self.Qc = -0.67
        self.k1 = 1.0e3
        self.onoffrate = 1.
        self.L0 = 1806
        self.Kc = 11.0e-3
        self.Ko = 1.1e-3

        self.pf0 = 2.39e-3
        self.pf1 = 7.0e-3
        self.pf2 = 40e-3
        self.pf3 = 295e-3
        self.pf4 = 557e-3

        self.pb0 = 3936e-3
        self.pb1 = 1152e-3
        self.pb2 = 659e-3
        self.pb3 = 486e-3
        self.pb4 = 92e-3

    def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):

        for i in range(5):
            setattr(self, f'C{i}', DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size)))

        for i in range(5):
            setattr(self, f'O{i}', DiffEqState(braintools.init.param(u.math.ones, self.varshape, batch_size)))

        self.normalize_states([getattr(self, f'C{i}') for i in range(5)] + [getattr(self, f'O{i}') for i in range(5)])

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
        self.normalize_states([getattr(self, f'C{i}') for i in range(5)] + [getattr(self, f'O{i}') for i in range(5)])

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * (self.O1.value + self.O2.value) * (K.E - V)

    def pre_integral(self, V, K: IonInfo, Ca: IonInfo):
        pass

    def normalize_states(self, states):
        total = 0.
        for state in states:
            state.value = u.math.maximum(state.value, 0)
            total = total + state.value
        for state in states:
            state.value = state.value / total

    def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
        self.normalize_states([getattr(self, f'C{i}') for i in range(5)] + [getattr(self, f'O{i}') for i in range(5)])

        self.C0.derivative = (self.C1 * self.c10(Ca) + self.O0 * self.b0(V) - self.C0 * (
            self.c01(Ca) + self.f0(V))) / u.ms
        self.C1.derivative = (self.C0 * self.c01(Ca) + self.C2 * self.c21(Ca) + self.O1 * self.b1(V) - self.C1 * (
            self.c10(Ca) + self.c12(Ca) + self.f1(V))) / u.ms
        self.C2.derivative = (self.C1 * self.c12(Ca) + self.C3 * self.c32(Ca) + self.O2 * self.b2(V) - self.C2 * (
            self.c21(Ca) + self.c23(Ca) + self.f2(V))) / u.ms
        self.C3.derivative = (self.C2 * self.c23(Ca) + self.C4 * self.c43(Ca) + self.O3 * self.b3(V) - self.C3 * (
            self.c32(Ca) + self.c34(Ca) + self.f3(V))) / u.ms
        self.C4.derivative = (self.C3 * self.c34(Ca) + self.O4 * self.b4(V) - self.C4 * (
            self.c43(Ca) + self.f4(V))) / u.ms

        self.O0.derivative = (self.O1 * self.o10(Ca) + self.C0 * self.f0(V) - self.O0 * (
            self.o01(Ca) + self.b0(V))) / u.ms
        self.O1.derivative = (self.O0 * self.o01(Ca) + self.O2 * self.o21(Ca) + self.C1 * self.f1(V) - self.O1 * (
            self.o10(Ca) + self.o12(Ca) + self.b1(V))) / u.ms
        self.O2.derivative = (self.O1 * self.o12(Ca) + self.O3 * self.o32(Ca) + self.C2 * self.f2(V) - self.O2 * (
            self.o21(Ca) + self.o23(Ca) + self.b2(V))) / u.ms
        self.O3.derivative = (self.O2 * self.o23(Ca) + self.O4 * self.o43(Ca) + self.C3 * self.f3(V) - self.O3 * (
            self.o32(Ca) + self.o34(Ca) + self.b3(V))) / u.ms
        self.O4.derivative = (self.O3 * self.o34(Ca) + self.C4 * self.f4(V) - self.O4 * (
            self.o43(Ca) + self.b4(V))) / u.ms

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * (self.O0.value + self.O1.value + self.O2.value + self.O3.value + self.O4.value) * (K.E - V)

    c01 = lambda self, Ca: 4 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    c12 = lambda self, Ca: 3 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    c23 = lambda self, Ca: 2 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    c34 = lambda self, Ca: 1 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi

    o01 = lambda self, Ca: 4 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    o12 = lambda self, Ca: 3 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    o23 = lambda self, Ca: 2 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi
    o34 = lambda self, Ca: 1 * (Ca.C / u.mM) * self.k1 * self.onoffrate * self.phi

    c10 = lambda self, Ca: 1 * self.Kc * self.k1 * self.onoffrate * self.phi
    c21 = lambda self, Ca: 2 * self.Kc * self.k1 * self.onoffrate * self.phi
    c32 = lambda self, Ca: 3 * self.Kc * self.k1 * self.onoffrate * self.phi
    c43 = lambda self, Ca: 4 * self.Kc * self.k1 * self.onoffrate * self.phi

    o10 = lambda self, Ca: 1 * self.Ko * self.k1 * self.onoffrate * self.phi
    o21 = lambda self, Ca: 2 * self.Ko * self.k1 * self.onoffrate * self.phi
    o32 = lambda self, Ca: 3 * self.Ko * self.k1 * self.onoffrate * self.phi
    o43 = lambda self, Ca: 4 * self.Ko * self.k1 * self.onoffrate * self.phi

    alpha = lambda self, V: u.math.exp(
        (self.Qo * u.faraday_constant * V) / (u.gas_constant * (273.15 + self.T) * u.kelvin))
    beta = lambda self, V: u.math.exp(
        (self.Qc * u.faraday_constant * V) / (u.gas_constant * (273.15 + self.T) * u.kelvin))

    f0 = lambda self, V: self.pf0 * self.alpha(V) * self.phi
    f1 = lambda self, V: self.pf1 * self.alpha(V) * self.phi
    f2 = lambda self, V: self.pf2 * self.alpha(V) * self.phi
    f3 = lambda self, V: self.pf3 * self.alpha(V) * self.phi
    f4 = lambda self, V: self.pf4 * self.alpha(V) * self.phi

    b0 = lambda self, V: self.pb0 * self.beta(V) * self.phi
    b1 = lambda self, V: self.pb1 * self.beta(V) * self.phi
    b2 = lambda self, V: self.pb2 * self.beta(V) * self.phi
    b3 = lambda self, V: self.pb3 * self.beta(V) * self.phi
    b4 = lambda self, V: self.pb4 * self.beta(V) * self.phi
