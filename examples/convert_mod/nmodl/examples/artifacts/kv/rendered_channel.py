"""Generated Braincell density channel from /home/swl/braincell/examples/convert_mod/nmodl/mod_files/kv.mod."""

import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel import PotassiumChannel
from braincell.mech import get_registry
from braincell.mech import register_channel
from braincell.quad import DiffEqState


def _to_decimal_if_possible(value, unit):
    if value is None:
        return None
    return value.to_decimal(unit) if hasattr(value, "to_decimal") else value


def _register_generated_channel(cls):
    registry = get_registry()
    if not registry.contains("channel", "IK_Kv"):
        register_channel("IK_Kv")(cls)
    return cls


class IK_Kv(PotassiumChannel):
    __module__ = "braincell.channel"

    def __init__(
        self,
        size,
        g_max=0.0 * (u.siemens / (u.cm ** 2)),
        V_sh=0. * u.mV,
        temp=u.celsius2kelvin(23),
        Ra=0.02 * (1 / u.mV / u.ms),
        Rb=0.006 * (1 / u.mV / u.ms),
        q=9 * (u.mV),
        v12=25 * (u.mV),
        name=None,
    ):
        super().__init__(size=size, name=name)

        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = temp
        self.Ra = braintools.init.param(Ra, self.varshape, allow_none=False)
        self.Rb = braintools.init.param(Rb, self.varshape, allow_none=False)
        self.q = braintools.init.param(q, self.varshape, allow_none=False)
        self.v12 = braintools.init.param(v12, self.varshape, allow_none=False)

        self.temp_ref = u.celsius2kelvin(23)
        self.Q10_n = 1.0

    def _q10(self, Q10):
        return Q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)

    def init_state(self, V, K: IonInfo, batch_size=None):
        self.n = DiffEqState(
            braintools.init.param(u.math.zeros, self.varshape, batch_size)
        )

    def reset_state(self, V, K: IonInfo, batch_size=None):
        self.n.value = self.f_n_inf(V)
        if isinstance(batch_size, int):
            assert self.n.value.shape[0] == batch_size

    def pre_integral(self, V, K: IonInfo):
        pass

    def post_integral(self, V, K: IonInfo):
        pass

    def compute_derivative(self, V, K: IonInfo):
        phi_n = self._q10(self.Q10_n)
        self.n.derivative = (
            phi_n
            * (self.f_n_inf(V) - self.n.value)
            / self.f_n_tau(V)
            / u.ms
        )

    def current(self, V, K: IonInfo):
        return self.g_max * self.n.value * (K.E - V)

    def f_n_inf(self, V):
        V = (V - self.V_sh) / u.mV
        Ra = self.Ra / (1 / u.mV / u.ms)
        Rb = self.Rb / (1 / u.mV / u.ms)
        q = self.q / (u.mV)
        v12 = self.v12 / (u.mV)
        return 1/(1+u.math.exp(-(V-v12)/q))

    def f_n_tau(self, V):
        V = (V - self.V_sh) / u.mV
        Ra = self.Ra / (1 / u.mV / u.ms)
        Rb = self.Rb / (1 / u.mV / u.ms)
        q = self.q / (u.mV)
        v12 = self.v12 / (u.mV)
        return 1/((1)*(((Ra)*((V)-(v12))/(1-u.math.exp(-((V)-(v12))/(q))))+((-Rb)*((V)-(v12))/(1-u.math.exp(-((V)-(v12))/(-q))))))

_register_generated_channel(IK_Kv)