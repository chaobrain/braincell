"""Generated one-ion HH ohmic channel from {{ context.source_file }}."""

from __future__ import annotations

import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.quad import DiffEqState
from braincell.mech.channel import {{ context.base_class_name }}

ONE_ION_HH_OHMIC_IR = {{ context_python }}


class {{ context.class_name }}({{ context.base_class_name }}):
    __module__ = "braincell.channel"

    source_file = ONE_ION_HH_OHMIC_IR["source_file"]
    mechanism_name = ONE_ION_HH_OHMIC_IR["mechanism_name"]
    manual_fix_required = ONE_ION_HH_OHMIC_IR["manual_fix_required"]

    def __init__(
        self,
        size,
        g_max={{ context.g_max_param.default_expression }},
        V_sh={{ context.v_shift_param.default_expression }},
        temp={{ context.temperature_param.default_expression }},
{% for parameter in context.extra_parameters %}
        {{ parameter.safe_name }}={{ parameter.default_expression }},
{% endfor %}
        name=None,
    ):
        super().__init__(size=size, name=name)

        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = temp
{% for parameter in context.extra_parameters %}
        self.{{ parameter.safe_name }} = braintools.init.param({{ parameter.safe_name }}, self.varshape, allow_none=False)
{% endfor %}

        self.Tref = {{ context.tref_expression }}
{% for gate in context.gates %}
        self.Q10_{{ gate.safe_name }} = {{ gate.q10_expression }}
{% endfor %}

    def _q10(self, Q10):
        return Q10 ** (((self.temp - self.Tref) / u.kelvin) / 10.0)

    def init_state(self, V, {{ context.ion_arg_name }}: IonInfo, batch_size=None):
{% for gate in context.gates %}
        self.{{ gate.safe_name }} = DiffEqState(
            braintools.init.param(u.math.zeros, self.varshape, batch_size)
        )
{% endfor %}

    def reset_state(self, V, {{ context.ion_arg_name }}: IonInfo, batch_size=None):
{% for gate in context.gates %}
        self.{{ gate.safe_name }}.value = self.f_{{ gate.safe_name }}_inf(V)
{% endfor %}
{% if context.gates %}
        if isinstance(batch_size, int):
{% for gate in context.gates %}
            assert self.{{ gate.safe_name }}.value.shape[0] == batch_size
{% endfor %}
{% endif %}

    def pre_integral(self, V, {{ context.ion_arg_name }}: IonInfo):
        pass

    def post_integral(self, V, {{ context.ion_arg_name }}: IonInfo):
        pass

    def compute_derivative(self, V, {{ context.ion_arg_name }}: IonInfo):
{% for gate in context.gates %}
        phi_{{ gate.safe_name }} = self._q10(self.Q10_{{ gate.safe_name }})
        self.{{ gate.safe_name }}.derivative = phi_{{ gate.safe_name }} * (self.f_{{ gate.safe_name }}_inf(V) - self.{{ gate.safe_name }}.value) / self.f_{{ gate.safe_name }}_tau(V) / u.ms
{% endfor %}

    def current(self, V, {{ context.ion_arg_name }}: IonInfo):
        return {{ context.current_model.current_expression_python }}

{% for gate in context.gates %}
    def f_{{ gate.safe_name }}_inf(self, V):
        V = (V - self.V_sh) / u.mV
{% for line in gate.helper_alias_lines %}
        {{ line }}
{% endfor %}
        return {{ gate.inf_expr_python }}

    def f_{{ gate.safe_name }}_tau(self, V):
        V = (V - self.V_sh) / u.mV
{% for line in gate.helper_alias_lines %}
        {{ line }}
{% endfor %}
        return {{ gate.tau_expr_python }}

{% endfor %}
    def debug_ir(self):
        return ONE_ION_HH_OHMIC_IR
