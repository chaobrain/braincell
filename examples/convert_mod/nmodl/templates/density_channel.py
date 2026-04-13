"""Generated Braincell density channel from {{ context.source_file }}."""

import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel import {{ context.base_class_name }}
from braincell.mech import get_registry
from braincell.mech import register_channel
from braincell.quad import DiffEqState


def _to_decimal_if_possible(value, unit):
    if value is None:
        return None
    return value.to_decimal(unit) if hasattr(value, "to_decimal") else value


def _register_generated_channel(cls):
    registry = get_registry()
    if not registry.contains("channel", "{{ context.registry_name }}"):
        register_channel("{{ context.registry_name }}")(cls)
    return cls


class {{ context.class_name }}({{ context.base_class_name }}):
    __module__ = "braincell.channel"

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

        self.temp_ref = {{ context.tref_expression }}
{% for gate in context.gates %}
        self.Q10_{{ gate.safe_name }} = {{ gate.q10_expression }}
{% endfor %}

    def _q10(self, Q10):
        return Q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)

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
        self.{{ gate.safe_name }}.derivative = (
            phi_{{ gate.safe_name }}
            * (self.f_{{ gate.safe_name }}_inf(V) - self.{{ gate.safe_name }}.value)
            / self.f_{{ gate.safe_name }}_tau(V)
            / u.ms
        )
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
_register_generated_channel({{ context.class_name }})
