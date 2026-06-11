from __future__ import annotations

from dataclasses import dataclass

import brainunit as u

from braincell import Cell, mech
from braincell.filter import AllRegion

from dcn_native import DcnMorphology, dcn_region, load_dcn_morphology


@dataclass(frozen=True)
class DcnTemplateScales:
    """Temperature scaling parameters for the DCN template.

    Parameters
    ----------
    celsius : float, optional
        Simulation temperature in Celsius.
    temp_orig_dcn : float, optional
        Reference temperature used by the source DCN template.
    q10_channel_gating : float, optional
        Q10 factor for channel gating kinetics.
    q10_conductances : float, optional
        Q10 factor for conductance and permeability values.
    q10_ca_conc : float, optional
        Q10 factor for calcium concentration dynamics.
    kdr_block : float, optional
        Scaling factor applied to delayed-rectifier potassium channels.
    """

    celsius: float = 32.0
    temp_orig_dcn: float = 32.0
    q10_channel_gating: float = 3.0
    q10_conductances: float = 1.4
    q10_ca_conc: float = 2.0
    kdr_block: float = 1.0

    @property
    def qdt_channel_gating(self) -> float:
        return self.q10_channel_gating ** ((self.celsius - self.temp_orig_dcn) / 10.0)

    @property
    def qdt_conductances(self) -> float:
        return self.q10_conductances ** ((self.celsius - self.temp_orig_dcn) / 10.0)

    @property
    def qdt_ca_conc(self) -> float:
        return self.q10_ca_conc ** ((self.celsius - self.temp_orig_dcn) / 10.0)

    @property
    def temperature(self):
        return u.celsius2kelvin(self.celsius)


@dataclass(frozen=True)
class DcnTemplateParameters:
    """Parameters used to paint the DCN BrainCell template.

    Parameters
    ----------
    scales : DcnTemplateScales, optional
        Temperature and Q10 scaling values.

    Notes
    -----
    The remaining fields are plain source-template constants. Units are
    attached at paint sites so the numeric values stay close to the
    original HOC template.
    """

    scales: DcnTemplateScales = DcnTemplateScales()

    ra: float = 235.3
    cm: float = 1.57
    passcond: float = 2.81e-5
    passcondmyel_factor: float = 1.0 / 2.81
    shell_thick: float = 0.2

    sodium_rev_pot: float = 71.0
    potassium_rev_pot: float = -90.0
    h_rev_pot: float = -45.0
    calcium_co: float = 2.0
    calcium_ci: float = 50e-6

    g_na_f_soma: float = 2.5e-2
    g_na_p_soma: float = 2e-4
    g_fkdr_soma: float = 1.5e-2
    g_skdr_soma: float = 1.25e-2
    g_sk_soma: float = 2.2e-4
    perm_ca_lva_soma: float = 2.33 * 1.77e-5
    perm_ca_hva_soma: float = 7.5e-6
    g_h_soma: float = 0.5e-4
    g_tnc_soma: float = 3e-5
    g_tnc_ax: float = 3.5e-5

    def qconductance(self, value: float) -> float:
        return value * self.scales.qdt_conductances

    @property
    def pass_g(self):
        return self.qconductance(self.passcond) * (u.siemens / u.cm**2)

    @property
    def pass_g_myel(self):
        return self.qconductance(self.passcond) * self.passcondmyel_factor * (u.siemens / u.cm**2)

    @property
    def qdeltat(self) -> float:
        return self.scales.qdt_channel_gating


@dataclass(frozen=True)
class DcnCellBuild:
    """Container returned by :func:`build_dcn_cell`.

    Parameters
    ----------
    native : DcnMorphology
        Parsed native DCN morphology and source lookup metadata.
    cell : Cell
        BrainCell cell with template mechanisms painted.
    params : DcnTemplateParameters
        Parameter object used during painting.
    """

    native: DcnMorphology
    cell: Cell
    params: DcnTemplateParameters


def build_dcn_cell(
    *,
    native: DcnMorphology | None = None,
    params: DcnTemplateParameters | None = None,
) -> DcnCellBuild:
    """Build a BrainCell DCN cell from the native HOC morphology.

    Parameters
    ----------
    native : DcnMorphology or None, optional
        Pre-parsed native morphology. If omitted,
        :func:`load_dcn_morphology` is called and therefore requires
        ``DCN_SOURCE_HOC`` or an explicit source path at that layer.
    params : DcnTemplateParameters or None, optional
        Template parameters. Defaults to :class:`DcnTemplateParameters`.

    Returns
    -------
    DcnCellBuild
        Native morphology, painted cell, and parameters.
    """

    native = load_dcn_morphology() if native is None else native
    params = DcnTemplateParameters() if params is None else params
    cell = Cell(native.morpho, name="DCNCell")
    paint_dcn_template(cell, native=native, params=params)
    return DcnCellBuild(native=native, cell=cell, params=params)


def paint_dcn_template(
    cell: Cell,
    *,
    native: DcnMorphology,
    params: DcnTemplateParameters,
) -> None:
    """Paint DCN cable, ion, and channel mechanisms onto a cell.

    Parameters
    ----------
    cell : Cell
        BrainCell cell whose morphology matches ``native``.
    native : DcnMorphology
        Native DCN morphology metadata used to select physiological
        regions.
    params : DcnTemplateParameters
        Source-template constants and temperature scales.
    """

    _paint_cable(cell, native=native, params=params)
    _paint_ions(cell, native=native, params=params)
    _paint_channels(cell, native=native, params=params)


def _paint_cable(cell: Cell, *, native: DcnMorphology, params: DcnTemplateParameters) -> None:
    common = {
        "resting_potential": -65.0 * u.mV,
        "membrane_capacitance": params.cm * (u.uF / u.cm**2),
        "axial_resistivity": params.ra * (u.ohm * u.cm),
        "temperature": params.scales.temperature,
    }
    cell.paint(AllRegion(), mech.CableProperty(**common))
    cell.paint(
        dcn_region(native.morpho, "axNode"),
        mech.CableProperty(
            resting_potential=common["resting_potential"],
            membrane_capacitance=(params.cm / 100.0) * (u.uF / u.cm**2),
            axial_resistivity=common["axial_resistivity"],
            temperature=common["temperature"],
        ),
    )
    cell.paint(
        AllRegion(),
        mech.Channel("IL", name="IL", g_max=params.pass_g, E=common["resting_potential"]),
    )
    cell.paint(
        dcn_region(native.morpho, "axNode"),
        mech.Channel("IL", name="IL_axNode", g_max=params.pass_g_myel, E=common["resting_potential"]),
    )


def _paint_ions(cell: Cell, *, native: DcnMorphology, params: DcnTemplateParameters) -> None:
    del native
    cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=params.sodium_rev_pot * u.mV))
    cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=params.potassium_rev_pot * u.mV))
    cell.paint(
        AllRegion(),
        mech.Ion(
            "CdpHVA_SU2015_DCN",
            name="ca_hva",
            Co=params.calcium_co * u.mM,
            Ci_initializer=params.calcium_ci * u.mM,
            tauCa=(70.0 / params.scales.qdt_ca_conc) * u.ms,
            caiBase=params.calcium_ci * u.mM,
        ),
    )
    cell.paint(
        AllRegion(),
        mech.Ion(
            "CdpLVA_SU2015_DCN",
            name="ca_lva",
            Co=params.calcium_co * u.mM,
            Ci_initializer=params.calcium_ci * u.mM,
            tauCal=(70.0 / params.scales.qdt_ca_conc) * u.ms,
            caliBase=params.calcium_ci * u.mM,
        ),
    )


def _paint_channels(cell: Cell, *, native: DcnMorphology, params: DcnTemplateParameters) -> None:
    r = lambda name: dcn_region(native.morpho, name)
    q = params.qdeltat
    temp = params.scales.temperature

    soma = r("soma")
    ax_hill = r("axHillock")
    ax_is = r("axIniSeg")
    prox = r("proxDend")
    dist = r("distDend")

    g_na_soma = params.qconductance(params.g_na_f_soma)
    g_fkdr_soma = params.qconductance(params.g_fkdr_soma) * params.scales.kdr_block
    g_skdr_soma = params.qconductance(params.g_skdr_soma) * params.scales.kdr_block
    g_sk_soma = params.qconductance(params.g_sk_soma)
    perm_lva_soma = params.qconductance(params.perm_ca_lva_soma)
    perm_hva_soma = params.qconductance(params.perm_ca_hva_soma)
    g_h_soma = params.qconductance(params.g_h_soma)
    g_tnc_soma = params.qconductance(params.g_tnc_soma)

    _paint_channel(cell, soma, "NaF_SU2015_DCN", "NaF_soma", g_na_soma, "na", q)
    _paint_channel(cell, soma, "NaP_SU2015_DCN", "NaP_soma", params.qconductance(params.g_na_p_soma), "na", q)
    _paint_channel(cell, soma, "fKdr_SU2015_DCN", "fKdr_soma", g_fkdr_soma, "k", q)
    _paint_channel(cell, soma, "sKdr_SU2015_DCN", "sKdr_soma", g_skdr_soma, "k", q)
    _paint_channel(cell, soma, "SK_SU2015_DCN", "SK_soma", g_sk_soma, {"k": "k", "ca": "ca_hva"}, q)
    _paint_h(cell, soma, "HCN_soma", g_h_soma, params, q)
    _paint_tnc_il(cell, soma, "TNC_soma", g_tnc_soma)
    _paint_ca(cell, soma, "CaLVA_soma", "CaLVA_SU2015_DCN", perm_lva_soma, "ca_lva", temp, q)
    _paint_ca(cell, soma, "CaHVA_soma", "CaHVA_SU2015_DCN", perm_hva_soma, "ca_hva", temp, q)

    _paint_channel(cell, ax_hill, "NaF_SU2015_DCN", "NaF_axHillock", 2.0 * g_na_soma, "na", q)
    _paint_channel(cell, ax_hill, "fKdr_SU2015_DCN", "fKdr_axHillock", 2.0 * g_fkdr_soma, "k", q)
    _paint_channel(cell, ax_hill, "sKdr_SU2015_DCN", "sKdr_axHillock", 2.0 * g_skdr_soma, "k", q)
    _paint_tnc_il(cell, ax_hill, "TNC_axHillock", params.qconductance(params.g_tnc_ax))

    _paint_channel(cell, ax_is, "NaF_SU2015_DCN", "NaF_axIniSeg", 2.0 * g_na_soma, "na", q)
    _paint_channel(cell, ax_is, "fKdr_SU2015_DCN", "fKdr_axIniSeg", 2.0 * g_fkdr_soma, "k", q)
    _paint_channel(cell, ax_is, "sKdr_SU2015_DCN", "sKdr_axIniSeg", 2.0 * g_skdr_soma, "k", q)
    _paint_tnc_il(cell, ax_is, "TNC_axIniSeg", params.qconductance(params.g_tnc_ax))

    _paint_channel(cell, prox, "NaF_SU2015_DCN", "NaF_proxDend", 0.4 * g_na_soma, "na", q)
    _paint_channel(cell, prox, "fKdr_SU2015_DCN", "fKdr_proxDend", 0.6 * g_fkdr_soma, "k", q)
    _paint_channel(cell, prox, "sKdr_SU2015_DCN", "sKdr_proxDend", 0.6 * g_skdr_soma, "k", q)
    _paint_channel(cell, prox, "SK_SU2015_DCN", "SK_proxDend", 0.3 * g_sk_soma, {"k": "k", "ca": "ca_hva"}, q)
    _paint_h(cell, prox, "HCN_proxDend", 2.0 * g_h_soma, params, q)
    _paint_tnc_il(cell, prox, "TNC_proxDend", 0.2 * g_tnc_soma)
    _paint_ca(cell, prox, "CaLVA_proxDend", "CaLVA_SU2015_DCN", 2.0 * perm_lva_soma, "ca_lva", temp, q)
    _paint_ca(cell, prox, "CaHVA_proxDend", "CaHVA_SU2015_DCN", perm_hva_soma / 1.5, "ca_hva", temp, q)

    _paint_channel(cell, dist, "SK_SU2015_DCN", "SK_distDend", 0.3 * g_sk_soma, {"k": "k", "ca": "ca_hva"}, q)
    _paint_h(cell, dist, "HCN_distDend", 3.0 * g_h_soma, params, q)
    _paint_ca(cell, dist, "CaLVA_distDend", "CaLVA_SU2015_DCN", 2.0 * perm_lva_soma, "ca_lva", temp, q)
    _paint_ca(cell, dist, "CaHVA_distDend", "CaHVA_SU2015_DCN", perm_hva_soma / 1.5, "ca_hva", temp, q)


def _paint_channel(cell: Cell, region, class_name: str, name: str, g_s_cm2: float, ion_name, qdeltat: float | None = None) -> None:
    kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
    if qdeltat is not None and class_name in {"SK_SU2015_DCN"}:
        kwargs["qdeltat"] = qdeltat
    cell.paint(
        region,
        mech.Channel(
            class_name,
            name=name,
            g_max=g_s_cm2 * (u.siemens / u.cm**2),
            **kwargs,
        ),
    )


def _paint_h(cell: Cell, region, name: str, g_s_cm2: float, params: DcnTemplateParameters, qdeltat: float) -> None:
    del qdeltat
    cell.paint(
        region,
        mech.Channel(
            "HCN_SU2015_DCN",
            name=name,
            g_max=g_s_cm2 * (u.siemens / u.cm**2),
            E=params.h_rev_pot * u.mV,
        ),
    )


def _paint_tnc_il(cell: Cell, region, name: str, g_s_cm2: float) -> None:
    cell.paint(
        region,
        mech.Channel(
            "IL",
            name=name,
            g_max=g_s_cm2 * (u.siemens / u.cm**2),
            E=-35.0 * u.mV,
        ),
    )


def _paint_ca(
    cell: Cell,
    region,
    name: str,
    class_name: str,
    perm_cm_s: float,
    ion_name: str,
    temp,
    qdeltat: float,
) -> None:
    cell.paint(
        region,
        mech.Channel(
            class_name,
            name=name,
            perm=perm_cm_s * (u.cm / u.second),
            temp=temp,
            qdeltat=qdeltat,
            ion_name=ion_name,
        ),
    )
