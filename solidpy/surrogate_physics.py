"""surrogate_physics — Exportação de grandezas físicas estáticas para surrogate ML.

Este módulo existe SOMENTE na branch ``ai-surrogate``.  O branch ``main`` mantém
apenas a lógica de simulação ODE pura.

Objetivo:
    Fornecer ao projeto MotorTransformer todas as grandezas que precisam ser
    reimplementadas em PyTorch para construir as PINN-lite losses do surrogate.
    A regra é: cada equação aqui representa a "verdade física" que o surrogate
    deve espelhar.  Se o SolidPy mudar uma equação, este módulo muda junto —
    e os loss terms do surrogate devem ser atualizados na sequência.

Uso típico no pipeline ML:
    from solidpy.surrogate_physics import compute_static_features, compute_burn_area_curve

    feats = compute_static_features(grain, motor, propellant, eta_c=design.isp_efficiency)
    ab_curve = compute_burn_area_curve(grain, n_points=64)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from .Burn import Burn
    from .Grain import Grain
    from .Motor import Motor
    from .Propellant import Propellant
except ImportError:
    from Burn import Burn
    from Grain import Grain
    from Motor import Motor
    from Propellant import Propellant


# ---------------------------------------------------------------------------
# Dataclasses de saída
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateStaticFeatures:
    """Grandezas escalares calculáveis a partir do design ANTES da ODE.

    Todos os valores estão em unidades SI.  Este dataclass é a "planta" do que
    o módulo ``physics/analytical.py`` do MotorTransformer deve reimplementar
    em PyTorch para os PINN-lite loss terms.

    Atributos
    ---------
    c_star_m_s:
        Velocidade característica c* [m/s].
        Derivada de: c* = sqrt(R_sp * T0 / γ) × ((γ+1)/2)^((γ+1)/(2(γ-1))).
        Fonte: Sutton & Biblarz §2.2.

    kn_initial:
        Razão Kn inicial = A_b_initial / A_throat [-].
        Governada pela pressão de câmara de equilíbrio: P_eq ~ (ρ*Kn*r*c*).

    Cf_ref:
        Coeficiente de empuxo à pressão de referência P_ref_pa [-].
        Inclui fator de divergência λ = (1+cos α)/2 (Sutton §3.4).
        Fonte: evaluate_Cf() de Burn.py.

    lambda_divergence:
        Fator de perda de divergência da tubeira cônica λ = (1+cos α)/2 [-].
        Para bocal de Bell ou ângulo não especificado: λ = 1.0.

    isp_theory_s:
        Isp teórico isentrópico [s] = Cf_ref * c_star / g0.
        Sem perdas de eficiência; base para validação.

    isp_effective_s:
        Isp efetivo real [s] = eta_c * Cf_ref * c_star / g0.
        Este é o Isp que o SolidPy entrega com eta_c aplicado.

    eta_c:
        Eficiência de combustão [-] passada ao Burn().
        Corresponde ao feature alpha.isp_efficiency do vetor de design.

    propellant_mass_kg:
        Massa total de propelente [kg] = ρ_p * V_grão.

    burn_area_initial_m2:
        Área de queima inicial A_b [m²] do grão não-regredido.

    expansion_ratio:
        Razão de expansão ε = (D_exit/D_throat)² [-].

    gamma:
        Razão de calores específicos γ do propelente (à pressão de referência).

    P_ref_pa:
        Pressão de câmara de referência usada para Cf e Isp [Pa].
    """
    c_star_m_s: float
    kn_initial: float
    Cf_ref: float
    lambda_divergence: float
    isp_theory_s: float
    isp_effective_s: float
    eta_c: float
    propellant_mass_kg: float
    burn_area_initial_m2: float
    expansion_ratio: float
    gamma: float
    P_ref_pa: float


@dataclass(frozen=True)
class BurnAreaCurve:
    """Curva A_b(w) — área de queima em função da regressão da teia.

    Permite ao surrogate aprender a evolução temporal da área de queima
    sem integrar a ODE inteira.  É a 'assinatura geométrica' do grão.

    Atributos
    ---------
    web_fraction:
        Array normalizado de profundidade de regressão w/W ∈ [0, 1].
        W = espessura total da teia do grão.

    burn_area_m2:
        Array de área de queima A_b [m²] correspondente a cada w/W.

    web_thickness_m:
        Espessura total da teia W [m] = r_outer - r_inner (tubular/star inicial).

    n_grains:
        Número de grãos empilhados (área total = n_grains × área de 1 grão).

    geometry:
        String identifier: 'tubular' ou 'star'.
    """
    web_fraction: np.ndarray
    burn_area_m2: np.ndarray
    web_thickness_m: float
    n_grains: int
    geometry: str


# ---------------------------------------------------------------------------
# Função principal: grandezas estáticas
# ---------------------------------------------------------------------------

_G0 = 9.80665  # m/s² — gravidade padrão (NIST)


def compute_static_features(
    grain: Grain,
    motor: Motor,
    propellant: Propellant,
    *,
    eta_c: float = 1.0,
    P_ref_pa: float = 3.5e6,
) -> SurrogateStaticFeatures:
    """Calcula grandezas físicas estáticas a partir do design, sem ODE.

    Estas são as equações que o módulo ``physics/analytical.py`` do
    MotorTransformer DEVE reimplementar em PyTorch para as PINN-lite losses.
    Qualquer mudança nesta função deve ser espelhada lá.

    Parâmetros
    ----------
    grain:
        Grão na configuração inicial (sem regressão).
    motor:
        Motor com geometria de tubeira definida.
    propellant:
        Propelente com termoquímica definida.
    eta_c:
        Eficiência de combustão.  Passar aqui o valor de ``alpha.isp_efficiency``
        do vetor de design do MotorTransformer.
    P_ref_pa:
        Pressão de câmara de referência para avaliação de Cf e Isp.
        Default 3,5 MPa (pressão típica de operação nominal).

    Retorna
    -------
    SurrogateStaticFeatures
    """
    eta_c = float(eta_c)
    P_ref_pa = float(P_ref_pa)

    # c* isentrópico (equação analítica fechada — Sutton §2.2)
    # c* = sqrt(R_sp * T0 / γ) × ((γ+1)/2)^((γ+1)/(2(γ-1)))
    gamma = float(propellant.get_gamma(P_ref_pa))
    R_sp = float(propellant.products_constant)
    T0 = float(propellant.Tc_at_pressure(P_ref_pa))
    c_star = (
        math.sqrt(R_sp * T0 / gamma)
        * ((gamma + 1.0) / 2.0) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )

    # Burn temporário — só para acessar evaluate_Cf e lambda
    burn = Burn(grain, motor, propellant, eta_c=eta_c)

    # Cf à pressão de referência (inclui λ_div internamente)
    Cf_ref = float(burn.evaluate_Cf(P_ref_pa))

    # λ_div — exportado explicitamente para reimplementação em PyTorch
    lambda_div = float(burn._nozzle_divergence_factor())

    # Isp teórico e efetivo
    isp_theory = Cf_ref * c_star / _G0
    isp_effective = eta_c * isp_theory

    # Kn inicial
    A_t = float(motor.nozzle_throat_area)
    A_b_initial = float(grain.burn_area)
    kn_initial = A_b_initial / A_t if A_t > 0 else 0.0

    # Massa de propelente
    rho_p = float(propellant.density) if propellant.density is not None else 0.0
    propellant_mass = rho_p * float(grain.volume)

    # Razão de expansão
    expansion_ratio = float(motor.expansion_ratio)

    return SurrogateStaticFeatures(
        c_star_m_s=c_star,
        kn_initial=kn_initial,
        Cf_ref=Cf_ref,
        lambda_divergence=lambda_div,
        isp_theory_s=isp_theory,
        isp_effective_s=isp_effective,
        eta_c=eta_c,
        propellant_mass_kg=propellant_mass,
        burn_area_initial_m2=A_b_initial,
        expansion_ratio=expansion_ratio,
        gamma=gamma,
        P_ref_pa=P_ref_pa,
    )


# ---------------------------------------------------------------------------
# Curva A_b(w) — assinatura geométrica do grão
# ---------------------------------------------------------------------------

def compute_burn_area_curve(
    grain: Grain,
    *,
    n_points: int = 64,
) -> BurnAreaCurve:
    """Gera a curva A_b(w/W) para o grão fornecido.

    Esta curva é a 'assinatura geométrica' que distingue tubular de star de
    hetero.  O surrogate pode usá-la como feature de entrada ou como alvo
    de uma loss de forma.  No MotorTransformer, ela não precisa de reimplementação
    em PyTorch — é usada como feature pré-computada no dataset.

    A regressão w varre de 0 até a espessura total da teia W = r_outer - r_inner.
    Para grãos star, o slot pode atingir a parede antes de w = W; a área retorna
    0 no burnout (conforme a geometria do SolidPy).

    Parâmetros
    ----------
    grain:
        Grão na configuração inicial.
    n_points:
        Número de pontos na curva (recomendado: 64 para surrogate, 512 para viz).

    Retorna
    -------
    BurnAreaCurve
    """
    web_thickness = float(grain.outer_radius - grain.initial_inner_radius)
    if web_thickness <= 0.0:
        empty = np.zeros(n_points)
        return BurnAreaCurve(
            web_fraction=empty,
            burn_area_m2=empty,
            web_thickness_m=0.0,
            n_grains=1,
            geometry=grain.geometry,
        )

    w_vals = np.linspace(0.0, web_thickness, n_points)
    ab_vals = np.array([
        float(grain.evaluate_burn_area(float(w), update_state=False))
        for w in w_vals
    ])
    w_frac = w_vals / web_thickness

    return BurnAreaCurve(
        web_fraction=w_frac,
        burn_area_m2=ab_vals,
        web_thickness_m=web_thickness,
        n_grains=1,
        geometry=grain.geometry,
    )


# ---------------------------------------------------------------------------
# Helper: converter SurrogateStaticFeatures → dict para salvar no dataset
# ---------------------------------------------------------------------------

def static_features_to_dict(feats: SurrogateStaticFeatures) -> dict[str, float]:
    """Converte SurrogateStaticFeatures para dict plano — prontas para parquet."""
    return {
        "surrogate.c_star_m_s": feats.c_star_m_s,
        "surrogate.kn_initial": feats.kn_initial,
        "surrogate.Cf_ref": feats.Cf_ref,
        "surrogate.lambda_divergence": feats.lambda_divergence,
        "surrogate.isp_theory_s": feats.isp_theory_s,
        "surrogate.isp_effective_s": feats.isp_effective_s,
        "surrogate.eta_c": feats.eta_c,
        "surrogate.propellant_mass_kg": feats.propellant_mass_kg,
        "surrogate.burn_area_initial_m2": feats.burn_area_initial_m2,
        "surrogate.expansion_ratio": feats.expansion_ratio,
        "surrogate.gamma": feats.gamma,
        "surrogate.P_ref_pa": feats.P_ref_pa,
    }
