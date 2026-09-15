"""surrogate_physics — Exportação de grandezas físicas estáticas para surrogate ML.

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
from typing import Optional, Union

import numpy as np

StructuralFeatureValue = Union[float, np.ndarray]
StructuralFeatureDictValue = Union[float, np.ndarray, list]

try:
    from .Burn import Burn
    from .Grain import Grain
    from .Motor import Motor
    from .Multiphysics import (
        CasingMaterial,
        NozzleMaterial,
        _casing_mass_with_bulkheads_kg,
        _casing_mass_with_bulkheads_kg_vectorized,
        _liner_mass_kg,
        _liner_mass_kg_vectorized,
        _nozzle_mass_kg,
        _nozzle_mass_kg_vectorized,
        _vector_broadcast_float_arrays,
        _validate_vector_finite,
        _validate_vector_nonnegative,
        _validate_vector_positive,
        _vector_result,
    )
    from .Propellant import Propellant
except ImportError:
    from Burn import Burn
    from Grain import Grain
    from Motor import Motor
    from Multiphysics import (
        CasingMaterial,
        NozzleMaterial,
        _casing_mass_with_bulkheads_kg,
        _casing_mass_with_bulkheads_kg_vectorized,
        _liner_mass_kg,
        _liner_mass_kg_vectorized,
        _nozzle_mass_kg,
        _nozzle_mass_kg_vectorized,
        _vector_broadcast_float_arrays,
        _validate_vector_finite,
        _validate_vector_nonnegative,
        _validate_vector_positive,
        _vector_result,
    )
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
        Massa total de propelente [kg] = ρ_p * V_propellant, onde
        ``V_propellant`` é a soma dos volumes de todos os grãos do motor.
        Para motores legados sem ``grains``/``propellant_volume``, usa o
        volume de ``grain`` como fallback.

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
class SurrogateStructuralFeatures:
    """Grandezas estruturais e de massa calculáveis sem integrar uma ODE.

    Todos os valores estão em unidades SI. As fórmulas são as formas fechadas
    que o surrogate externo deve reimplementar para manter consistência com o
    SolidPy.

    Atributos
    ---------
    casing_mass_kg:
        Massa da casca cilíndrica mais duas tampas planas (bulkheads), com a
        espessura das tampas escalada por ``bulkhead_fraction``.
        Fonte: ``Multiphysics.py::geometry_from_components``.

    liner_mass_kg:
        Massa do liner cilíndrico, ``π (r_i²-r_l²) L ρ_l``; zero quando a
        espessura do liner é não positiva.
        Fonte: campos de liner de ``Multiphysics.py::CasingMaterial``.

    nozzle_mass_kg:
        Massa aproximada da parede cônica convergente e divergente. Cada
        contribuição é ``A_lateral * wall_thickness * density``. O
        convergente usa o semiângulo fixo de 45°; o divergente usa
        ``motor.nozzle_angle`` como semiângulo, exclusivamente para esta
        aproximação geométrica. O ângulo divergente é obrigatório para a
        estimativa de massa e valores ausentes são rejeitados.
        Fonte: raios/ângulo de ``Motor.py`` e ``NozzleMaterial``.

    dry_mass_kg:
        ``casing_mass_kg + liner_mass_kg + nozzle_mass_kg``.
        Fonte: composição dos componentes de ``Multiphysics.py``.

    motor_initial_mass_kg:
        ``dry_mass_kg + propellant_mass_kg``.
        Fonte: convenção de massa de ``Multiphysics.py::MotorGeometry``.

    motor_final_mass_kg:
        Massa seca após a queima completa; igual a ``dry_mass_kg``.
        Fonte: ``Multiphysics.py::geometry_from_components``.

    structural_mass_ratio:
        ``dry_mass_kg / motor_initial_mass_kg``.
        Fonte: razão estrutural definida no plano de surrogate.

    port_throat_ratio:
        ``grain.evaluate_port_area(0) / motor.nozzle_throat_area``.
        Fonte: ``Grain.py::evaluate_port_area`` e ``Motor.py``.

    von_mises_at_reference_pa:
        Tensão equivalente de von Mises triaxial na parede interna para a
        pressão de referência, usando Lamé quando ``t/r > 0.1`` e Barlow
        quando não. Fonte: ``Multiphysics.py::simulate_structural_response``.

    burst_pressure_pa:
        ``(2/√3) Su ln(r_o/r_i)`` pelo critério de Tresca.
        Fonte: ``Multiphysics.py::simulate_structural_response``.

    burst_safety_factor_at_reference_pa:
        ``burst_pressure_pa / max(chamber_pressure_pa, 1)``.
        Fonte: ``Multiphysics.py::simulate_structural_response``.
    """
    casing_mass_kg: StructuralFeatureValue
    liner_mass_kg: StructuralFeatureValue
    nozzle_mass_kg: StructuralFeatureValue
    dry_mass_kg: StructuralFeatureValue
    motor_initial_mass_kg: StructuralFeatureValue
    motor_final_mass_kg: StructuralFeatureValue
    structural_mass_ratio: StructuralFeatureValue
    port_throat_ratio: StructuralFeatureValue
    von_mises_at_reference_pa: StructuralFeatureValue
    burst_pressure_pa: StructuralFeatureValue
    burst_safety_factor_at_reference_pa: StructuralFeatureValue


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

    A massa de propelente usa ``ρ_p`` vezes o volume total, somando os volumes
    de ``motor.grains``. ``motor.propellant_volume`` é usado quando essa lista
    não está disponível; ``grain.volume`` fica reservado como fallback para
    motores legados.

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

    # Massa de propelente. Uma densidade ausente ou não finita não representa
    # uma carga conhecida; nunca transforme esse caso silenciosamente em zero.
    if propellant.density is None:
        raise ValueError(
            "massa de propelente desconhecida: propellant.density não foi definido"
        )
    try:
        rho_p = float(propellant.density)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "massa de propelente desconhecida: propellant.density "
            "deve ser finita e maior que zero"
        ) from exc
    if not math.isfinite(rho_p) or rho_p <= 0.0:
        raise ValueError(
            "massa de propelente desconhecida: propellant.density "
            "deve ser finita e maior que zero"
        )
    motor_grains = getattr(motor, "grains", None)
    if motor_grains:
        propellant_volume_m3 = 0.0
        for index, motor_grain in enumerate(motor_grains):
            try:
                grain_volume_m3 = float(motor_grain.volume)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "grain.volume deve ser finito e maior ou igual a zero "
                    f"(motor.grains[{index}])"
                ) from exc
            if not math.isfinite(grain_volume_m3) or grain_volume_m3 < 0.0:
                raise ValueError(
                    "grain.volume deve ser finito e maior ou igual a zero "
                    f"(motor.grains[{index}])"
                )
            try:
                propellant_volume_m3 += grain_volume_m3
            except OverflowError as exc:
                raise ValueError(
                    "propellant_volume_m3 excede o intervalo numérico"
                ) from exc
            if not math.isfinite(propellant_volume_m3):
                raise ValueError("propellant_volume_m3 deve ser finito")
    elif getattr(motor, "propellant_volume", None) is not None:
        try:
            propellant_volume_m3 = float(motor.propellant_volume)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("motor.propellant_volume deve ser finito") from exc
        if not math.isfinite(propellant_volume_m3) or propellant_volume_m3 < 0.0:
            raise ValueError(
                "motor.propellant_volume deve ser finito e maior ou igual a zero"
            )
    else:
        try:
            propellant_volume_m3 = float(grain.volume)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "grain.volume deve ser finito e maior ou igual a zero"
            ) from exc
        if not math.isfinite(propellant_volume_m3) or propellant_volume_m3 < 0.0:
            raise ValueError("grain.volume deve ser finito e maior ou igual a zero")

    try:
        propellant_mass = rho_p * propellant_volume_m3
    except OverflowError as exc:
        raise ValueError(
            "propellant_mass_kg excede o intervalo numérico"
        ) from exc
    if not math.isfinite(propellant_mass):
        raise ValueError(
            "propellant_mass_kg não é finita: o produto de "
            "propellant.density pelo volume total do motor excede o intervalo numérico"
        )

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
# Massa e grandezas estruturais fechadas
# ---------------------------------------------------------------------------

def _finite_value(name: str, value: float) -> float:
    """Converte um escalar para float e rejeita ``NaN``/infinito."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} deve ser um número finito") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser um número finito")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    value = _finite_value(name, value)
    if value < 0.0:
        raise ValueError(f"{name} deve ser maior ou igual a zero")
    return value


def _finite_positive(name: str, value: float) -> float:
    value = _finite_value(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} deve ser maior que zero")
    return value


def _cone_half_angle(name: str, value: float) -> float:
    value = _finite_value(name, value)
    if not 0.0 < value < math.pi / 2.0:
        raise ValueError(f"{name} deve estar no intervalo aberto (0, pi/2)")
    return value


def _infer_propellant_mass_kg(motor: Motor) -> float:
    """Infere massa de propelente apenas quando o grão a armazena.

    ``Motor`` não guarda densidade de propelente. Portanto, sem o kwarg
    ``propellant_mass_kg``, esta função usa ``Grain.mass`` ou
    ``Grain.density`` quando disponíveis. Se algum grão não tiver nenhuma
    dessas fontes, a massa total não é conhecida e a operação é rejeitada.
    """
    mass_kg = 0.0
    for index, grain in enumerate(motor.grains):
        if grain.mass is not None:
            mass_kg += _finite_nonnegative(
                f"motor.grains[{index}].mass", grain.mass
            )
        elif grain.density is not None:
            density = _finite_positive(
                f"motor.grains[{index}].density", grain.density
            )
            volume = _finite_positive(
                f"motor.grains[{index}].volume", grain.volume
            )
            mass_kg += density * volume
        else:
            raise ValueError(
                "massa de propelente desconhecida: forneça "
                "propellant_mass_kg ou defina Grain.mass/Grain.density "
                f"para motor.grains[{index}]"
            )
    return mass_kg


def compute_structural_features(
    motor: Motor,
    casing_material: Optional[CasingMaterial] = None,
    nozzle_material: Optional[NozzleMaterial] = None,
    *,
    chamber_pressure_pa: float,
    casing_wall_thickness_m: float = 0.005,
    grain: Optional[Grain] = None,
    propellant_mass_kg: Optional[float] = None,
    casing_strength_factor: float = 1.0,
) -> SurrogateStructuralFeatures:
    """Calcula massa e razões estruturais sem ``solve_ivp``.

    ``casing_wall_thickness_m`` é explícito porque nem ``Motor`` nem
    ``CasingMaterial`` possuem espessura de casing. O padrão de 5 mm é apenas
    uma convenção de avaliação. Após validar finitude, a espessura segue a
    convenção canônica de ``Multiphysics.py``: ``max(valor, 1e-5)``. Para
    comparar com ``geometry_from_components``, passe a mesma espessura.

    ``propellant_mass_kg`` pode receber o valor calculado por
    :func:`compute_static_features`. Se omitido, é inferido de ``Grain.mass``
    ou ``Grain.density``; isso evita inventar uma densidade que não existe em
    ``Motor``. ``grain`` controla o cálculo de área do port e, por padrão, é o
    primeiro grão do motor.

    Parâmetros
    ----------
    motor:
        Motor com raios de câmara, garganta e saída definidos.
    casing_material:
        Material do casing e do liner. Usa ``CasingMaterial()`` por padrão.
    nozzle_material:
        Material da parede da tubeira. Usa ``NozzleMaterial()`` por padrão.
    chamber_pressure_pa:
        Pressão de câmara de referência [Pa] para a tensão e o fator de burst.
    casing_wall_thickness_m:
        Espessura radial do casing [m].
    grain:
        Grão cuja área de port inicial será usada.
    propellant_mass_kg:
        Massa total de propelente [kg], quando conhecida externamente.
    casing_strength_factor:
        Fator multiplicativo de resistência, com a mesma convenção da análise
        estrutural transiente: após validar finitude, usa
        ``max(valor, 0.01)``.
    """
    casing_material = CasingMaterial() if casing_material is None else casing_material
    nozzle_material = NozzleMaterial() if nozzle_material is None else nozzle_material
    grain = grain or motor.grain

    pressure_pa = _finite_nonnegative("chamber_pressure_pa", chamber_pressure_pa)
    casing_strength_factor = max(
        _finite_value("casing_strength_factor", casing_strength_factor), 0.01
    )
    chamber_area_m2 = _finite_positive("motor.chamber_area", motor.chamber_area)
    throat_area_m2 = _finite_positive(
        "motor.nozzle_throat_area", motor.nozzle_throat_area
    )
    exit_area_m2 = _finite_positive("motor.nozzle_exit_area", motor.nozzle_exit_area)
    chamber_length_m = _finite_positive(
        "motor.chamber_length", motor.chamber_length
    )

    _finite_nonnegative(
        "casing_material.density_kg_m3", casing_material.density_kg_m3
    )
    # O caminho térmico canônico trata espessuras não positivas como liner
    # inativo. Valores não finitos continuam inválidos, mas uma espessura
    # negativa não deve exigir densidade nem gerar massa de liner.
    liner_thickness_m = _finite_value(
        "casing_material.liner_thickness_m", casing_material.liner_thickness_m
    )
    _finite_positive(
        "casing_material.yield_strength_mpa", casing_material.yield_strength_mpa
    )
    ultimate_strength_mpa = _finite_positive(
        "casing_material.resolved_ultimate_strength_mpa",
        casing_material.resolved_ultimate_strength_mpa,
    )
    if casing_material.allowable_stress_mpa is not None:
        _finite_positive(
            "casing_material.allowable_stress_mpa",
            casing_material.allowable_stress_mpa,
        )
    bulkhead_fraction = _finite_value(
        "casing_material.bulkhead_fraction", casing_material.bulkhead_fraction
    )
    if bulkhead_fraction <= 0.0:
        raise ValueError("casing_material.bulkhead_fraction deve ser maior que zero")
    if liner_thickness_m > 0.0:
        _finite_positive(
            "casing_material.liner_density_kg_m3",
            casing_material.liner_density_kg_m3,
        )
    nozzle_density_kg_m3 = _finite_positive(
        "nozzle_material.density_kg_m3", nozzle_material.density_kg_m3
    )
    wall_thickness_factor = _finite_positive(
        "nozzle_material.wall_thickness_factor",
        nozzle_material.wall_thickness_factor,
    )
    min_wall_thickness_m = _finite_nonnegative(
        "nozzle_material.min_wall_thickness_m",
        nozzle_material.min_wall_thickness_m,
    )

    wall_m = max(_finite_value("casing_wall_thickness_m", casing_wall_thickness_m), 1e-5)
    chamber_radius_m = math.sqrt(chamber_area_m2 / math.pi)
    throat_radius_m = math.sqrt(throat_area_m2 / math.pi)
    exit_radius_m = math.sqrt(exit_area_m2 / math.pi)
    if chamber_radius_m <= throat_radius_m:
        raise ValueError("o raio interno da câmara deve ser maior que o da garganta")
    if exit_radius_m <= throat_radius_m:
        raise ValueError("o raio de saída deve ser maior que o da garganta")
    if liner_thickness_m > 0.0 and liner_thickness_m >= chamber_radius_m:
        raise ValueError("a espessura do liner deve ser menor que o raio da câmara")

    divergent_half_angle_rad = _cone_half_angle(
        "motor.nozzle_angle", motor.nozzle_angle
    )
    try:
        casing_mass_kg = _casing_mass_with_bulkheads_kg(
            chamber_radius_m,
            wall_m,
            chamber_length_m,
            casing_material,
        )
        liner_mass_kg = _liner_mass_kg(
            chamber_radius_m,
            chamber_length_m,
            casing_material,
        )
        nozzle_mass_kg = _nozzle_mass_kg(
            chamber_radius_m,
            throat_radius_m,
            exit_radius_m,
            divergent_half_angle_rad,
            wall_m,
            nozzle_material,
        )
    except OverflowError as exc:
        raise ValueError(
            "cálculo da massa do casing, liner ou tubeira excede o intervalo numérico"
        ) from exc
    for name, value in (
        ("cálculo da massa do casing", casing_mass_kg),
        ("cálculo da massa do liner", liner_mass_kg),
        ("cálculo da massa da tubeira", nozzle_mass_kg),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} não é finito")

    try:
        dry_mass_kg = casing_mass_kg + liner_mass_kg + nozzle_mass_kg
    except OverflowError as exc:
        raise ValueError(
            "cálculo da massa seca excede o intervalo numérico"
        ) from exc
    if not math.isfinite(dry_mass_kg):
        raise ValueError("cálculo da massa seca não é finito")
    if propellant_mass_kg is None:
        propellant_mass_kg = _infer_propellant_mass_kg(motor)
    propellant_mass_kg = _finite_nonnegative(
        "propellant_mass_kg", propellant_mass_kg
    )
    try:
        motor_initial_mass_kg = dry_mass_kg + propellant_mass_kg
    except OverflowError as exc:
        raise ValueError(
            "cálculo da massa inicial excede o intervalo numérico"
        ) from exc
    if not math.isfinite(motor_initial_mass_kg):
        raise ValueError("cálculo da massa inicial não é finito")
    motor_final_mass_kg = dry_mass_kg
    structural_mass_ratio = (
        dry_mass_kg / motor_initial_mass_kg if motor_initial_mass_kg > 0.0 else 0.0
    )

    port_area_m2 = _finite_nonnegative(
        "grain.evaluate_port_area(0)", grain.evaluate_port_area(0.0)
    )
    port_throat_ratio = port_area_m2 / throat_area_m2

    # Keep this branch identical to simulate_structural_response: Lamé is
    # selected for t/r > 0.1, otherwise the existing Barlow approximation is.
    inner_radius_m = max(chamber_radius_m, 1e-5)
    outer_radius_m = inner_radius_m + wall_m
    inner_radius_sq = inner_radius_m**2
    outer_radius_sq = outer_radius_m**2
    if wall_m / max(inner_radius_m, 1e-9) > 0.1:
        hoop_pa = (
            pressure_pa
            * inner_radius_sq
            * (outer_radius_sq + inner_radius_sq)
            / max(outer_radius_sq - inner_radius_sq, 1e-9)
            / max(inner_radius_sq, 1e-9)
        )
        axial_pa = pressure_pa * inner_radius_sq / max(
            outer_radius_sq - inner_radius_sq, 1e-9
        )
    else:
        hoop_pa = pressure_pa * inner_radius_m / wall_m
        axial_pa = pressure_pa * inner_radius_m / (2.0 * wall_m)
    radial_pa = -pressure_pa
    try:
        stress_invariant_pa2 = 0.5 * (
            (hoop_pa - radial_pa) ** 2
            + (radial_pa - axial_pa) ** 2
            + (axial_pa - hoop_pa) ** 2
        )
        if not all(
            math.isfinite(value)
            for value in (hoop_pa, axial_pa, radial_pa, stress_invariant_pa2)
        ):
            raise ValueError("cálculo da tensão de von Mises não é finito")
        von_mises_pa = math.sqrt(max(stress_invariant_pa2, 0.0))
    except OverflowError as exc:
        raise ValueError(
            "cálculo da tensão de von Mises excede o intervalo numérico"
        ) from exc
    if not math.isfinite(von_mises_pa):
        raise ValueError("cálculo da tensão de von Mises não é finito")

    # Keep the same material source and lower-bound convention as the
    # transient structural path. Finiteness is rejected, while finite values
    # below the canonical minimum are clamped above.
    try:
        ultimate_pa = ultimate_strength_mpa * 1e6 * casing_strength_factor
        burst_pressure_pa = (
            (2.0 / math.sqrt(3.0))
            * ultimate_pa
            * math.log(max(outer_radius_m / max(inner_radius_m, 1e-9), 1.0))
        )
        burst_safety_factor = burst_pressure_pa / max(pressure_pa, 1.0)
    except OverflowError as exc:
        raise ValueError(
            "cálculo da pressão de burst excede o intervalo numérico"
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (ultimate_pa, burst_pressure_pa, burst_safety_factor)
    ):
        raise ValueError("cálculo da pressão de burst não é finito")

    # Finite inputs can still overflow during multiplication/squaring. Keep
    # the public result free of NaN/inf instead of returning a contaminated
    # feature vector.
    casing_mass_kg = _finite_nonnegative("casing_mass_kg", casing_mass_kg)
    liner_mass_kg = _finite_nonnegative("liner_mass_kg", liner_mass_kg)
    nozzle_mass_kg = _finite_nonnegative("nozzle_mass_kg", nozzle_mass_kg)
    dry_mass_kg = _finite_nonnegative("dry_mass_kg", dry_mass_kg)
    motor_initial_mass_kg = _finite_positive(
        "motor_initial_mass_kg", motor_initial_mass_kg
    )
    motor_final_mass_kg = _finite_nonnegative(
        "motor_final_mass_kg", motor_final_mass_kg
    )
    structural_mass_ratio = _finite_nonnegative(
        "structural_mass_ratio", structural_mass_ratio
    )
    port_throat_ratio = _finite_nonnegative("port_throat_ratio", port_throat_ratio)
    von_mises_pa = _finite_nonnegative("von_mises_at_reference_pa", von_mises_pa)
    burst_pressure_pa = _finite_nonnegative("burst_pressure_pa", burst_pressure_pa)
    burst_safety_factor = _finite_nonnegative(
        "burst_safety_factor_at_reference_pa", burst_safety_factor
    )

    return SurrogateStructuralFeatures(
        casing_mass_kg=casing_mass_kg,
        liner_mass_kg=liner_mass_kg,
        nozzle_mass_kg=nozzle_mass_kg,
        dry_mass_kg=dry_mass_kg,
        motor_initial_mass_kg=motor_initial_mass_kg,
        motor_final_mass_kg=motor_final_mass_kg,
        structural_mass_ratio=structural_mass_ratio,
        port_throat_ratio=port_throat_ratio,
        von_mises_at_reference_pa=von_mises_pa,
        burst_pressure_pa=burst_pressure_pa,
        burst_safety_factor_at_reference_pa=burst_safety_factor,
    )


def compute_structural_features_vectorized(
    *,
    chamber_radius_m: StructuralFeatureValue,
    throat_radius_m: StructuralFeatureValue,
    exit_radius_m: StructuralFeatureValue,
    chamber_length_m: StructuralFeatureValue,
    casing_wall_thickness_m: StructuralFeatureValue,
    casing_density_kg_m3: StructuralFeatureValue,
    bulkhead_fraction: StructuralFeatureValue,
    liner_thickness_m: StructuralFeatureValue,
    liner_density_kg_m3: StructuralFeatureValue,
    nozzle_density_kg_m3: StructuralFeatureValue,
    nozzle_wall_thickness_factor: StructuralFeatureValue,
    nozzle_min_wall_thickness_m: StructuralFeatureValue,
    divergent_half_angle_rad: StructuralFeatureValue,
    chamber_pressure_pa: StructuralFeatureValue,
    port_area_m2: StructuralFeatureValue,
    propellant_mass_kg: StructuralFeatureValue,
    ultimate_strength_mpa: StructuralFeatureValue,
    casing_strength_factor: StructuralFeatureValue = 1.0,
) -> SurrogateStructuralFeatures:
    """Calcula features estruturais para um lote com broadcasting NumPy.

    Todos os argumentos aceitam escalares ou ``numpy.ndarray`` broadcastable;
    a conversão interna usa ``float64``. O retorno é um
    :class:`SurrogateStructuralFeatures` cujos onze campos são arrays NumPy
    com o shape broadcastado; entradas escalares produzem arrays 0-D. As
    fórmulas, clamps e validações físicas correspondem ao caminho escalar
    :func:`compute_structural_features`, sem integração de ODE. O liner é
    validado somente nas posições com espessura positiva, e um bocal com raio
    de saída igual ao da garganta continua sendo rejeitado neste caminho
    superior, tal como no caminho escalar.
    """
    if divergent_half_angle_rad is None:
        raise ValueError(
            "divergent_half_angle_rad deve ser finito e estar em (0, pi/2)"
        )
    (
        chamber_radius,
        throat_radius,
        exit_radius,
        chamber_length,
        casing_wall,
        casing_density,
        bulkhead_fraction,
        liner_thickness,
        liner_density,
        nozzle_density,
        nozzle_wall_factor,
        nozzle_min_wall,
        divergent_angle,
        chamber_pressure,
        port_area,
        propellant_mass,
        ultimate_strength,
        strength_factor,
    ) = _vector_broadcast_float_arrays(
        chamber_radius_m,
        throat_radius_m,
        exit_radius_m,
        chamber_length_m,
        casing_wall_thickness_m,
        casing_density_kg_m3,
        bulkhead_fraction,
        liner_thickness_m,
        liner_density_kg_m3,
        nozzle_density_kg_m3,
        nozzle_wall_thickness_factor,
        nozzle_min_wall_thickness_m,
        divergent_half_angle_rad,
        chamber_pressure_pa,
        port_area_m2,
        propellant_mass_kg,
        ultimate_strength_mpa,
        casing_strength_factor,
    )

    _validate_vector_positive("chamber_radius_m", chamber_radius)
    _validate_vector_positive("throat_radius_m", throat_radius)
    _validate_vector_positive("exit_radius_m", exit_radius)
    if np.any(chamber_radius <= throat_radius):
        raise ValueError("chamber_radius_m deve ser maior que throat_radius_m")
    if np.any(exit_radius <= throat_radius):
        raise ValueError("exit_radius_m deve ser maior que throat_radius_m")
    _validate_vector_positive("chamber_length_m", chamber_length)
    _validate_vector_finite("casing_wall_thickness_m", casing_wall)
    _validate_vector_nonnegative("casing_density_kg_m3", casing_density)
    _validate_vector_positive("bulkhead_fraction", bulkhead_fraction)
    _validate_vector_finite("liner_thickness_m", liner_thickness)
    liner_active = liner_thickness > 0.0
    if np.any(liner_active & (liner_thickness >= chamber_radius)):
        raise ValueError("liner_thickness_m deve ser menor que chamber_radius_m")
    if np.any(liner_active & ~np.isfinite(liner_density)):
        raise ValueError(
            "liner_density_kg_m3 deve conter valores finitos quando o liner está ativo"
        )
    if np.any(liner_active & (liner_density <= 0.0)):
        raise ValueError(
            "liner_density_kg_m3 deve ser maior que zero quando o liner está ativo"
        )
    _validate_vector_positive("nozzle_density_kg_m3", nozzle_density)
    _validate_vector_positive(
        "nozzle_wall_thickness_factor", nozzle_wall_factor
    )
    _validate_vector_nonnegative(
        "nozzle_min_wall_thickness_m", nozzle_min_wall
    )
    _validate_vector_finite("divergent_half_angle_rad", divergent_angle)
    if np.any((divergent_angle <= 0.0) | (divergent_angle >= np.pi / 2.0)):
        raise ValueError("divergent_half_angle_rad deve estar em (0, pi/2)")
    _validate_vector_nonnegative("chamber_pressure_pa", chamber_pressure)
    _validate_vector_nonnegative("port_area_m2", port_area)
    _validate_vector_nonnegative("propellant_mass_kg", propellant_mass)
    _validate_vector_positive("ultimate_strength_mpa", ultimate_strength)
    _validate_vector_finite("casing_strength_factor", strength_factor)
    strength_factor = np.maximum(strength_factor, 0.01)
    casing_wall = np.maximum(casing_wall, 1e-5)

    casing_mass = _casing_mass_with_bulkheads_kg_vectorized(
        chamber_radius,
        casing_wall,
        chamber_length,
        casing_density,
        bulkhead_fraction,
    )
    liner_mass = _liner_mass_kg_vectorized(
        chamber_radius,
        chamber_length,
        liner_thickness,
        liner_density,
    )
    nozzle_mass = _nozzle_mass_kg_vectorized(
        chamber_radius,
        throat_radius,
        exit_radius,
        divergent_angle,
        casing_wall,
        nozzle_density,
        nozzle_wall_factor,
        nozzle_min_wall,
    )

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            dry_mass = casing_mass + liner_mass + nozzle_mass
            initial_mass = dry_mass + propellant_mass
            final_mass = dry_mass
            structural_ratio = np.where(
                initial_mass > 0.0, dry_mass / initial_mass, 0.0
            )
            port_throat_ratio = port_area / (np.pi * throat_radius**2)

            inner_radius = np.maximum(chamber_radius, 1e-5)
            outer_radius = inner_radius + casing_wall
            inner_radius_sq = inner_radius**2
            outer_radius_sq = outer_radius**2
            lame_branch = casing_wall / np.maximum(inner_radius, 1e-9) > 0.1
            lame_hoop = (
                chamber_pressure
                * inner_radius_sq
                * (outer_radius_sq + inner_radius_sq)
                / np.maximum(outer_radius_sq - inner_radius_sq, 1e-9)
                / np.maximum(inner_radius_sq, 1e-9)
            )
            lame_axial = chamber_pressure * inner_radius_sq / np.maximum(
                outer_radius_sq - inner_radius_sq, 1e-9
            )
            barlow_hoop = chamber_pressure * inner_radius / casing_wall
            barlow_axial = chamber_pressure * inner_radius / (2.0 * casing_wall)
            hoop = np.where(lame_branch, lame_hoop, barlow_hoop)
            axial = np.where(lame_branch, lame_axial, barlow_axial)
            radial = -chamber_pressure
            stress_invariant = 0.5 * (
                (hoop - radial) ** 2
                + (radial - axial) ** 2
                + (axial - hoop) ** 2
            )
            von_mises = np.sqrt(np.maximum(stress_invariant, 0.0))

            ultimate_pa = ultimate_strength * 1e6 * strength_factor
            burst_pressure = (
                (2.0 / np.sqrt(3.0))
                * ultimate_pa
                * np.log(np.maximum(outer_radius / np.maximum(inner_radius, 1e-9), 1.0))
            )
            burst_safety_factor = burst_pressure / np.maximum(chamber_pressure, 1.0)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "cálculo vetorizado das features estruturais excede o intervalo numérico"
        ) from exc

    results = (
        ("casing_mass_kg", casing_mass),
        ("liner_mass_kg", liner_mass),
        ("nozzle_mass_kg", nozzle_mass),
        ("dry_mass_kg", dry_mass),
        ("motor_initial_mass_kg", initial_mass),
        ("motor_final_mass_kg", final_mass),
        ("structural_mass_ratio", structural_ratio),
        ("port_throat_ratio", port_throat_ratio),
        ("von_mises_at_reference_pa", von_mises),
        ("burst_pressure_pa", burst_pressure),
        ("burst_safety_factor_at_reference_pa", burst_safety_factor),
    )
    for name, result in results:
        _vector_result(name, result)
        if np.any(result < 0.0):
            raise ValueError(f"{name} não pode conter valores negativos")
    if np.any(initial_mass <= 0.0):
        raise ValueError("motor_initial_mass_kg deve ser maior que zero")

    return SurrogateStructuralFeatures(
        casing_mass_kg=np.asarray(casing_mass),
        liner_mass_kg=np.asarray(liner_mass),
        nozzle_mass_kg=np.asarray(nozzle_mass),
        dry_mass_kg=np.asarray(dry_mass),
        motor_initial_mass_kg=np.asarray(initial_mass),
        motor_final_mass_kg=np.asarray(final_mass),
        structural_mass_ratio=np.asarray(structural_ratio),
        port_throat_ratio=np.asarray(port_throat_ratio),
        von_mises_at_reference_pa=np.asarray(von_mises),
        burst_pressure_pa=np.asarray(burst_pressure),
        burst_safety_factor_at_reference_pa=np.asarray(burst_safety_factor),
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
# Helper: pressão de câmara de equilíbrio via iteração de ponto fixo
# ---------------------------------------------------------------------------

def _estimate_equilibrium_pressure(
    kn_initial: float,
    propellant: "Propellant",
    *,
    n_iter: int = 6,
    P0_pa: float = 3.5e6,
) -> float:
    """Pressão de câmara de equilíbrio inicial via iteração de ponto fixo.

    Vieille: r(P) = a * P^n  →  P_eq = rho_p * Kn * r(P_eq) * c*(P_eq)
    Para n < 1 (todos os propelentes APCP), a iteração converge em 4–6 steps.

    Retorna P_eq em Pa. Útil para passar como P_ref_pa a compute_static_features
    em vez de um escalar fixo global, eliminando erro basal de 2–3% em motores
    que operam fora do regime de 3,5 MPa.
    """
    P = float(P0_pa)
    kn = float(kn_initial)
    for _ in range(n_iter):
        r = float(propellant.evaluate_burn_rate(P, 0.0))
        gamma = float(propellant.get_gamma(P))
        R_sp = float(propellant.products_constant)
        T0 = float(propellant.Tc_at_pressure(P))
        c_star = (
            math.sqrt(R_sp * T0 / gamma)
            * ((gamma + 1.0) / 2.0) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
        rho_p = float(propellant.density)
        P = max(rho_p * kn * r * c_star, 1e3)
    return P


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


def _structural_feature_value_to_python(
    value: StructuralFeatureValue,
    *,
    json_compatible: bool = False,
) -> StructuralFeatureDictValue:
    """Converte escalares e arrays 0-D em ``float`` sem achatar lotes.

    Quando ``json_compatible`` é ``True``, arrays com dimensões são convertidos
    para listas Python; o padrão preserva ``numpy.ndarray`` para consumidores
    de lotes e Parquet.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    if json_compatible:
        return array.tolist()
    return array


def structural_features_to_dict(
    feats: SurrogateStructuralFeatures,
    *,
    json_compatible: bool = False,
) -> dict[str, StructuralFeatureDictValue]:
    """Converte features estruturais escalares ou vetorizadas em dict plano.

    Valores escalares, inclusive arrays NumPy 0-D produzidos por
    :func:`compute_structural_features_vectorized` com entradas escalares,
    tornam-se ``float``. Arrays com uma ou mais dimensões permanecem
    ``numpy.ndarray`` no shape broadcastado, para que lotes não sejam
    convertidos acidentalmente em escalares. Com ``json_compatible=True``,
    esses arrays tornam-se listas Python aninhadas, adequadas para
    ``json.dumps``.
    """
    return {
        "surrogate.casing_mass_kg": _structural_feature_value_to_python(
            feats.casing_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.liner_mass_kg": _structural_feature_value_to_python(
            feats.liner_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.nozzle_mass_kg": _structural_feature_value_to_python(
            feats.nozzle_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.dry_mass_kg": _structural_feature_value_to_python(
            feats.dry_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.motor_initial_mass_kg": _structural_feature_value_to_python(
            feats.motor_initial_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.motor_final_mass_kg": _structural_feature_value_to_python(
            feats.motor_final_mass_kg, json_compatible=json_compatible
        ),
        "surrogate.structural_mass_ratio": _structural_feature_value_to_python(
            feats.structural_mass_ratio, json_compatible=json_compatible
        ),
        "surrogate.port_throat_ratio": _structural_feature_value_to_python(
            feats.port_throat_ratio, json_compatible=json_compatible
        ),
        "surrogate.von_mises_at_reference_pa": _structural_feature_value_to_python(
            feats.von_mises_at_reference_pa, json_compatible=json_compatible
        ),
        "surrogate.burst_pressure_pa": _structural_feature_value_to_python(
            feats.burst_pressure_pa, json_compatible=json_compatible
        ),
        "surrogate.burst_safety_factor_at_reference_pa": _structural_feature_value_to_python(
            feats.burst_safety_factor_at_reference_pa,
            json_compatible=json_compatible,
        ),
    }
