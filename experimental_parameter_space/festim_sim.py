import festim as F
import numpy as np
import h_transport_materials as htm

# obtain material properties
# inconel
substrate_D = htm.diffusivities.filter(material="inconel_600").filter(
    author="kishimoto"
)[0]
substrate_D_0 = substrate_D.pre_exp.magnitude
substrate_E_D = substrate_D.act_energy.magnitude

substrate_recomb = htm.recombination_coeffs.filter(material="inconel_600").filter(
    author="rota"
)[0]
substrate_Kr_0 = substrate_recomb.pre_exp.magnitude
substrate_E_Kr = substrate_recomb.act_energy.magnitude

substrate_diss = htm.dissociation_coeffs.filter(material="inconel_600").filter(
    author="rota"
)[0]
substrate_Kd_0 = substrate_diss.pre_exp.magnitude
substrate_E_Kd = substrate_diss.act_energy.magnitude


substrate_S = htm.Solubility(
    S_0=(substrate_diss.pre_exp / substrate_recomb.pre_exp) ** 0.5,
    E_S=(0.5 * (substrate_diss.act_energy - substrate_recomb.act_energy)),
)
substrate_S_0 = substrate_S.pre_exp.magnitude
substrate_E_S = substrate_S.act_energy.magnitude


# # alumina
barrier_D = htm.diffusivities.filter(material="alumina").filter(author="serra")[0]
barrier_D_0 = barrier_D.pre_exp.magnitude
barrier_E_D = barrier_D.act_energy.magnitude

barrier_S = htm.solubilities.filter(material="alumina").filter(author="serra")[0]
barrier_S_0 = barrier_S.pre_exp.magnitude
barrier_E_S = barrier_S.act_energy.magnitude


def festim_model_standard(
    T, pressure, foldername, steady=True, regime="diff", atol=1e-08
):
    model_standard = F.Simulation(log_level=40)

    model_standard.mesh = F.MeshFromVertices(vertices=np.linspace(0, 5e-04, num=500))

    substrate_standard = F.Material(
        id=1, D_0=substrate_D_0, E_D=substrate_E_D, S_0=substrate_S_0, E_S=substrate_E_S
    )
    model_standard.materials = F.Materials([substrate_standard])

    model_standard.T = T

    if regime == "diff":
        model_standard.boundary_conditions = [
            F.SievertsBC(
                pressure=pressure, S_0=substrate_S_0, E_S=substrate_E_S, surfaces=[1]
            ),
            F.DirichletBC(value=0, surfaces=[2], field="solute"),
        ]
    elif regime == "surf":
        model_standard.boundary_conditions = [
            F.RecombinationFlux(
                Kr_0=substrate_E_Kr, E_Kr=substrate_E_Kr, order=2, surfaces=[1, 2]
            ),
            F.DissociationFlux(
                Kd_0=substrate_Kd_0, E_Kd=substrate_E_Kd, P=pressure, surfaces=[1]
            ),
        ]
    else:
        ValueError(f"permeation regime {regime} not recognised, should be diff or surf")

    if steady is True:
        model_standard.settings = F.Settings(
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            maximum_iterations=500,
            transient=False,
            linear_solver="mumps",
        )
    else:
        model_standard.settings = F.Settings(
            absolute_tolerance=atol,
            relative_tolerance=1e-10,
            maximum_iterations=100,
            transient=True,
            final_time=1e4,
            linear_solver="mumps",
        )
        model_standard.dt = F.Stepsize(
            initial_value=1, stepsize_change_ratio=1.05, dt_min=1e-03
        )

    outflux = F.HydrogenFlux(surface=2)
    my_derived_quantites = F.DerivedQuantities(
        [outflux],
        filename=f"{foldername}/permeation_standard.csv",
        show_units=True,
    )
    model_standard.exports = F.Exports(
        [
            my_derived_quantites,
            F.XDMFExport(
                field="solute", checkpoint=False, mode=1, folder=f"{foldername}/"
            ),
            # F.TXTExport(
            #     field="solute",
            #     filename=f"{foldername}/mobile_conc_profile_standard.txt",
            # ),
        ]
    )

    model_standard.initialise()
    model_standard.run()


def festim_model_barrier(T, pressure, foldername):
    model_barrier = F.Simulation(log_level=30)

    l_barrier = 1e-06
    l_substrate = 5e-04

    vertices = np.concatenate(
        [
            np.linspace(0, l_barrier, num=1000),
            np.linspace(l_barrier, l_barrier + l_substrate, num=1000),
            np.linspace(l_barrier + l_substrate, 2 * l_barrier + l_substrate, num=1000),
        ]
    )
    model_barrier.mesh = F.MeshFromVertices(vertices=vertices)

    barrier_left = F.Material(
        id=1,
        D_0=barrier_D_0,
        E_D=barrier_E_D,
        S_0=barrier_S_0,
        E_S=barrier_E_S,
        borders=[0, l_barrier],
    )
    substrate = F.Material(
        id=2,
        D_0=substrate_D_0,
        E_D=substrate_E_D,
        S_0=substrate_S_0,
        E_S=substrate_E_S,
        borders=[l_barrier, l_barrier + l_substrate],
    )
    barrier_right = F.Material(
        id=3,
        D_0=barrier_D_0,
        E_D=barrier_E_D,
        S_0=barrier_S_0,
        E_S=barrier_E_S,
        borders=[l_barrier + l_substrate, 2 * l_barrier + l_substrate],
    )
    model_barrier.materials = F.Materials([barrier_left, substrate, barrier_right])

    model_barrier.T = T

    # model_barrier.boundary_conditions = [
    #     F.RecombinationFlux(
    #         Kr_0=substrate_E_Kr, E_Kr=substrate_E_Kr, order=2, surfaces=[1, 2]
    #     ),
    #     F.DissociationFlux(
    #         Kd_0=substrate_Kd_0, E_Kd=substrate_E_Kd, P=pressure, surfaces=[1]
    #     ),
    # ]

    model_barrier.boundary_conditions = [
        F.SievertsBC(pressure=pressure, S_0=barrier_S_0, E_S=barrier_E_S, surfaces=[1]),
        F.DirichletBC(value=0, surfaces=[2], field="solute"),
    ]

    model_barrier.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=1000,
        transient=False,
        final_time=4000,
        linear_solver="mumps",
        chemical_pot=True,
    )

    # model_barrier.dt = F.Stepsize(
    #     initial_value=0.1, stepsize_change_ratio=1.02, dt_min=1e-08
    # )

    my_derived_quantites = F.DerivedQuantities(
        [F.HydrogenFlux(surface=2)],
        filename=f"{foldername}/permeation_barrier.csv",
        nb_iterations_between_exports=1,
        show_units=True,
    )
    model_barrier.exports = F.Exports(
        [
            my_derived_quantites,
            F.XDMFExport(field="solute", checkpoint=False, mode=1, folder="results"),
            F.TXTExport(
                field="solute", filename=f"{foldername}/mobile_conc_profile_barrier.txt"
            ),
        ]
    )

    model_barrier.initialise()
    model_barrier.run()


# festim_model_standard(
#     T=400,
#     pressure=1e3,
#     foldername="results/transient",
#     steady=False,
#     regime="surf",
#     atol=1.5e3,
# )
if __name__ == "__main__":
    pressure_values = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    atols = [2e-01, 2e0, 2e01, 2e2, 2e3, 2e4]
    final_times = [1e7, 1e5, 2e4]

    atols_all_Ts = [
        [
            1e-06,
            1e-05,
            1e-04,
            1e-03,
            1e-02,
            1e-01,
        ],
        [
            1e-04,
            1e-03,
            1e-02,
            1e-01,
            1e0,
            1e1,
        ],
        [
            1e-02,
            1e-01,
            1e00,
            1e1,
            2e1,
            2e2,
        ],
    ]
    # test_temp_values = np.linspace(300, 800, num=10)
    # test_temp_values = [300, 400, 500, 600, 700, 800]

    # for pressure, atol in zip(pressure_values[0:], atols[2][0:]):
    #     print(f"Running case: P={pressure:.1e}")
    #     festim_model_standard(
    #         T=500,
    #         pressure=pressure,
    #         foldername=f"results/testing/P={pressure:.1e}",
    #         steady=False,
    #         regime="surf",
    #         atol=atol,
    #     )

    ##### pressure testing ##### #
    for P_value, atol in zip(pressure_values, atols):
        print(f"Running at {P_value:.1e} Pa")
        festim_model_standard(
            T=800,
            pressure=P_value,
            foldername=f"results/pressure_testing/P={P_value:.0e}",
            regime="surf",
            steady=False,
            atol=atol,
        )

    # test temperture at constant pressure

    # test_temp_values = np.linspace(300, 800, num=10)
    # for temp_value in test_temp_values:
    #     print(f"Testing case P=1e+03, T={temp_value:.1f}")
    #     festim_model_standard(
    #         T=temp_value,
    #         pressure=1e3,
    #         foldername=f"results/parameter_exploration/transient/P=1.0e+03/T={temp_value:.0f}",
    #         # foldername=f"results/parameter_exploration/transient/temp_testing/P=1.0e+03/T={temp_value:.0f}",
    #         steady=False,
    #     )

    # # test pressure and temperature

    # for pressure_value, atol in zip(pressure_values, atols):
    #     for temp_value in test_temp_values:
    #         print(f"Testing case P={pressure_value:.0e}, T={temp_value:.0f}")
    #         festim_model_standard(
    #             T=temp_value,
    #             pressure=pressure_value,
    #             foldername=f"results/parameter_exploration/P={pressure_value:.0e}/T={temp_value:.0f}",
    #             steady=False,
    #             regime="surf",
    #             atol=atol,
    #         )
