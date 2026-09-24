# PFTR-Koopman-operator-model
Background: Nonadiabatic tubular reactors have been shown to exhibit multiple steady
states for all possible parameters. For example, for a given range of finite activation
energies, up to 7 steady state solutions have been found. To model the system,
Heinemann and Poore (1981) utilized a two coupled, non-linear system of ODEs that
represented the mass and energy balances for an exothermic, first-order, A→B,
reaction.

This study demonstrated that bifurcation occurs at assigned Damköhler numbers (Da),
with the output measured as maximum temperature. Other necessary parameters were
arbitrarily determined and used throughout the model design. Additionally, variations in
the heat transfer coefficient were shown to dramatically alter the bifurcation profiles.
Goal One: A finite-difference discretization and pseudo-arclength continuation method
was previously developed for the Heinemann–Poore tubular reactor model by Iker
Batidor as part of his undergraduate career. This analysis successfully recovered the
S-shaped steady-state branch and identified a bistable regime associated with fold
bifurcations.

For this project, we will extend the steady-state analysis into the time domain and
investigate whether a finite-dimensional Koopman representation can capture the
resulting nonlinear transient and oscillatory dynamics. We aim to compare the
Koopman-based representation with the original nonlinear reactor model.
Goal Two: We will use the house-made parallel tempering model to generate
independent simulations using the dataset provided in Heinemann and Poore (1981).
Those simulations will add structured noise to the system, which will test the robustness
of the Koopman-based model. Additionally, we will generate a machine-learning
algorithm for MATLAB that will also calculate those same parameters and compare
them to the prior models.

Publication: Robert F. Heinemann and Aubrey B. Poore, Multiplicity, stability, and
oscillatory dynamics of the tubular reactor, Chemical Engineering Science, 1981.
