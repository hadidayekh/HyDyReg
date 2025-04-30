'''
**HyDyReg** is a research project that aims at the identification of switched nonlinear dynamical systems through two main methods : the *optimization-based* and *incremental* appraoches, the latter being compatible with active learning by its nature.

The project code is **very far** from being perfect, and the aim of these docs are to clarify a bit the code structure for future users.

This page will explain in *high level* the main components of *some* of the source code files and packages:

* `snds` contains the classes which defined switched nonlinear dynamical systems (SNDS) as well as trajectories. It thus contains methods for simulation, segmentation, etc.
* `opti` contains the main class of the learner, which is `HybridRegressor`. In this model, the main learning algorithms are implemented, such as `fit_opti` to fit using the optimization-based approach, `fit_first_incremental` to fit a first hypothesis with the incremental approach, and `update_continuous` to update the learned dynamics with new data in the scope of the incremental approach. It also contains methods of classification and improving mode regions.
* `utils` contains utility functions.
* `diff_methods` contains methods used to calculate the derivatives by numerical estimation, notably using the five-point-stencil method.
* `plotting` contains the main plotting functions that are used in the experiments, such as plotting in 2D, 3D, or plotting each component with respect to time.
* `main` contains the main learning flow as well as the equivalence query approximations. These methods are called by the experiments. Inside this file, we have a function called `learn_system` which takes an SNDS and does all the active learning on it. Model evaluation and plots are also called in `main`.
* `experiments` is a package with two main files that can be run to reproduce experiments in the thesis. These files also contain the experimental setup *as well as the SNDS implementations of each system*. These are notably:
    - `experiments.experiment_infos`: contains all the experimental setups and definitions of every experiment except the parametric one. Calls the `learn_system` method in `main` on each of these experiment.
    - `experiments.parametric`: contains the setup and definition of the parametric example. Calls the `learn_system` method in `main` on the parametric example with different numbers of modes.

'''