import numpy as np
from rlquantopt.rl_envs.zcqubits import setup_ZCQubits4MKrauss_params
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


class ZCQPEEWRD(ZCQPEE):
    """
    ZCQPEE with Resonance Drift (ZCQPEEWRD)

    A wrapper around the base ZCQPEE environment that introduces a fixed random drift
    to the computational qubit resonance frequencies (`omega_s`) during environment
    initialization. The drift simulates per-episode variability, such as qubit frequency
    fluctuations or hardware imperfections.

    On initialization, each element of `omega_s` is independently perturbed by a random
    factor sampled uniformly from the range [-MAX_DRIFT, +MAX_DRIFT], where `MAX_DRIFT` is a class attribute set to 0.01 (±1%).

    Attributes
    ----------
        model_params : dict, optional
            Optional dictionary of model parameters to pass to `setup_ZCQubits4MKrauss_params`.
            If None, the default parameters are used.

        **env_kwargs : dict
            Additional keyword arguments forwarded to the base `ZCQPEE` environment.

    Key Features
    ------------
    - The drift is applied once upon environment instantiation, not per reset().
    - Only the `omega_s` values are affected; all other parameters remain unchanged.
    - Perturbed values can be accessed through `self.model_params['omega_s']`.
    - To control the drift range, modify `ZCQPEEWRD.MAX_DRIFT`.

    Example
    -------
    >>> env = ZCQPEEWRD()
    >>> obs = env.reset()
    >>> print(env.model_params["omega_s"])  # View perturbed resonances
    """

    def __init__(self, max_drift=1e-3, model_params=None, **env_kwargs):
        self.MAX_DRIFT = max_drift  # ratio
        if model_params is None:
            model_params = dict()
        model_params = setup_ZCQubits4MKrauss_params(**model_params)

        # Apply ±1% uniform variation to omega_s
        omega_s = np.array(model_params.get("omega_s"))
        variation = np.random.uniform(-self.MAX_DRIFT, self.MAX_DRIFT, size=omega_s.shape)
        # variation = np.random.normal(0.0, self.MAX_DRIFT)
        omega_s_perturbed = omega_s * (1 + variation)

        # Update model parameters
        model_params["omega_s"] = omega_s_perturbed

        # Initialise base environment
        super().__init__(model_params, **env_kwargs)

    def __repr__(self):
        return (f"ZCQPEEWRD:  Pulse length = {self.pulse_length}\n"
                f"            T = {self.T} ns \n"
                f"            ΔT = {self.dt:.3f} ns\n"
                f"            Max. omega_s drift = {self.MAX_DRIFT * 100.:.4f}%\n"
                f"            Fidelity threshold = {self.FID_THRESH*100.:.2f}%\n"
                f"            Reward scale = {self.REW_SCALE}\n"
                f"            TV penalty scale = {self.TV_PENALTY_SCALE}\n"
                f"            Action scaling = {self.action_scaling}\n"
                f"            A_norm_max = {self.A_norm_max}\n"
                f"            sqrt_iSWAP gate optimisation.\n"
                f"            Action delta_mode={self.delta_mode}\n"
                f"            Action polynomial order={ZCQPEE.ACT_POLY_ORDER}")

    def __str__(self):
        return f"ZCQPEEWRD_pl-{self.pulse_length}_T-{int(self.T)}ns{'_delta_mode' if self.delta_mode else '_abs_mode'}"
