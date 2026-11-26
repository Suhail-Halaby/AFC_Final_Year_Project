# Stochastic Langevin bluff-body wake environment
# Physics (non-dimensional, after Rigas et al. 2015 / Brackston et al. 2016)

# Langevin wake environment governing equations (non-dimensional):
    #  dx = (1/tau_x) * (x - x^3 + bias + g_x * u_diff) dt + sqrt(2 D / tau_x) dW_x

# Vortex Shedding mode, stochastic Stuart-Landau equation (non-dimensional):
    #  dA = [(lam - g_A * u_sym + i*omega)A - lam |A| ^2 A]


# Actuators: two jets, sysmetric and antisymmetric, with control inputs u_sym and u_diff
    # u_sym = (j_R + j_L) / 2
    # u_diff = (j_R - j_L) 

from typing import NamedTuple
import jax
import jax.numpy as jnp

class WakeParams(NamedTuple):
    tau_x: jax.Array     # bistable time scale
    D: jax.Array         # noise intensity on x (sets switching rate, Kramers)
    bias: jax.Array      # geometric asymmetry (0 = symmetric body)
    g_x: jax.Array       # actuator gain on x
    lam: jax.Array       # shedding growth rate
    omega: jax.Array     # shedding angular frequency
    sigma_A: jax.Array   # noise on shedding mode
    g_A: jax.Array       # base-bleed gain on shedding growth rate
    k_x: jax.Array       # drag weight of asymmetry
    k_A: jax.Array       # drag weight of shedding
    c_u: jax.Array       # actuation power cost
    tau_u: jax.Array     # actuator lag
    eta: jax.Array       # shedding visibility in sensors
    obs_noise: jax.Array # sensor noise std


class WakeState(NamedTuple):
    x: jax.Array          # bistable mode
    A: jax.Array          # shedding mode (complex64)
    jets: jax.Array       # actual jet outputs (2,)
    obs_hist: jax.Array   # (hist, obs_dim)
    t: jax.Array          # control step counter
    params: WakeParams    # per-episode parameters (domain randomisation)


class LangevinWake:
    def __init__(self, dt=0.01, substeps=5, max_steps=1000, hist=8,
                 n_levels=3, randomise=True):
        self.dt = dt                   # solver step
        self.substeps = substeps       # solver steps per control step
        self.max_steps = max_steps     # control steps per episode
        self.hist = hist
        self.n_levels = n_levels
        self.randomise = randomise
        self.levels = jnp.linspace(0.0, 1.0, n_levels)
        self.obs_dim = 4               # CoP_z, CoP_y, jet_L cmd, jet_R cmd

    # ------------------------------------------------------------------ spaces

    @property
    def num_actions(self):
        return self.n_levels ** 2

    @property
    def observation_shape(self):
        return (self.hist * self.obs_dim,)

    # -------------------------------------------------------------- parameters

    def default_params(self) -> WakeParams:
        f = lambda v: jnp.asarray(v, jnp.float32)
        return WakeParams(tau_x=f(0.5), D=f(0.11), bias=f(0.0), g_x=f(1.0),
                          lam=f(1.0), omega=f(9.0), sigma_A=f(0.1), g_A=f(1.0),
                          k_x=f(1.0), k_A=f(0.5), c_u=f(0.15), tau_u=f(0.05),
                          eta=f(0.3), obs_noise=f(0.05))

    def sample_params(self, key) -> WakeParams:
        """Random wake from the family. Ranges are design choices: widen them
        to make the meta-training distribution harder / more diverse."""
        k = jax.random.split(key, 14)
        u = lambda i, lo, hi: jax.random.uniform(k[i], (), jnp.float32, lo, hi)
        lam = u(4, 0.5, 2.0)
        return WakeParams(
            tau_x=u(0, 0.25, 1.0),
            D=u(1, 0.08, 0.15),
            bias=u(2, -0.15, 0.15),     # non-symmetric bodies -> biased wells
            g_x=u(3, 0.5, 1.5),
            lam=lam,
            omega=u(5, 6.0, 12.0),
            sigma_A=u(6, 0.05, 0.2),
            g_A=lam * u(7, 0.6, 1.5),   # some wakes can be fully stabilised, some not
            k_x=u(8, 0.5, 1.5),
            k_A=u(9, 0.2, 0.8),
            c_u=u(10, 0.05, 0.3),
            tau_u=u(11, 0.02, 0.1),
            eta=u(12, 0.2, 0.5),
            obs_noise=u(13, 0.02, 0.1),
        )

    # ----------------------------------------------------------------- physics
    def _substep(self, carry, key, cmd, p: WakeParams):
        """One Euler-Maruyama step; shedding rotation integrated exactly."""
        x, A, jets = carry
        dt = self.dt
        kx, kr, ki = jax.random.split(key, 3)

        # actuator lag (exact first-order discretisation)
        a = jnp.exp(-dt / p.tau_u)
        jets = a * jets + (1.0 - a) * cmd
        u_diff = jets[1] - jets[0]
        u_sym = 0.5 * (jets[0] + jets[1])

        # bistable mode
        drift_x = (x - x**3 + p.bias + p.g_x * u_diff) / p.tau_x
        x = x + drift_x * dt + jnp.sqrt(2.0 * p.D / p.tau_x * dt) * jax.random.normal(kx)

        # shedding mode: exact rotation, then Euler for growth/saturation + noise
        A = A * jnp.exp(1j * p.omega * dt)
        growth = (p.lam - p.g_A * u_sym) - p.lam * jnp.abs(A) ** 2
        noise = (jax.random.normal(kr) + 1j * jax.random.normal(ki)) / jnp.sqrt(2.0)
        A = A + growth * A * dt + p.sigma_A * jnp.sqrt(dt) * noise
        A = A.astype(jnp.complex64)

        r = -(p.k_x * x**2 + p.k_A * jnp.abs(A) ** 2) - p.c_u * jnp.sum(jets**2)
        cd = 1.0 + p.k_x * x**2 + p.k_A * jnp.abs(A) ** 2
        return (x, A, jets), (r, cd)

    def _observe(self, key, x, A, cmd, p):
        cop = jnp.array([x + p.eta * A.real, p.eta * A.imag])
        cop = cop + p.obs_noise * jax.random.normal(key, (2,))
        return jnp.concatenate([cop, cmd]).astype(jnp.float32)

# --------------------------------------------------------------------- API
    def reset(self, key, params: WakeParams | None = None):
        kp, kx, ka, ko = jax.random.split(key, 4)
        if params is None:
            params = self.sample_params(kp) if self.randomise else self.default_params()
        x = jnp.sign(jax.random.normal(kx)) * 1.0          # start in a well
        phase = jax.random.uniform(ka, (), minval=0, maxval=2 * jnp.pi)
        A = jnp.exp(1j * phase).astype(jnp.complex64)       # on the limit cycle
        jets = jnp.zeros(2, jnp.float32)
        o = self._observe(ko, x, A, jets, params)
        hist = jnp.tile(o, (self.hist, 1))
        state = WakeState(x=x, A=A, jets=jets, obs_hist=hist,
                          t=jnp.array(0, jnp.int32), params=params)
        return hist.reshape(-1), state

    def step_env(self, key, state: WakeState, action):
        p = state.params
        iL, iR = action // self.n_levels, action % self.n_levels
        cmd = jnp.array([self.levels[iL], self.levels[iR]], jnp.float32)
        ks = jax.random.split(key, self.substeps + 1)
        (x, A, jets), (rs, cds) = jax.lax.scan(
            lambda c, k: self._substep(c, k, cmd, p),
            (state.x, state.A, state.jets), ks[:-1])
        o = self._observe(ks[-1], x, A, cmd, p)
        hist = jnp.concatenate([state.obs_hist[1:], o[None]], axis=0)
        t = state.t + 1
        done = t >= self.max_steps
        new = WakeState(x=x, A=A, jets=jets, obs_hist=hist, t=t, params=p)
        info = {"cd": cds.mean(), "x": x, "absA": jnp.abs(A)}
        return hist.reshape(-1), new, rs.mean(), done, info

    def step(self, key, state, action):
        """Step with automatic reset (new random wake) when the episode ends."""
        k_step, k_reset = jax.random.split(key)
        obs, new, r, done, info = self.step_env(k_step, state, action)
        obs_r, state_r = self.reset(k_reset, None if self.randomise else state.params)
        pick = lambda a, b: jax.tree_util.tree_map(lambda u, v: jnp.where(done, u, v), a, b)
        return pick(obs_r, obs), pick(state_r, new), r, done, info



