"""
FibDisp propagation and analysis core
=====================================
Numerical model for ultrashort-pulse propagation in gas-filled hollow-core
fibers. The module provides gas and waveguide coefficients, pulse generation,
symmetric split-step propagation, nonlinear self-steepening solvers, spectral
phase analysis, GDD compensation, and parameter-sweep utilities.

Citation
--------
If FibDisp is used in scientific work, please cite both the software and the
original scientific publication.

Software:
D. Faccialà, C. Vozzi, G. Sansone, S. De Silvestri, M. Nisoli, and S. Stagira,
"FibDisp," computer software, https://github.com/omegatype/fibdisp-web.

Original scientific publication:
C. Vozzi, M. Nisoli, G. Sansone, S. Stagira, and S. De Silvestri,
"Optimal spectral broadening in hollow-fiber compressor systems,"
Applied Physics B 80, 285-289 (2005).
https://doi.org/10.1007/s00340-004-1721-1

See CITATION.cff for machine-readable software citation metadata and NOTICE.md
for attribution information.
"""

import numpy as np
from scipy.optimize import minimize_scalar

C_LIGHT = 299792458.0  # speed of light, m/s


# ---------------------------------------------------------------------------
# Gas and waveguide propagation coefficients
# ---------------------------------------------------------------------------
def _calculate_gas_coefficients(gas, f0, f, p, R):
    """Return beta2, beta3, alpha(f), gamma for one of the built-in gases.

    beta2 : s^2/m
    beta3 : s^3/m
    alpha : 1/m, array on the supplied FFT-frequency grid
    gamma : 1/(W*m)
    """
    omega0 = f0 * 2 * np.pi
    lambda0 = C_LIGHT / f0 * 1e10          # central wavelength in angstrom
    Aeff = np.pi * R ** 2 * 0.4766         # effective mode area
    n2He = 4.12e-25                        # n2 helium, m^2/W/bar
    nv = 1.45356                           # fused-silica refractive index

    # Sellmeier-like coefficients + nonlinear index per gas.
    # These remain internal implementation details; the GUI exposes only the
    # direct propagation coefficients beta2, beta3, alpha and gamma.
    table = {
        'Helium':   dict(k=6.927e-5, A=2.24e5, B=5.94e10, C=1.72e16, D=0.0,     E=0.0,     n2=n2He),
        'Neon':     dict(k=1.335e-4, A=2.24e5, B=8.09e10, C=3.56e16, D=0.0,     E=0.0,     n2=1.8 * n2He),
        'Argon':    dict(k=5.547e-4, A=5.15e5, B=4.19e11, C=4.09e17, D=4.32e23, E=0.0,     n2=33.98 * n2He),
        'Krypton':  dict(k=8.377e-4, A=6.7e5,  B=8.84e11, C=1.49e18, D=2.74e24, E=5.10e30, n2=64.0 * n2He),
        'Xenon':    dict(k=1.366e-3, A=9.02e5, B=1.81e12, C=4.89e18, D=1.45e25, E=4.34e31, n2=188.2 * n2He),
        'Nitrogen': dict(k=5.547e-4, A=5.15e5, B=4.19e11, C=4.09e17, D=4.32e23, E=0.0,     n2=21.1 * n2He),
    }
    if gas not in table:
        raise ValueError("Unknown gas: %r" % (gas,))
    c = table[gas]
    k, A, B, C, D, E, n2 = c['k'], c['A'], c['B'], c['C'], c['D'], c['E'], c['n2']

    N = 50
    l = np.linspace(lambda0 - lambda0 / 20, lambda0 + lambda0 / 20, N)

    m = k * (1 + A / l ** 2 + B / l ** 4 + C / l ** 6 + D / l ** 8 + E / l ** 10)
    n0 = np.sqrt(m + 1)
    ngas = (np.sqrt(2.0 * ((n0 ** 2 - 1) * p / (n0 ** 2 + 2)) + 1)
            / np.sqrt(1 - (n0 ** 2 - 1) * p / (n0 ** 2 + 2)))
    n = ngas * (1 - 0.5 * (2.405 * l / ((2 * np.pi * R * 1e10) * ngas)) ** 2)

    dl = l[1] - l[0]
    dndl = np.diff(n) / dl
    d2ndl = np.diff(dndl) / dl
    d3ndl = np.diff(d2ndl) / dl
    l2 = l[0:N - 2]
    l3 = l[0:N - 3]

    cc = C_LIGHT * 1e10
    bet2 = l2 ** 3 * d2ndl / (2 * np.pi * cc ** 2)
    bet3 = (-l3 ** 4 / (4 * np.pi ** 2 * cc ** 3)) * (
        3 * d2ndl[0:N - 3] + l3 * d3ndl)

    pix = np.where(l <= lambda0)[0]
    pixel_center = pix[-1]

    ng = ngas[pixel_center]
    n_ratio = nv / ng
    beta2 = bet2[pixel_center] * 1e10       # s^2/m
    beta3 = bet3[pixel_center] * 1e10       # s^3/m

    # Frequency-dependent capillary loss.  With the adopted FFT convention,
    # physical optical frequency is f0 - f_FFT.
    alphae = ((2.405 * C_LIGHT / (2 * np.pi * (f0 - f))) ** 2 / (R ** 3)
              * (n_ratio ** 2 + 1) / np.sqrt(n_ratio ** 2 - 1))

    gam = n2 * p * omega0 / (C_LIGHT * Aeff)
    return beta2, beta3, alphae, gam


def display_gas_coefficients(gas, wavelength_nm, pressure, radius_um):
    """Coefficients displayed in the Settings GUI for a built-in gas.

    Returns a dict with convenient GUI units:
        beta2_fs2_m : fs^2/m
        beta3_fs3_m : fs^3/m
        alpha_1_m   : 1/m, evaluated at the carrier frequency
        gamma_1_W_m : 1/(W*m)

    Built-in gases use the frequency-dependent capillary loss alpha(f).
    """
    if gas == "Manual":
        raise ValueError("Manual coefficients must be supplied explicitly.")

    wavelength_nm = float(wavelength_nm)
    pressure = float(pressure)
    radius_um = float(radius_um)
    if wavelength_nm <= 0 or pressure < 0 or radius_um <= 0:
        raise ValueError("Wavelength and radius must be positive; pressure cannot be negative.")

    f0 = C_LIGHT / (wavelength_nm * 1e-9)
    beta2, beta3, alphae, gam = _calculate_gas_coefficients(
        gas, f0, np.array([0.0]), pressure, radius_um * 1e-6)

    return {
        'beta2_fs2_m': float(beta2 * 1e30),
        'beta3_fs3_m': float(beta3 * 1e45),
        'alpha_1_m': float(alphae[0]),
        'gamma_1_W_m': float(gam),
    }


def gas_parameters(gas, f0, f, p, R, enable_gvd=True, enable_tod=True,
                   enable_loss=True, manual_coeffs=None, coefficient_overrides=None):
    """
    Compute the linear propagation operator Dop(f) and nonlinear gamma.

    For built-in gases, beta2, beta3, alpha(f) and gamma are calculated from
    gas/fiber parameters.  For gas == 'Manual', the four direct coefficients
    are taken from `manual_coeffs`:

        beta2_fs2_m : fs^2/m
        beta3_fs3_m : fs^3/m
        alpha_1_m   : 1/m (constant across the spectrum)
        gamma_1_W_m : 1/(W*m)
    """
    f = np.asarray(f)

    if gas == "Manual":
        if manual_coeffs is None:
            raise ValueError("Manual gas mode requires beta2, beta3, alpha and gamma.")
        try:
            beta2 = float(manual_coeffs['beta2_fs2_m']) * 1e-30
            beta3 = float(manual_coeffs['beta3_fs3_m']) * 1e-45
            alpha_value = float(manual_coeffs['alpha_1_m'])
            gam = float(manual_coeffs['gamma_1_W_m'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid manual propagation coefficients.") from exc

        alphae = np.full(f.shape, alpha_value, dtype=float)
    else:
        beta2, beta3, alphae, gam = _calculate_gas_coefficients(
            gas, f0, f, p, R)

    # Direct-coefficient overrides replace only the specified coefficients.
    # beta2, beta3, and gamma overrides leave the calculated alpha(f) in place;
    # an alpha override sets a constant attenuation coefficient.
    if coefficient_overrides:
        try:
            if 'beta2_fs2_m' in coefficient_overrides:
                beta2 = float(coefficient_overrides['beta2_fs2_m']) * 1e-30
            if 'beta3_fs3_m' in coefficient_overrides:
                beta3 = float(coefficient_overrides['beta3_fs3_m']) * 1e-45
            if 'alpha_1_m' in coefficient_overrides:
                alpha_value = float(coefficient_overrides['alpha_1_m'])
                alphae = np.full(f.shape, alpha_value, dtype=float)
            if 'gamma_1_W_m' in coefficient_overrides:
                gam = float(coefficient_overrides['gamma_1_W_m'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid sweep propagation-coefficient override.') from exc

    Dop = np.zeros_like(f, dtype=complex)
    if enable_gvd:
        Dop = Dop + (-1j / 2 * beta2 * (2 * np.pi * 1j * f) ** 2)
    if enable_tod:
        Dop = Dop + (1.0 / 6 * beta3 * (2 * np.pi * 1j * f) ** 3)
    if enable_loss:
        Dop = Dop - alphae / 2

    return Dop, gam


# ---------------------------------------------------------------------------
# Input pulse generation
# ---------------------------------------------------------------------------
def generate_pulse(E, fw, l, GDD, TOD, NP, XN, pulse_shape="Gaussian"):
    """
    Build the input field (time & frequency) for a chirped pulse.

    Supported transform-limited intensity shapes are:
      * Gaussian:       P(t) = P0 exp[-(t/T0)^2]
      * Sech:           P(t) = P0 sech^2(t/T0)
      * Super-Gaussian: P(t) = P0 exp[-(t/T0)^4]

    The user-specified ``fw`` is always the INTENSITY FWHM and ``E`` is the
    pulse energy.  Each shape therefore gets its own T0 and peak-power
    normalization while the input GDD/TOD are applied identically in the
    spectral domain.
    """
    tau = float(fw) * 1e-15
    if tau <= 0:
        raise ValueError("TL FWHM must be positive.")

    shape = str(pulse_shape).strip().lower().replace('_', '-').replace(' ', '-')
    if shape in ('gaussian', 'gauss'):
        shape_name = 'Gaussian'
        # Gaussian intensity exp[-(t/T0)^2] with T0 = FWHM/1.665.
        T0 = tau / 1.665
        energy_factor = np.sqrt(np.pi) * T0
        field_builder = lambda tt: np.exp(-tt ** 2 / (2.0 * T0 ** 2))
    elif shape in ('sech', 'sech2', 'sech^2'):
        shape_name = 'Sech'
        # Intensity: sech^2(t/T0), FWHM = 2 acosh(sqrt(2)) T0.
        T0 = tau / (2.0 * np.arccosh(np.sqrt(2.0)))
        energy_factor = 2.0 * T0
        field_builder = lambda tt: 1.0 / np.cosh(tt / T0)
    elif shape in ('super-gaussian', 'supergaussian', 'super-gaussian-4', 'supergaussian4'):
        shape_name = 'Super-Gaussian'
        # Intensity: exp[-(t/T0)^4], FWHM = 2 (ln2)^(1/4) T0.
        T0 = tau / (2.0 * np.log(2.0) ** 0.25)
        # Integral exp[-(t/T0)^4] dt = 2 T0 Gamma(5/4).
        from math import gamma as gamma_function
        energy_factor = 2.0 * T0 * gamma_function(1.25)
        field_builder = lambda tt: np.exp(-0.5 * (tt / T0) ** 4)
    else:
        raise ValueError(
            "Unknown pulse shape %r. Choose Gaussian, Sech, or Super-Gaussian."
            % pulse_shape)

    lambda_ = float(l) * 1e-9
    f0 = C_LIGHT / lambda_

    Epin = float(E) * 1e-3
    P0 = Epin / energy_factor

    # XN scales the half-width of the computational time window in units of
    # the characteristic width T0 of the selected pulse shape.
    t = np.linspace(-T0 * XN, T0 * XN, NP + 1)[:-1]
    delt = t[1] - t[0]
    delf = 1.0 / (t[-1] - t[0] + delt)
    f = delf * np.arange(-NP / 2, NP / 2)

    Ut_in_tf = field_builder(t).astype(complex)

    # `f` is the NumPy-FFT coordinate. With fft ~ exp(-i*Omega_FFT*T),
    # physical optical detuning is Omega = omega-omega0 = -Omega_FFT.
    omega_phys_fs = -2 * np.pi * f * 1e-15
    chirp_phase = (float(GDD) * omega_phys_fs ** 2 / 2.0
                   + float(TOD) * omega_phys_fs ** 3 / 6.0)

    Uf_in_tf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ut_in_tf)))
    Uf_in = Uf_in_tf * np.exp(+1j * chirp_phase)
    Ut_in = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf_in)))

    pulse_power = np.abs(Ut_in) ** 2 * P0
    pulse_phase = np.angle(Ut_in)
    spec_power = np.abs(Uf_in / NP / delf) ** 2 * P0
    spec_phase = np.angle(Uf_in)

    return (Ut_in, Uf_in, pulse_power, pulse_phase, spec_power, spec_phase,
            P0, t, f, f0, delf, delt)


# ---------------------------------------------------------------------------
# Split-step pulse propagation
# ---------------------------------------------------------------------------
def _spectral_time_derivative(A, f):
    """Return dA/dT using the same centered FFT convention as the SSFM."""
    Af = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))
    Af = Af * (2j * np.pi * f)
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Af)))


def _nonlinear_rhs(U, f, f0, gamP0, enable_spm, enable_self_steepening):
    """Nonlinear GNLSE right-hand side for one frozen linear half-step.

    dU/dz = i gamma P0 |U|^2 U
            - gamma P0 / omega0 * d(|U|^2 U)/dT

    Unlike the exponential shock update, this evaluates the shock
    derivative directly and never divides by U near a field zero.
    """
    rhs = np.zeros_like(U, dtype=complex)
    cubic = np.abs(U) ** 2 * U
    if enable_spm:
        rhs = rhs + 1j * gamP0 * cubic
    if enable_self_steepening:
        rhs = rhs - (gamP0 / (2 * np.pi * f0)) * _spectral_time_derivative(cubic, f)
    return rhs


def _nonlinear_step_rk4_adaptive(
        U, f, f0, delt, deltaz, gamP0,
        enable_spm=True, enable_self_steepening=True):
    """Accurate nonlinear sub-step using RK4 with automatic internal z steps.

    Two simple local criteria control the number of internal steps:
      * nonlinear phase increment <= 0.20 rad per internal step;
      * self-steepening Courant number <= 0.25 per internal step.

    The latter uses the intensity-characteristic speed ~3 gamma P/omega0.
    These are conservative accuracy/stability guards, not new physics.
    """
    if deltaz == 0 or (not enable_spm and not enable_self_steepening):
        return U.copy(), 1

    max_i = float(np.max(np.abs(U) ** 2))
    if not np.isfinite(max_i) or max_i <= 0:
        return U.copy(), 1

    strength = abs(float(gamP0)) * max_i
    n_phase = 1
    if enable_spm:
        n_phase = max(1, int(np.ceil(strength * abs(deltaz) / 0.20)))

    n_shock = 1
    if enable_self_steepening:
        omega0 = 2 * np.pi * float(f0)
        shock_courant = 3.0 * strength * abs(deltaz) / (omega0 * float(delt))
        n_shock = max(1, int(np.ceil(shock_courant / 0.25)))

    n_sub = max(n_phase, n_shock)
    if n_sub > 5000:
        raise RuntimeError(
            "Accurate self-steepening requires more than 5000 internal RK4 "
            "steps in one SSFM interval. Increase NZ or enlarge the temporal "
            "resolution/window before continuing.")

    h = deltaz / n_sub
    V = U.copy()
    for _ in range(n_sub):
        k1 = _nonlinear_rhs(
            V, f, f0, gamP0, enable_spm, enable_self_steepening)
        k2 = _nonlinear_rhs(
            V + 0.5 * h * k1, f, f0, gamP0,
            enable_spm, enable_self_steepening)
        k3 = _nonlinear_rhs(
            V + 0.5 * h * k2, f, f0, gamP0,
            enable_spm, enable_self_steepening)
        k4 = _nonlinear_rhs(
            V + h * k3, f, f0, gamP0,
            enable_spm, enable_self_steepening)
        V = V + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(V)):
            raise RuntimeError(
                "Accurate self-steepening RK4 step diverged. Increase NZ, "
                "increase the temporal resolution, or reduce nonlinear strength.")
    return V, n_sub


def propagate_pulse_ii(
        Ut, f, f0, Dop, deltaz, gamP0, enable_spm=True,
        enable_self_steepening=True, self_steepening_solver='fast', delt=None):
    """One symmetric split-step Fourier step of length ``deltaz``.

    ``self_steepening_solver='fast'`` uses a frozen exponential nonlinear update.
    ``'rk4'`` integrates SPM and self-steepening together with adaptive internal
    RK4 substeps and is more robust for strong nonlinearities.
    """
    Uf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ut)))
    Uf = Uf * np.exp(0.5 * deltaz * Dop)              # half dispersive step
    Ud = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf)))

    solver = str(self_steepening_solver).strip().lower()
    accurate = solver.startswith('rk4') or solver.startswith('accurate')

    if accurate and enable_self_steepening:
        if delt is None or delt <= 0:
            if len(f) < 2:
                raise ValueError("Accurate self-steepening requires a valid time step.")
            df = abs(float(f[1] - f[0]))
            delt = 1.0 / (len(f) * df)
        Un, _ = _nonlinear_step_rk4_adaptive(
            Ud, f, f0, delt, deltaz, gamP0,
            enable_spm=enable_spm,
            enable_self_steepening=enable_self_steepening)
    else:
        # Fast nonlinear update using frozen exponential SPM and shock factors.
        Ud2 = np.abs(Ud) ** 2
        if enable_spm:
            Un1 = np.exp(+1j * gamP0 * deltaz * np.abs(Ud) ** 2)
        else:
            Un1 = 1.0

        if enable_self_steepening:
            B1 = Ud2 * Ud
            Bff1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(B1)))
            Bff1 = Bff1 * 2 * 1j * np.pi * f
            B1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Bff1))) / (Ud + 0.002)
            Un2 = np.exp((-gamP0 * deltaz / (f0 * 2 * np.pi)) * B1)
        else:
            Un2 = 1.0
        Un = Ud * Un1 * Un2

    Uf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Un)))
    Uf = Uf * np.exp(0.5 * deltaz * Dop)              # second half dispersive step
    Ud = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf)))

    Ut = Ud
    Uf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ut)))
    return Ut, Uf


# ---------------------------------------------------------------------------
# Driver: full propagation over the fiber (mirrors StartButtonPushed)
# ---------------------------------------------------------------------------
def run_simulation(params, progress_cb=None):
    """
    Run the complete simulation given a parameter dict and return a results dict.

    `params` keys:
        gas (str), pressure (atm), fiber_length_cm, radius_um, energy_mJ,
        tl_fwhm_fs, wavelength_nm, GDD_fs2, TOD_fs3, NZ (int),
        n_exp (int; NP = 2**n_exp), XN.

    `progress_cb(percent)` is called (if given) with an integer 0..100.

    Returns a dict with the input/output fields, axes, and scalar results.
    Raises RuntimeError on split-step divergence.
    """
    gas = params['gas']
    p = params['pressure']
    R = params['radius_um'] * 1e-6
    E = params['energy_mJ']
    fw = params['tl_fwhm_fs']
    l = params['wavelength_nm']
    GDD = params['GDD_fs2']
    TOD = params['TOD_fs3']
    NZ = int(params['NZ'])
    NP = 2 ** int(params['n_exp'])
    XN = params['XN']
    pulse_shape = params.get('pulse_shape', 'Gaussian')
    Lu = params['fiber_length_cm']
    en_mj = E

    # Number of spectra retained only for propagation visualization.
    # It does not change the SSFM step size or propagation physics.
    n_z_spectra_requested = int(params.get('n_z_spectra', 10))
    if n_z_spectra_requested < 2:
        raise ValueError("Propagation spectra must be at least 2.")

    # Optional propagation-term switches; omitted entries default to enabled.
    enable_spm = bool(params.get('enable_spm', True))
    enable_self_steepening = bool(params.get('enable_self_steepening', True))
    self_steepening_solver = params.get('self_steepening_solver', 'fast')
    enable_gvd = bool(params.get('enable_gvd', True))
    enable_tod = bool(params.get('enable_tod', True))
    enable_loss = bool(params.get('enable_loss', True))

    manual_coeffs = None
    if gas == 'Manual':
        manual_coeffs = {
            'beta2_fs2_m': params['beta2_fs2_m'],
            'beta3_fs3_m': params['beta3_fs3_m'],
            'alpha_1_m': params['alpha_1_m'],
            'gamma_1_W_m': params['gamma_1_W_m'],
        }

    (Ut_in, Uf_in, pulse_power_in, pulse_phase_in, spec_power_in, spec_phase_in,
     P0, t, f, f0, delf, delt) = generate_pulse(
        E, fw, l, GDD, TOD, NP, XN, pulse_shape=pulse_shape)

    Dop, gam = gas_parameters(
        gas, f0, f, p, R,
        enable_gvd=enable_gvd,
        enable_tod=enable_tod,
        enable_loss=enable_loss,
        manual_coeffs=manual_coeffs,
        coefficient_overrides=params.get('coefficient_overrides'))

    L = Lu * 1e-2
    if NZ < 2:
        raise ValueError("NZ must be at least 2.")
    # NZ is the number of points on the propagation axis, hence there are
    # NZ-1 SSFM intervals between z=0 and z=L.
    deltaz = L / (NZ - 1)

    Ut_start = Ut_in.copy()
    error_prop = False
    error_bound_f = False
    error_bound_t = False
    found_error_bound = False
    perc_reached = 0

    # Boundary checks are referenced to the actual peak field amplitudes
    # in their respective domains.
    Ut_max = float(np.max(np.abs(Ut_in)))
    Uf_max = float(np.max(np.abs(Uf_in)))

    Ut_ = Ut_start
    Uf_ = Uf_in

    # Save spectra at approximately equally spaced SSFM step indices, including
    # the input (z=0) and the last simulated step.  If the requested number is
    # larger than the number of available steps, all steps are retained.
    n_z_spectra = min(max(2, n_z_spectra_requested), max(2, NZ))
    spectral_save_steps = np.unique(
        np.rint(np.linspace(0, NZ - 1, n_z_spectra)).astype(int))
    spectral_save_set = set(int(x) for x in spectral_save_steps)

    spectral_z_m = [0.0]
    spectral_power_z = [spec_power_in.copy()]
    temporal_power_z = [pulse_power_in.copy()]

    prev_perc = -1
    for ii in range(1, NZ):
        Ut_, Uf_ = propagate_pulse_ii(
            Ut_start, f, f0, Dop, deltaz, gam * P0,
            enable_spm=enable_spm,
            enable_self_steepening=enable_self_steepening,
            self_steepening_solver=self_steepening_solver, delt=delt)
        if not np.all(np.isfinite(Ut_)):
            error_prop = True
            break
        if not found_error_bound:
            perc = round(ii / NZ * 100)
            if (np.abs(Uf_[0]) / np.abs(Uf_max) > 0.001
                    or np.abs(Uf_[-1]) / np.abs(Uf_max) > 0.001):
                error_bound_f = True
                found_error_bound = True
                perc_reached = perc
            if (np.abs(Ut_[0]) / np.abs(Ut_max) > 0.001
                    or np.abs(Ut_[-1]) / np.abs(Ut_max) > 0.001):
                error_bound_t = True
                found_error_bound = True
                perc_reached = perc
        Ut_start = Ut_

        if ii in spectral_save_set:
            spectral_z_m.append(ii * deltaz)
            spectral_power_z.append(
                np.abs(Uf_ / NP / delf) ** 2 * P0)
            temporal_power_z.append(np.abs(Ut_) ** 2 * P0)

        perc = round(ii / NZ * 100)
        if perc != prev_perc:
            prev_perc = perc
            if progress_cb is not None:
                progress_cb(perc)

    if error_prop:
        raise RuntimeError(
            "Computation error: the split-step procedure diverged. Increase the "
            "number of z points, or reduce pressure / fiber length / energy, or "
            "increase pulse duration / fiber radius / n, or pick a gas with higher Ip."
        )

    Ut = Ut_
    Uf = Uf_
    pulse_power_out = np.abs(Ut) ** 2 * P0
    pulse_phase_out = np.angle(Ut)
    spec_power_out = np.abs(Uf / NP / delf) ** 2 * P0
    spec_phase_out = np.angle(Uf)

    out_energy_mJ = (np.sum(spec_power_out) * delf) * 1e3
    transmission = out_energy_mJ / en_mj * 100.0

    Dw1 = np.sqrt(np.sum(f ** 2 * spec_power_in) / np.sum(spec_power_in)
                  - (np.sum(f * spec_power_in) / np.sum(spec_power_in)) ** 2)
    Dw2 = np.sqrt(np.sum(f ** 2 * spec_power_out) / np.sum(spec_power_out)
                  - (np.sum(f * spec_power_out) / np.sum(spec_power_out)) ** 2)
    broadening = Dw2 / Dw1

    # Chirp and group delay.
    # With E^(+) = A(T) exp(-i*omega0*t), the physical instantaneous
    # frequency shift is delta_nu = -(1/2*pi) d(arg A)/dT.
    # Store chirp in Hz so the GUI conversion chirp*1e-12 is correctly THz.
    pulse_phase_unwrapped = np.unwrap(pulse_phase_out)
    chirp = -np.gradient(pulse_phase_unwrapped, delt) / (2 * np.pi)

    # Group delay is d(phi)/d(omega), not d(phi)/d(f).  Since
    # omega = 2*pi*f_phys and f_phys = f0 - f_FFT:
    # GD = -(1/2*pi) d(phi)/d(f_FFT).
    # Store spectral group delay under the result key used by the GUI.
    gdd_out = -np.append(np.diff(np.unwrap(spec_phase_out)), 0.0) / (2 * np.pi * delf)

    # transform-limited pulse (flat spectral phase)
    Ut_TL = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.abs(Uf))))
    pulse_power_out_TL = np.abs(Ut_TL) ** 2 * P0
    tl_norm = pulse_power_out_TL / np.max(pulse_power_out_TL)
    p5 = np.where(tl_norm > 0.5)[0]
    tl_fwhm = (t[p5[-1]] - t[p5[0]]) * 1e15

    # Reference the unwrapped spectral phase to the significant spectral region.
    def sig_range(power, thresh):
        norm = power / np.max(power)
        pix = np.where(norm > thresh)[0]
        first, last = pix[0], pix[-1]
        rng = last - first
        fn = max(first - rng // 2, 0)
        ln = min(last + rng // 2, len(power) - 1)
        return fn, ln

    _, last_pix_new = sig_range(spec_power_out, 0.1)
    phase_unwr = np.unwrap(spec_phase_out)
    phase_unwr = phase_unwr - phase_unwr[last_pix_new] + np.mod(phase_unwr[last_pix_new], 2 * np.pi)

    return dict(
        t=t, f=f, f0=f0, delf=delf, delt=delt, P0=P0, NP=NP,
        Ut_in=Ut_in, Uf_in=Uf_in,
        pulse_power_in=pulse_power_in, pulse_phase_in=pulse_phase_in,
        spec_power_in=spec_power_in, spec_phase_in=spec_phase_in,
        Ut=Ut, Uf=Uf,
        pulse_power_out=pulse_power_out, pulse_phase_out=pulse_phase_out,
        spec_power_out=spec_power_out, spec_phase_out=spec_phase_out,
        pulse_power_out_TL=pulse_power_out_TL,
        chirp=chirp, gdd_out=gdd_out, phase_unwrapped=phase_unwr,
        out_energy_mJ=out_energy_mJ, transmission=transmission,
        broadening=broadening, tl_fwhm=tl_fwhm,
        spectral_z_m=np.asarray(spectral_z_m),
        spectral_power_z=np.asarray(spectral_power_z),
        temporal_power_z=np.asarray(temporal_power_z),
        n_z_spectra_saved=len(spectral_z_m),
        enabled_terms=dict(
            spm=enable_spm,
            self_steepening=enable_self_steepening,
            self_steepening_solver=str(self_steepening_solver),
            gvd=enable_gvd,
            tod=enable_tod,
            loss=enable_loss),
        error_bound_f=error_bound_f, error_bound_t=error_bound_t,
        perc_reached=perc_reached,
    )


def polynomial_fit(res, min_f_THz, max_f_THz, order):
    """Spectral-power-weighted polynomial fit of the spectral phase.

    The fit variable is the physical angular-frequency detuning (rad/fs).
    The least-squares objective is weighted by spectral power density, so
    spectral regions carrying more power contribute more strongly.

    Returns (coeffs_with_factorials, x_axis_THz, fitted_phase).
    """
    f = res['f']
    f0 = res['f0']
    phase_unwrapped = res['phase_unwrapped']
    spec_power = res['spec_power_out']

    # NumPy FFT coordinate and corresponding physical optical frequency:
    # nu_phys = nu0 - f_FFT, hence Omega_phys = -2*pi*f_FFT.
    f_axis = f * 1e-12
    f_phys_THz = f0 * 1e-12 - f_axis
    mask = (f_phys_THz > min_f_THz) & (f_phys_THz < max_f_THz)

    x_axis = -f_axis[mask] * 1e-3 * 2 * np.pi       # physical detuning, rad/fs
    y_axis = phase_unwrapped[mask]
    power_axis = np.maximum(spec_power[mask], 0.0)

    keep = (np.isfinite(x_axis) & np.isfinite(y_axis)
            & np.isfinite(power_axis) & (power_axis > 0))
    x_axis = x_axis[keep]
    y_axis = y_axis[keep]
    power_axis = power_axis[keep]
    x_THz = f_phys_THz[mask][keep]

    order = int(order)
    if x_axis.size <= order:
        raise ValueError(
            "Not enough non-zero spectral-power points in the selected fit range."
        )

    # np.polyfit minimizes sum((w * residual)^2); sqrt(S) therefore gives
    # the weighted objective sum(S * residual^2).
    power_axis = power_axis / np.max(power_axis)
    weights = np.sqrt(power_axis)
    p = np.polyfit(x_axis, y_axis, order, w=weights)

    from math import factorial
    fackt = np.array([factorial(i) for i in range(len(p))])
    coeffs = p[::-1] * fackt
    fitted = np.polyval(p, x_axis)
    return coeffs, x_THz, fitted


# ---------------------------------------------------------------------------
# GDD compressor design
# ---------------------------------------------------------------------------
def pulse_fwhm_fs(t, power):
    """FWHM in fs around the global pulse maximum, with linear interpolation."""
    t = np.asarray(t, dtype=float)
    power = np.asarray(power, dtype=float)
    finite = np.isfinite(t) & np.isfinite(power)
    if not np.any(finite):
        return np.nan

    p = np.where(finite, power, 0.0)
    imax = int(np.argmax(p))
    pmax = float(p[imax])
    if pmax <= 0:
        return np.nan
    half = 0.5 * pmax

    left_candidates = np.where(p[:imax] <= half)[0]
    if left_candidates.size == 0 or imax == 0:
        return np.nan
    i0 = int(left_candidates[-1])
    i1 = i0 + 1
    if p[i1] == p[i0]:
        t_left = t[i1]
    else:
        t_left = t[i0] + (half - p[i0]) * (t[i1] - t[i0]) / (p[i1] - p[i0])

    right_rel = np.where(p[imax + 1:] <= half)[0]
    if right_rel.size == 0:
        return np.nan
    j1 = int(imax + 1 + right_rel[0])
    j0 = j1 - 1
    if p[j1] == p[j0]:
        t_right = t[j0]
    else:
        t_right = t[j0] + (half - p[j0]) * (t[j1] - t[j0]) / (p[j1] - p[j0])

    return float((t_right - t_left) * 1e15)


def pulse_rms_duration_fs(t, power):
    """Intensity RMS duration sigma_t in fs, using the entire temporal profile."""
    t = np.asarray(t, dtype=float)
    power = np.asarray(power, dtype=float)
    finite = np.isfinite(t) & np.isfinite(power) & (power >= 0)
    if not np.any(finite):
        return np.nan

    tt = t[finite]
    pp = power[finite]
    total = np.sum(pp)
    if total <= 0:
        return np.nan

    mean_t = np.sum(tt * pp) / total
    variance = np.sum((tt - mean_t) ** 2 * pp) / total
    variance = max(float(variance), 0.0)
    return float(np.sqrt(variance) * 1e15)


def pulse_energy_interval(t, power, fraction=0.95):
    """Shortest contiguous temporal interval containing `fraction` of energy.

    Returns
    -------
    (t_left, t_right) in seconds.
    """
    t = np.asarray(t, dtype=float)
    power = np.asarray(power, dtype=float)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("Energy fraction must be in (0, 1].")

    finite = np.isfinite(t) & np.isfinite(power)
    if not np.any(finite):
        return np.nan, np.nan

    p = np.where(finite & (power > 0), power, 0.0)
    if len(t) < 2:
        return float(t[0]), float(t[0])

    if np.any(np.diff(t) <= 0):
        order = np.argsort(t)
        t = t[order]
        p = p[order]

    dt = np.diff(t)
    increments = 0.5 * (p[:-1] + p[1:]) * dt
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if total <= 0:
        return np.nan, np.nan

    cdf = cumulative / total

    # Remove repeated CDF values caused by exact zero-power regions.
    keep = np.concatenate(([True], np.diff(cdf) > 0))
    cdf_u = cdf[keep]
    t_u = t[keep]
    if cdf_u.size < 2:
        return float(t_u[0]), float(t_u[0])

    qmax = 1.0 - fraction
    if qmax <= 0:
        return float(t_u[0]), float(t_u[-1])

    # Find the shortest interval [Q(q), Q(q+fraction)].
    q = np.linspace(0.0, qmax, 401)
    t_left = np.interp(q, cdf_u, t_u)
    t_right = np.interp(q + fraction, cdf_u, t_u)
    widths = t_right - t_left
    ibest = int(np.argmin(widths))
    return float(t_left[ibest]), float(t_right[ibest])


def pulse_energy_width_fs(t, power, fraction=0.95):
    """Shortest contiguous interval containing `fraction` of pulse energy."""
    t_left, t_right = pulse_energy_interval(t, power, fraction)
    if not (np.isfinite(t_left) and np.isfinite(t_right)):
        return np.nan
    return float((t_right - t_left) * 1e15)


def pulse_outer_fwhm95_fs(t, power, fraction=0.95):
    """FWHM found from the outer edges of the 95%-energy interval.

    Algorithm:
      1. Find the shortest interval containing `fraction` of the pulse energy.
      2. Inside that interval, use half the GLOBAL peak intensity as threshold.
      3. Starting at the LEFT 95%-energy boundary, move toward the main peak and
         take the first crossing of Imax/2.
      4. Starting at the RIGHT 95%-energy boundary, move toward the main peak and
         take the first crossing of Imax/2.
      5. Return the distance between those two outer half-height crossings.

    This differs from the usual main-peak FWHM: any sufficiently intense
    pre/post-pulse inside the 95%-energy window can move an outer crossing
    outward and therefore increases this metric.
    """
    t = np.asarray(t, dtype=float)
    power = np.asarray(power, dtype=float)

    finite = np.isfinite(t) & np.isfinite(power)
    if not np.any(finite):
        return np.nan

    if np.any(np.diff(t) <= 0):
        order = np.argsort(t)
        t = t[order]
        power = power[order]

    t95_left, t95_right = pulse_energy_interval(t, power, fraction)
    if not (np.isfinite(t95_left) and np.isfinite(t95_right)):
        return np.nan

    # Construct the signal restricted to the 95%-energy window, explicitly
    # including interpolated samples at the two continuous interval boundaries.
    inside = (t > t95_left) & (t < t95_right)
    t_seg = np.concatenate((
        [t95_left],
        t[inside],
        [t95_right]
    ))
    p_seg = np.concatenate((
        [np.interp(t95_left, t, power)],
        power[inside],
        [np.interp(t95_right, t, power)]
    ))

    if t_seg.size < 2:
        return 0.0

    # Half height is referenced to the global pulse maximum, as for FWHM.
    pmax = float(np.nanmax(power))
    if not np.isfinite(pmax) or pmax <= 0:
        return np.nan
    half = 0.5 * pmax

    # The central/main peak is the global maximum inside the 95%-energy window.
    ipeak = int(np.argmax(p_seg))

    # LEFT: start from the outer 95% boundary and move right toward the peak.
    above_left = np.where(p_seg[:ipeak + 1] >= half)[0]
    if above_left.size == 0:
        return np.nan
    i1 = int(above_left[0])
    if i1 == 0:
        t_cross_left = t_seg[0]
    else:
        i0 = i1 - 1
        if p_seg[i1] == p_seg[i0]:
            t_cross_left = t_seg[i1]
        else:
            t_cross_left = (
                t_seg[i0]
                + (half - p_seg[i0]) * (t_seg[i1] - t_seg[i0])
                / (p_seg[i1] - p_seg[i0])
            )

    # RIGHT: start from the outer 95% boundary and move left toward the peak.
    above_right = np.where(p_seg[ipeak:] >= half)[0]
    if above_right.size == 0:
        return np.nan
    # Last >=half point when scanning left->right is the first one encountered
    # when scanning inward from the right boundary.
    j0 = int(ipeak + above_right[-1])
    if j0 == len(p_seg) - 1:
        t_cross_right = t_seg[-1]
    else:
        j1 = j0 + 1
        if p_seg[j1] == p_seg[j0]:
            t_cross_right = t_seg[j0]
        else:
            t_cross_right = (
                t_seg[j0]
                + (half - p_seg[j0]) * (t_seg[j1] - t_seg[j0])
                / (p_seg[j1] - p_seg[j0])
            )

    return float((t_cross_right - t_cross_left) * 1e15)


def pulse_temporal_metrics(t, power):
    """Return all pulse-duration metrics used by the compressor tab."""
    return {
        'fwhm_fs': pulse_fwhm_fs(t, power),
        'outer_fwhm95_fs': pulse_outer_fwhm95_fs(t, power, 0.95),
        'rms_fs': pulse_rms_duration_fs(t, power),
        'energy95_fs': pulse_energy_width_fs(t, power, 0.95),
    }


def apply_gdd_to_output(res, gdd_fs2):
    """Apply a pure quadratic spectral phase to the simulated output spectrum.

    The physical detuning is Omega = -2*pi*f_FFT.  The transfer function is

        H(Omega) = exp(+i * GDD * Omega^2 / 2)

    using Omega in rad/fs and GDD in fs^2, consistently with generate_pulse().

    This is a phase-only transfer function: spectral power is unchanged.
    """
    gdd_fs2 = float(gdd_fs2)
    f = np.asarray(res['f'])
    Uf = np.asarray(res['Uf'])
    t = np.asarray(res['t'])
    P0 = float(res['P0'])

    omega_phys_fs = -2 * np.pi * f * 1e-15
    transfer_phase = 0.5 * gdd_fs2 * omega_phys_fs ** 2
    H = np.exp(+1j * transfer_phase)

    Uf_comp = Uf * H
    Ut_comp = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf_comp)))
    pulse_power_comp = np.abs(Ut_comp) ** 2 * P0
    pulse_phase_comp = np.angle(Ut_comp)
    chirp_comp = -np.gradient(
        np.unwrap(pulse_phase_comp), float(res['delt'])) / (2 * np.pi)

    spec_power_comp = np.abs(Uf_comp / res['NP'] / res['delf']) ** 2 * P0
    spec_phase_comp = np.angle(Uf_comp)
    phase_unwrapped_comp = np.unwrap(spec_phase_comp)
    gdd_out_comp = -np.append(
        np.diff(phase_unwrapped_comp), 0.0) / (2 * np.pi * float(res['delf']))

    comp_metrics = pulse_temporal_metrics(t, pulse_power_comp)
    output_metrics = pulse_temporal_metrics(t, res['pulse_power_out'])
    tl_metrics = pulse_temporal_metrics(t, res['pulse_power_out_TL'])

    return {
        'gdd_fs2': gdd_fs2,
        'H': H,
        'transfer_phase': transfer_phase,
        'Uf_comp': Uf_comp,
        'Ut_comp': Ut_comp,
        'pulse_power_comp': pulse_power_comp,
        'pulse_phase_comp': pulse_phase_comp,
        'chirp_comp': chirp_comp,
        'spec_power_comp': spec_power_comp,
        'spec_phase_comp': spec_phase_comp,
        'phase_unwrapped_comp': phase_unwrapped_comp,
        'gdd_out_comp': gdd_out_comp,
        'metrics': comp_metrics,
        'output_metrics': output_metrics,
        'tl_metrics': tl_metrics,

        # Alternative metric names used by the sweep result interface.
        'fwhm_fs': comp_metrics['fwhm_fs'],
        'outer_fwhm95_fs': comp_metrics['outer_fwhm95_fs'],
        'rms_fs': comp_metrics['rms_fs'],
        'energy95_fs': comp_metrics['energy95_fs'],
        'tl_fwhm_fs': tl_metrics['fwhm_fs'],
        'output_fwhm_fs': output_metrics['fwhm_fs'],
    }



def _estimate_output_gdd_fs2(res):
    """Weighted quadratic-phase estimate used only to center the GDD search."""
    power = np.asarray(res['spec_power_out'])
    f = np.asarray(res['f'])
    f0 = float(res['f0'])

    if power.size == 0 or np.max(power) <= 0:
        return 0.0

    mask = power > 0.1 * np.max(power)
    if np.count_nonzero(mask) < 5:
        return 0.0

    f_phys_THz = (f0 - f) * 1e-12
    min_f = float(np.min(f_phys_THz[mask]))
    max_f = float(np.max(f_phys_THz[mask]))
    try:
        coeffs, _, _ = polynomial_fit(res, min_f, max_f, 2)
        if len(coeffs) > 2 and np.isfinite(coeffs[2]):
            return float(coeffs[2])
    except Exception:
        pass
    return 0.0


def optimize_output_gdd(res, metric='rms'):
    """Find the phase-only GDD minimizing a selected temporal pulse metric.

    The objective can be strongly non-unimodal, especially for
    ``outer_fwhm95`` because pre/post-pulses can make the outer half-height
    crossings jump from one feature to another.  Therefore this routine does
    NOT rely on a single bounded scalar minimization.

    Search strategy
    ---------------
    1. Compute a spectral-power-weighted quadratic phase fit and use
       ``-GDD_fit`` as an explicit seed.
    2. Perform a broad coarse scan to look for other minima.
    3. Run independent multi-level grid refinements around the fit seed and
       around several of the best coarse-scan basins.
    4. Keep the best value among *all* points ever evaluated.  A refinement is
       never allowed to replace a known better solution with a worse one.

    Parameters
    ----------
    metric : {'fwhm', 'outer_fwhm95', 'rms', 'energy95'}
        'fwhm'         -> FWHM of the main peak only.
        'outer_fwhm95' -> half-height width obtained by scanning inward from
                          the two boundaries of the 95%-energy interval.
        'rms'          -> full-profile intensity RMS duration sigma_t.
        'energy95'     -> shortest interval containing 95% of pulse energy.
    """
    metric = str(metric).lower()
    metric_functions = {
        'fwhm': pulse_fwhm_fs,
        'outer_fwhm95': lambda t, p: pulse_outer_fwhm95_fs(t, p, 0.95),
        'rms': pulse_rms_duration_fs,
        'energy95': lambda t, p: pulse_energy_width_fs(t, p, 0.95),
    }
    if metric not in metric_functions:
        raise ValueError("Unknown optimization metric: %s" % metric)
    metric_fn = metric_functions[metric]

    t = np.asarray(res['t'])
    f = np.asarray(res['f'])
    Uf = np.asarray(res['Uf'])
    P0 = float(res['P0'])
    omega_phys_fs = -2 * np.pi * f * 1e-15

    # Cache repeated evaluations; local refinement grids frequently revisit
    # the same points to within floating-point precision.
    objective_cache = {}

    def objective(gdd_fs2):
        gdd_fs2 = float(gdd_fs2)
        key = round(gdd_fs2, 10)
        if key in objective_cache:
            return objective_cache[key]

        H = np.exp(+0.5j * gdd_fs2 * omega_phys_fs ** 2)
        Ut = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf * H)))
        power = np.abs(Ut) ** 2 * P0
        value = metric_fn(t, power)
        if not np.isfinite(value):
            value = 1e99
        value = float(value)
        objective_cache[key] = value
        return value

    # Weighted quadratic phase estimate.  For a predominantly quadratic
    # spectral phase the compensating seed is its negative.
    gdd_est = _estimate_output_gdd_fs2(res)
    fit_seed = -float(gdd_est)

    tl_fwhm = pulse_fwhm_fs(t, res['pulse_power_out_TL'])
    tl_rms = pulse_rms_duration_fs(t, res['pulse_power_out_TL'])
    characteristic_time = max(
        x for x in (tl_fwhm, 2.355 * tl_rms, 20.0)
        if np.isfinite(x) and x > 0
    )

    span = max(200.0, 4.0 * abs(gdd_est), 8.0 * characteristic_time ** 2)

    # ------------------------------------------------------------------
    # Broad global scan.
    # ------------------------------------------------------------------
    scan_gdd = None
    scan_metric = None
    for _ in range(3):
        scan_gdd = np.linspace(fit_seed - span, fit_seed + span, 121)
        scan_metric = np.array([objective(x) for x in scan_gdd])
        ibest = int(np.argmin(scan_metric))
        if 0 < ibest < len(scan_gdd) - 1:
            break
        span *= 3.0

    coarse_step = float(abs(scan_gdd[1] - scan_gdd[0]))

    # The exact weighted-fit seed is always evaluated even if it does not
    # coincide with a coarse-grid point after future search changes.
    best_gdd = fit_seed
    best_value = objective(best_gdd)
    best_source = "weighted-fit seed"

    coarse_best_i = int(np.argmin(scan_metric))
    coarse_best_gdd = float(scan_gdd[coarse_best_i])
    coarse_best_value = float(scan_metric[coarse_best_i])
    if coarse_best_value < best_value:
        best_gdd = coarse_best_gdd
        best_value = coarse_best_value
        best_source = "global coarse scan"

    # ------------------------------------------------------------------
    # Multi-start local grid refinement.
    #
    # Use the fit seed explicitly plus several best global points.  Deduplicate
    # centers that are effectively the same basin on the coarse-grid scale.
    # ------------------------------------------------------------------
    sorted_i = np.argsort(scan_metric)
    centers = [fit_seed]
    for idx in sorted_i:
        candidate = float(scan_gdd[int(idx)])
        if all(abs(candidate - c) > 0.45 * coarse_step for c in centers):
            centers.append(candidate)
        if len(centers) >= 6:   # fit seed + up to five global basins
            break

    refinement_records = []

    def refine_grid(center, half_width, levels=4, points=81):
        """Repeated dense-grid refinement; robust to non-smooth objectives."""
        local_center = float(center)
        local_half = float(max(half_width, 1e-6))
        local_best_gdd = local_center
        local_best_value = objective(local_center)

        for _ in range(levels):
            grid = np.linspace(
                local_center - local_half,
                local_center + local_half,
                int(points))
            values = np.array([objective(x) for x in grid])
            j = int(np.argmin(values))

            if float(values[j]) < local_best_value:
                local_best_value = float(values[j])
                local_best_gdd = float(grid[j])

            # Recenter on the best grid point and shrink to about two grid
            # spacings on either side.  This gives progressively finer
            # resolution without assuming differentiability or unimodality.
            local_center = float(grid[j])
            grid_step = float(abs(grid[1] - grid[0]))
            local_half = max(2.0 * grid_step, 1e-7)

        return local_best_gdd, local_best_value

    for ic, center in enumerate(centers):
        # The fit-derived basin gets a slightly wider dedicated search.
        half_width = (2.0 * coarse_step if ic == 0 else coarse_step)
        g_local, v_local = refine_grid(
            center, half_width=half_width, levels=4, points=81)
        refinement_records.append((float(center), g_local, v_local))

        if v_local < best_value:
            best_gdd = g_local
            best_value = v_local
            best_source = (
                "weighted-fit local refinement"
                if ic == 0 else "global-basin local refinement"
            )

    # Optional final smooth refinement inside a *very small* neighborhood.
    # Its result is accepted only if it improves the already known best point.
    # This preserves robustness for discontinuous metrics while still giving
    # sub-grid precision for smooth RMS/FWHM cases.
    final_half = max(coarse_step / (40.0 ** 3), 1e-5)
    try:
        opt = minimize_scalar(
            objective, method='bounded',
            bounds=(best_gdd - final_half, best_gdd + final_half),
            options={'xatol': 1e-7, 'maxiter': 80})
        if np.isfinite(opt.fun) and float(opt.fun) < best_value:
            best_gdd = float(opt.x)
            best_value = float(opt.fun)
            best_source = best_source + " + micro-refinement"
    except Exception:
        pass

    result = apply_gdd_to_output(res, best_gdd)
    result['optimization_metric'] = metric
    result['optimization_value'] = best_value
    result['optimization_source'] = best_source
    result['gdd_estimate_fs2'] = gdd_est
    result['fit_seed_gdd_fs2'] = fit_seed
    result['fit_seed_metric'] = objective(fit_seed)
    result['coarse_best_gdd_fs2'] = coarse_best_gdd
    result['coarse_best_metric'] = coarse_best_value
    result['scan_gdd_fs2'] = scan_gdd
    result['scan_metric'] = scan_metric
    result['refinement_records'] = refinement_records
    return result


# ---------------------------------------------------------------------------
# Parameter-sweep utilities
# ---------------------------------------------------------------------------
def _sweep_significant_frequency_range_THz(res, threshold=0.1):
    """Same significant-spectrum window logic used by the GUI phase analysis."""
    power = np.asarray(res['spec_power_out'])
    if power.size == 0 or np.max(power) <= 0:
        raise ValueError("Empty output spectrum.")

    norm = power / np.max(power)
    pix = np.where(norm > threshold)[0]
    if pix.size == 0:
        raise ValueError("No significant spectral points for phase fit.")

    first, last = int(pix[0]), int(pix[-1])
    rng = last - first
    first = max(first - rng // 2, 0)
    last = min(last + rng // 2, len(power) - 1)

    f_phys_THz = (res['f0'] - res['f']) * 1e-12
    return tuple(sorted((float(f_phys_THz[first]), float(f_phys_THz[last]))))


def _sweep_phase_gdd_fs2(res, phase_fit_config=None):
    """Return the spectral-power-weighted phase-fit GDD in fs^2."""
    cfg = phase_fit_config or {}
    order = int(cfg.get('order', 2))
    if order < 2:
        raise ValueError("Phase-fit polynomial order must be at least 2 for GDD.")

    mode = cfg.get('mode', 'auto')
    if mode == 'fixed':
        min_f = float(cfg['min_f_THz'])
        max_f = float(cfg['max_f_THz'])
        if not (np.isfinite(min_f) and np.isfinite(max_f) and min_f < max_f):
            raise ValueError("Invalid fixed phase-fit frequency range.")
    else:
        min_f, max_f = _sweep_significant_frequency_range_THz(res, 0.1)

    coeffs, _, _ = polynomial_fit(res, min_f, max_f, order)
    if len(coeffs) < 3:
        raise ValueError("Phase fit did not return a quadratic coefficient.")
    return float(coeffs[2]), (min_f, max_f)


def _sweep_metric_value(metrics, metric):
    key_map = {
        'fwhm': 'fwhm_fs',
        'outer_fwhm95': 'outer_fwhm95_fs',
        'rms': 'rms_fs',
        'energy95': 'energy95_fs',
    }
    if metric not in key_map:
        raise ValueError("Unknown compressor metric: %s" % metric)
    return float(metrics[key_map[metric]])


def _finalize_sweep_field_result(
        Ut, Uf, *,
        Ut_in, Uf_in, pulse_power_in, pulse_phase_in,
        spec_power_in, spec_phase_in,
        P0, t, f, f0, delf, delt, NP, energy_mJ
    ):
    """Build the result subset needed by phase analysis and GDD optimization."""
    pulse_power_out = np.abs(Ut) ** 2 * P0
    pulse_phase_out = np.angle(Ut)
    spec_power_out = np.abs(Uf / NP / delf) ** 2 * P0
    spec_phase_out = np.angle(Uf)

    Ut_TL = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.abs(Uf))))
    pulse_power_out_TL = np.abs(Ut_TL) ** 2 * P0
    tl_fwhm = pulse_fwhm_fs(t, pulse_power_out_TL)

    phase_unwr = np.unwrap(spec_phase_out)

    out_energy_mJ = (np.sum(spec_power_out) * delf) * 1e3
    transmission = out_energy_mJ / float(energy_mJ) * 100.0

    return dict(
        t=t, f=f, f0=f0, delf=delf, delt=delt, P0=P0, NP=NP,
        Ut_in=Ut_in, Uf_in=Uf_in,
        pulse_power_in=pulse_power_in, pulse_phase_in=pulse_phase_in,
        spec_power_in=spec_power_in, spec_phase_in=spec_phase_in,
        Ut=Ut, Uf=Uf,
        pulse_power_out=pulse_power_out, pulse_phase_out=pulse_phase_out,
        spec_power_out=spec_power_out, spec_phase_out=spec_phase_out,
        pulse_power_out_TL=pulse_power_out_TL,
        phase_unwrapped=phase_unwr,
        tl_fwhm=tl_fwhm,
        out_energy_mJ=out_energy_mJ,
        transmission=transmission,
    )


def _analyze_sweep_result(res, phase_fit_config, optimize_metric):
    """Analyze one propagated spectrum for the parameter-sweep tab.

    Boundary-warning metadata is preserved so the sweep cannot silently use a
    simulation whose temporal/frequency window is too small.
    """
    phase_gdd, fit_range = _sweep_phase_gdd_fs2(res, phase_fit_config)
    optimized = optimize_output_gdd(res, metric=optimize_metric)

    if bool(res.get('error_bound_f', False)):
        warning_type = 'frequency'
    elif bool(res.get('error_bound_t', False)):
        warning_type = 'time'
    else:
        warning_type = ''

    return {
        'phase_gdd_fs2': phase_gdd,
        'optimized_gdd_fs2': float(optimized['gdd_fs2']),
        # Main-peak FWHM of the pulse exactly as it exits the simulated
        # propagation point, before any external GDD compensation.
        'actual_output_fwhm_fs': float(
            pulse_fwhm_fs(res['t'], res['pulse_power_out'])),
        'tl_duration_fs': float(
            pulse_fwhm_fs(res['t'], res['pulse_power_out_TL'])),
        'compressed_duration_fs': _sweep_metric_value(
            optimized['metrics'], optimize_metric),
        'phase_fit_min_THz': fit_range[0],
        'phase_fit_max_THz': fit_range[1],
        'boundary_warning': bool(warning_type),
        'boundary_warning_type': warning_type,
        'boundary_warning_percent': int(res.get('perc_reached', 0)),
        'boundary_warning_z_cm': float(
            res.get('boundary_warning_z_cm', np.nan)),

        # Keep the uncompressed output temporal profile for the sweep
        # time-domain pcolor and for data export.
        'time_fs': np.asarray(res['t'], dtype=float).copy() * 1e15,
        'pulse_power_W': np.asarray(
            res['pulse_power_out'], dtype=float).copy(),

        # Keep the physical spectral axis and spectral power density for the
        # sweep spectral pcolor and for data export. These are output spectra.
        'spectral_frequency_THz': (
            (float(res['f0']) - np.asarray(res['f'])) * 1e-12),
        'spectral_power_W_per_Hz': np.asarray(
            res['spec_power_out'], dtype=float).copy(),

        # Complex output field and native FFT axes for post-fiber GDD processing.
        't_s': np.asarray(res['t'], dtype=float).copy(),
        'f_Hz': np.asarray(res['f'], dtype=float).copy(),
        'Uf_complex': np.asarray(res['Uf'], dtype=complex).copy(),
        'P0_W': float(res['P0']),
    }


def _run_single_pass_length_sweep(
        base_params, lengths_cm, phase_fit_config, optimize_metric,
        progress_cb=None
    ):
    """Sweep fiber length with ONE propagation to max(lengths_cm).

    The nominal SSFM dz is the same as a normal simulation whose fiber length
    is the requested maximum and whose NZ is the current Settings value.
    Whenever a requested length falls between nominal SSFM grid points, the
    current step is shortened so the field is saved at the exact requested
    physical length before propagation continues.
    """
    values = np.asarray(lengths_cm, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Fiber-length sweep requires at least one value.")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Fiber lengths must be finite and non-negative.")

    order = np.argsort(values)
    sorted_values = values[order]
    max_cm = float(sorted_values[-1])

    gas = base_params['gas']
    p = float(base_params['pressure'])
    R = float(base_params['radius_um']) * 1e-6
    E = float(base_params['energy_mJ'])
    fw = float(base_params['tl_fwhm_fs'])
    wavelength = float(base_params['wavelength_nm'])
    GDD = float(base_params['GDD_fs2'])
    TOD = float(base_params['TOD_fs3'])
    NZ = int(base_params['NZ'])
    NP = 2 ** int(base_params['n_exp'])
    XN = float(base_params['XN'])
    pulse_shape = base_params.get('pulse_shape', 'Gaussian')

    if NZ < 2:
        raise ValueError("NZ must be at least 2.")

    enable_spm = bool(base_params.get('enable_spm', True))
    enable_self_steepening = bool(
        base_params.get('enable_self_steepening', True))
    self_steepening_solver = base_params.get('self_steepening_solver', 'fast')
    enable_gvd = bool(base_params.get('enable_gvd', True))
    enable_tod = bool(base_params.get('enable_tod', True))
    enable_loss = bool(base_params.get('enable_loss', True))

    manual_coeffs = None
    if gas == 'Manual':
        manual_coeffs = {
            'beta2_fs2_m': base_params['beta2_fs2_m'],
            'beta3_fs3_m': base_params['beta3_fs3_m'],
            'alpha_1_m': base_params['alpha_1_m'],
            'gamma_1_W_m': base_params['gamma_1_W_m'],
        }

    (Ut_in, Uf_in, pulse_power_in, pulse_phase_in,
     spec_power_in, spec_phase_in, P0, t, f, f0, delf, delt) = generate_pulse(
        E, fw, wavelength, GDD, TOD, NP, XN, pulse_shape=pulse_shape)

    Dop, gam = gas_parameters(
        gas, f0, f, p, R,
        enable_gvd=enable_gvd,
        enable_tod=enable_tod,
        enable_loss=enable_loss,
        manual_coeffs=manual_coeffs)

    max_m = max_cm * 1e-2
    nominal_dz = max_m / (NZ - 1) if max_m > 0 else 0.0

    Ut_current = Ut_in.copy()
    Uf_current = Uf_in.copy()
    z_current = 0.0

    # Boundary validity check for a one-pass fiber-length sweep. Once the field
    # exceeds 1e-3 of its reference peak at a computational boundary, all longer
    # target lengths are flagged. Frequency-boundary warnings take priority.
    Ut_max = float(np.max(np.abs(Ut_in)))
    Uf_max = float(np.max(np.abs(Uf_in)))

    boundary_warning_type = ''
    boundary_warning_z_m = np.nan

    analyses_sorted = []
    total_targets = len(sorted_values)

    for itarget, target_cm in enumerate(sorted_values):
        target_m = float(target_cm) * 1e-2

        if target_m < z_current - 1e-15:
            raise RuntimeError("Internal fiber-length sweep ordering error.")

        while z_current < target_m - max(1e-15, 1e-12 * max(target_m, 1.0)):
            dz = min(nominal_dz, target_m - z_current)
            if dz <= 0:
                break

            Ut_current, Uf_current = propagate_pulse_ii(
                Ut_current, f, f0, Dop, dz, gam * P0,
                enable_spm=enable_spm,
                enable_self_steepening=enable_self_steepening,
                self_steepening_solver=self_steepening_solver, delt=delt)

            if not np.all(np.isfinite(Ut_current)):
                raise RuntimeError(
                    "Computation error: split-step propagation diverged "
                    "during the fiber-length sweep.")

            z_current += dz

            if not boundary_warning_type:
                freq_hit = (
                    np.abs(Uf_current[0]) / np.abs(Uf_max) > 0.001
                    or np.abs(Uf_current[-1]) / np.abs(Uf_max) > 0.001
                )
                time_hit = (
                    np.abs(Ut_current[0]) / np.abs(Ut_max) > 0.001
                    or np.abs(Ut_current[-1]) / np.abs(Ut_max) > 0.001
                )
                if freq_hit:
                    boundary_warning_type = 'frequency'
                    boundary_warning_z_m = z_current
                elif time_hit:
                    boundary_warning_type = 'time'
                    boundary_warning_z_m = z_current

            if progress_cb is not None and max_m > 0:
                # Propagation is most of the work; reserve the last 15% for
                # per-target phase/compressor analysis.
                progress_cb(min(85, int(round(85 * z_current / max_m))))

        # Exact target is reached by construction (last step may be shortened).
        res = _finalize_sweep_field_result(
            Ut_current, Uf_current,
            Ut_in=Ut_in, Uf_in=Uf_in,
            pulse_power_in=pulse_power_in,
            pulse_phase_in=pulse_phase_in,
            spec_power_in=spec_power_in,
            spec_phase_in=spec_phase_in,
            P0=P0, t=t, f=f, f0=f0, delf=delf, delt=delt, NP=NP,
            energy_mJ=E)

        warning_active = (
            bool(boundary_warning_type)
            and np.isfinite(boundary_warning_z_m)
            and target_m >= boundary_warning_z_m - 1e-15
        )
        if warning_active:
            res['error_bound_f'] = boundary_warning_type == 'frequency'
            res['error_bound_t'] = boundary_warning_type == 'time'
            if target_m > 0:
                res['perc_reached'] = int(round(
                    100.0 * boundary_warning_z_m / target_m))
            else:
                res['perc_reached'] = 0
            res['boundary_warning_z_cm'] = (
                float(boundary_warning_z_m) * 1e2)
        else:
            res['error_bound_f'] = False
            res['error_bound_t'] = False
            res['perc_reached'] = 0
            res['boundary_warning_z_cm'] = np.nan

        analyses_sorted.append(
            _analyze_sweep_result(
                res, phase_fit_config, optimize_metric))

        if progress_cb is not None:
            progress_cb(
                85 + int(round(15 * (itarget + 1) / total_targets)))

    # Restore the user-requested value order.
    analyses = [None] * len(values)
    for sorted_i, original_i in enumerate(order):
        analyses[int(original_i)] = analyses_sorted[sorted_i]

    return analyses



def _build_common_sweep_temporal_map(analyses):
    """Interpolate uncompressed output pulse power onto one common time grid.

    Raw per-point time axes and pulse powers are preserved separately.
    The common grid exists only for pcolor visualization/export.

    The displayed union range is based on regions above 1e-6 of each pulse's
    peak power so that very large unused temporal windows do not dominate the
    plot. Any nonlinear temporal shift is retained.
    """
    if not analyses:
        raise ValueError("No sweep temporal profiles available.")

    axes = [
        np.asarray(a['time_fs'], dtype=float)
        for a in analyses
    ]
    powers = [
        np.asarray(a['pulse_power_W'], dtype=float)
        for a in analyses
    ]

    range_mins = []
    range_maxs = []
    max_len = max(len(x) for x in axes)

    for x, p in zip(axes, powers):
        valid = np.isfinite(x) & np.isfinite(p)
        if not np.any(valid):
            continue

        xv = x[valid]
        pv = np.maximum(p[valid], 0.0)
        pmax = float(np.max(pv))

        if pmax > 0:
            significant = pv > 1e-6 * pmax
            if np.any(significant):
                range_mins.append(float(np.min(xv[significant])))
                range_maxs.append(float(np.max(xv[significant])))
                continue

        range_mins.append(float(np.min(xv)))
        range_maxs.append(float(np.max(xv)))

    if not range_mins:
        raise ValueError("No finite temporal profiles in sweep.")

    xmin = min(range_mins)
    xmax = max(range_maxs)
    if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
        raise ValueError("Invalid common temporal range for sweep.")

    # Keep enough temporal samples to preserve the native resolution while
    # avoiding unnecessarily huge visualization arrays.
    n_common = int(min(max(max_len, 512), 16384))
    common_time_fs = np.linspace(xmin, xmax, n_common)
    temporal_map = np.full(
        (len(analyses), n_common), np.nan, dtype=float)

    for i, (x, p) in enumerate(zip(axes, powers)):
        valid = np.isfinite(x) & np.isfinite(p)
        if np.count_nonzero(valid) < 2:
            continue

        xv = x[valid]
        pv = p[valid]
        order = np.argsort(xv)
        xv = xv[order]
        pv = pv[order]

        temporal_map[i] = np.interp(
            common_time_fs, xv, pv, left=np.nan, right=np.nan)

    # NP is fixed within a sweep, so raw arrays can be stacked directly.
    raw_axes = np.vstack(axes)
    raw_powers = np.vstack(powers)

    return common_time_fs, temporal_map, raw_axes, raw_powers



def build_sweep_compressed_temporal_map(
        sweep_result, mode='optimized', fixed_gdd_fs2=0.0, metric=None):
    """Build post-fiber compressed pulse profiles for every sweep point.

    Parameters
    ----------
    sweep_result : dict
        Result returned by :func:`run_parameter_sweep`.
    mode : {'optimized', 'fixed', 'swept'}
        Which compensating GDD to apply to each sweep point.
    fixed_gdd_fs2 : float
        Common GDD used when ``mode='fixed'``.
    metric : {'fwhm', 'outer_fwhm95', 'rms', 'energy95'} or None
        Duration metric.  None uses the sweep's optimization metric.

    Returns a dict containing the common time grid, pulse-power map, raw
    per-point profiles, applied GDD values, and durations.  The compressor is
    purely post-fiber; it never modifies the in-fiber propagation result.
    """
    mode = str(mode).lower()
    if mode not in ('optimized', 'fixed', 'swept'):
        raise ValueError("Unknown sweep compressor mode: %s" % mode)

    t_raw = np.asarray(sweep_result['output_t_s_raw'], dtype=float)
    f_raw = np.asarray(sweep_result['output_f_Hz_raw'], dtype=float)
    Uf_raw = np.asarray(sweep_result['output_Uf_raw'], dtype=complex)
    P0_raw = np.asarray(sweep_result['output_P0_W'], dtype=float)
    n = len(np.asarray(sweep_result['values']))

    if t_raw.shape[0] != n or f_raw.shape[0] != n or Uf_raw.shape[0] != n:
        raise ValueError("Inconsistent raw sweep field arrays.")

    if mode == 'optimized':
        gdds = np.asarray(sweep_result['optimized_gdd_fs2'], dtype=float)
    elif mode == 'fixed':
        gdds = np.full(n, float(fixed_gdd_fs2), dtype=float)
    else:
        if sweep_result.get('parameter_key') != 'output_GDD_fs2':
            raise ValueError(
                "Swept GDD is available only for a compensating-GDD sweep.")
        gdds = np.asarray(sweep_result['values'], dtype=float)

    metric = str(metric or sweep_result.get('optimize_metric', 'outer_fwhm95'))
    if metric not in ('fwhm', 'outer_fwhm95', 'rms', 'energy95'):
        raise ValueError("Unknown compressor metric: %s" % metric)

    profiles = []
    durations = np.full(n, np.nan, dtype=float)
    raw_powers = []

    for i in range(n):
        t = t_raw[i]
        f = f_raw[i]
        Uf = Uf_raw[i]
        P0 = float(P0_raw[i])
        omega_phys_fs = -2 * np.pi * f * 1e-15
        H = np.exp(+0.5j * float(gdds[i]) * omega_phys_fs ** 2)
        Ut = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Uf * H)))
        power = np.abs(Ut) ** 2 * P0
        metrics = pulse_temporal_metrics(t, power)
        durations[i] = _sweep_metric_value(metrics, metric)
        raw_powers.append(power)
        profiles.append({
            'time_fs': t * 1e15,
            'pulse_power_W': power,
        })

    (time_fs, power_map_W, time_fs_raw, power_raw_W) = \
        _build_common_sweep_temporal_map(profiles)

    return {
        'mode': mode,
        'applied_gdd_fs2': gdds,
        'duration_fs': durations,
        'time_fs': time_fs,
        'pulse_power_map_W': power_map_W,
        'time_fs_raw': time_fs_raw,
        'pulse_power_raw_W': power_raw_W,
    }

def _build_common_sweep_spectral_map(analyses):
    """Interpolate sweep spectra onto one common physical-frequency grid.

    Raw per-point axes/spectra are preserved separately in the sweep result.
    This common grid exists only for convenient pcolor visualization/export.

    The displayed union range is based on regions above 1e-6 of each spectrum's
    peak, preventing unused FFT-window extremes from dominating the map.
    """
    if not analyses:
        raise ValueError("No sweep spectra available.")

    axes = [
        np.asarray(a['spectral_frequency_THz'], dtype=float)
        for a in analyses
    ]
    spectra = [
        np.asarray(a['spectral_power_W_per_Hz'], dtype=float)
        for a in analyses
    ]

    range_mins = []
    range_maxs = []
    max_len = max(len(x) for x in axes)

    for x, p in zip(axes, spectra):
        valid = np.isfinite(x) & np.isfinite(p) & (x > 0)
        if not np.any(valid):
            continue
        xv = x[valid]
        pv = np.maximum(p[valid], 0.0)
        pmax = float(np.max(pv))
        if pmax > 0:
            significant = pv > 1e-6 * pmax
            if np.any(significant):
                range_mins.append(float(np.min(xv[significant])))
                range_maxs.append(float(np.max(xv[significant])))
                continue
        range_mins.append(float(np.min(xv)))
        range_maxs.append(float(np.max(xv)))

    if not range_mins:
        raise ValueError("No positive physical frequencies in sweep spectra.")

    xmin = min(range_mins)
    xmax = max(range_maxs)
    if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
        raise ValueError("Invalid common spectral range for sweep.")

    # Keep the native FFT point count (capped only at a very large value).
    n_common = int(min(max(max_len, 512), 16384))
    common_THz = np.linspace(xmin, xmax, n_common)
    spectral_map = np.full((len(analyses), n_common), np.nan, dtype=float)

    for i, (x, p) in enumerate(zip(axes, spectra)):
        valid = np.isfinite(x) & np.isfinite(p) & (x > 0)
        if np.count_nonzero(valid) < 2:
            continue
        xv = x[valid]
        pv = p[valid]
        order = np.argsort(xv)
        xv = xv[order]
        pv = pv[order]

        # np.interp cannot directly emit NaN outside the source range unless
        # left/right are specified.
        spectral_map[i] = np.interp(
            common_THz, xv, pv, left=np.nan, right=np.nan)

    # Raw grids have equal NP for a given sweep because n_exp is fixed.
    raw_axes = np.vstack(axes)
    raw_spectra = np.vstack(spectra)

    return common_THz, spectral_map, raw_axes, raw_spectra


def run_parameter_sweep(
        base_params, parameter_key, values,
        phase_fit_config=None, optimize_metric='outer_fwhm95',
        progress_cb=None
    ):
    """Run a parameter sweep and return phase/compression metrics.

    Fiber length is swept with one propagation to max(values).  A sweep of
    ``output_GDD_fs2`` is also special: the fiber is propagated only once,
    the best output GDD is optimized once, and each requested value is then
    applied as a phase-only compressor to that same output spectrum.  All
    other parameters run one independent propagation per requested value.
    """
    supported = {
        'pressure', 'fiber_length_cm', 'radius_um', 'energy_mJ',
        'tl_fwhm_fs', 'wavelength_nm', 'GDD_fs2', 'TOD_fs3',
        'beta2_fs2_m', 'beta3_fs3_m', 'alpha_1_m', 'gamma_1_W_m',
        'output_GDD_fs2',
    }
    if parameter_key not in supported:
        raise ValueError("Unsupported sweep parameter: %s" % parameter_key)

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Parameter sweep requires at least one value.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Sweep values must be finite.")

    if optimize_metric not in ('fwhm', 'outer_fwhm95', 'rms', 'energy95'):
        raise ValueError("Unknown compressor optimization metric.")

    output_gdd_sweep = (parameter_key == 'output_GDD_fs2')

    if parameter_key == 'fiber_length_cm':
        analyses = _run_single_pass_length_sweep(
            base_params, values, phase_fit_config, optimize_metric,
            progress_cb=progress_cb)
        single_pass_length = True

    elif output_gdd_sweep:
        # The fiber propagation is identical for every compensating GDD.
        # Propagate once, optimize the reference GDD once, then apply each
        # requested quadratic phase to the same output spectrum.
        params = dict(base_params)
        params['n_z_spectra'] = 2

        def propagation_progress(pc):
            if progress_cb is not None:
                progress_cb(int(round(0.60 * float(pc))))

        res = run_simulation(params, progress_cb=propagation_progress)
        base_analysis = _analyze_sweep_result(
            res, phase_fit_config, optimize_metric)
        if progress_cb is not None:
            progress_cb(75)

        analyses = []
        n = len(values)
        for i, gdd_fs2 in enumerate(values):
            applied = apply_gdd_to_output(res, float(gdd_fs2))
            analysis = dict(base_analysis)
            analysis['applied_gdd_fs2'] = float(gdd_fs2)
            # Store optimized and swept-GDD compression durations separately.
            analysis['swept_compressed_duration_fs'] = _sweep_metric_value(
                applied['metrics'], optimize_metric)
            analyses.append(analysis)

            if progress_cb is not None:
                progress_cb(int(round(75 + 25 * (i + 1) / n)))

        single_pass_length = False

    else:
        analyses = []
        n = len(values)
        for i, value in enumerate(values):
            params = dict(base_params)
            if parameter_key in (
                    'beta2_fs2_m', 'beta3_fs3_m', 'alpha_1_m',
                    'gamma_1_W_m'):
                params['coefficient_overrides'] = {
                    parameter_key: float(value)}
            else:
                params[parameter_key] = float(value)

            # Sweep tab does not need a detailed spectral-z visualization for
            # every independent simulation; keep only input/output snapshots.
            params['n_z_spectra'] = 2

            def subprogress(pc, i=i):
                if progress_cb is not None:
                    overall = ((i + float(pc) / 100.0) / n) * 100.0
                    progress_cb(int(round(overall)))

            res = run_simulation(params, progress_cb=subprogress)
            analyses.append(
                _analyze_sweep_result(
                    res, phase_fit_config, optimize_metric))

            if progress_cb is not None:
                progress_cb(int(round(100 * (i + 1) / n)))

        single_pass_length = False

    (time_fs,
     pulse_power_map_W,
     time_fs_raw,
     pulse_power_raw_W) = _build_common_sweep_temporal_map(
         analyses)

    (spectral_frequency_THz,
     spectral_power_map_W_per_Hz,
     spectral_frequency_THz_raw,
     spectral_power_raw_W_per_Hz) = _build_common_sweep_spectral_map(
         analyses)

    return {
        'parameter_key': parameter_key,
        'values': values,
        'phase_gdd_fs2': np.asarray(
            [a['phase_gdd_fs2'] for a in analyses], dtype=float),
        'optimized_gdd_fs2': np.asarray(
            [a['optimized_gdd_fs2'] for a in analyses], dtype=float),
        'actual_output_fwhm_fs': np.asarray(
            [a['actual_output_fwhm_fs'] for a in analyses], dtype=float),
        'tl_duration_fs': np.asarray(
            [a['tl_duration_fs'] for a in analyses], dtype=float),
        # Optimized compressor duration used by the sweep table and plots.
        'compressed_duration_fs': np.asarray(
            [a['compressed_duration_fs'] for a in analyses], dtype=float),
        'optimized_compressed_duration_fs': np.asarray(
            [a['compressed_duration_fs'] for a in analyses], dtype=float),
        'swept_compressed_duration_fs': np.asarray(
            [a.get('swept_compressed_duration_fs', np.nan)
             for a in analyses], dtype=float),
        'phase_fit_min_THz': np.asarray(
            [a['phase_fit_min_THz'] for a in analyses], dtype=float),
        'phase_fit_max_THz': np.asarray(
            [a['phase_fit_max_THz'] for a in analyses], dtype=float),
        'boundary_warning': np.asarray(
            [a['boundary_warning'] for a in analyses], dtype=bool),
        'boundary_warning_type': np.asarray(
            [a['boundary_warning_type'] for a in analyses], dtype=object),
        'boundary_warning_percent': np.asarray(
            [a['boundary_warning_percent'] for a in analyses], dtype=int),
        'boundary_warning_z_cm': np.asarray(
            [a['boundary_warning_z_cm'] for a in analyses], dtype=float),

        # Common interpolated temporal map for visualization.
        'time_fs': time_fs,
        'pulse_power_map_W': pulse_power_map_W,

        # Original time grids and uncompressed output pulse powers for each
        # sweep point, before interpolation.
        'time_fs_raw': time_fs_raw,
        'pulse_power_raw_W': pulse_power_raw_W,

        # Native complex output fields for post-fiber compressor rendering.
        # NP is fixed within a sweep, so these arrays can be stacked even when
        # the actual t/f values change (for example in a TL-duration sweep).
        'output_t_s_raw': np.vstack([a['t_s'] for a in analyses]),
        'output_f_Hz_raw': np.vstack([a['f_Hz'] for a in analyses]),
        'output_Uf_raw': np.vstack([a['Uf_complex'] for a in analyses]),
        'output_P0_W': np.asarray([a['P0_W'] for a in analyses], dtype=float),

        # Common interpolated spectral map for visualization.
        'spectral_frequency_THz': spectral_frequency_THz,
        'spectral_power_map_W_per_Hz': spectral_power_map_W_per_Hz,

        # Original physical-frequency grids and output spectra for each sweep
        # point, before interpolation.
        'spectral_frequency_THz_raw': spectral_frequency_THz_raw,
        'spectral_power_raw_W_per_Hz': spectral_power_raw_W_per_Hz,

        'optimize_metric': optimize_metric,
        'phase_fit_config': dict(phase_fit_config or {}),
        'single_pass_length_sweep': single_pass_length,
        'single_pass_output_gdd_sweep': output_gdd_sweep,
        'applied_gdd_fs2': (
            values.copy() if output_gdd_sweep
            else np.full(values.shape, np.nan, dtype=float)),
    }

