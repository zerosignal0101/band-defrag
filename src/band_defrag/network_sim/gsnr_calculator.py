from itertools import islice
from typing import List, Literal, Union, Optional, Annotated, Tuple
import numpy as np
from numpy.typing import NDArray
import math
import random
from numba import njit, prange


# one_link_transmission references
# [1] A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems
# [2] Modeling and mitigation of fiber nonlinearity in wideband optical signal transmission

# 1. one_link_transmission function
@njit(nogil=True, cache=True)
def get_alpha(frequency: float) -> float:
    """
    计算不同频率处的alpha值
    """
    wavelength0 = 1550e-9
    wavelength = 3e8 / frequency
    alpha0 = 0.162
    alpha1 = -7.3764e-5
    alpha2 = 3.7685e-6
    alpha = alpha2 * (wavelength - wavelength0) ** 2 + alpha1 * (wavelength - wavelength0) + alpha0
    return alpha / 4.343 / 1e3


@njit(nogil=True, cache=True)
def get_r_f(frequency: float, P_total: float) -> float:
    """
    计算SRS的中间过程
    """
    delta_f = 15e12
    f_m1 = 184.325e12
    f_M1 = 190.325e12
    f_m2 = 190.675e12
    f_M2 = 196.675e12
    B_t = 12e12
    if frequency - delta_f < f_m1 and frequency + delta_f > f_M1:
        return P_total * frequency
    elif frequency - delta_f > f_m1 and frequency + delta_f < f_M1:
        return 0.0
    elif frequency - delta_f < f_m1 and frequency + delta_f < f_M1:
        return P_total / B_t * (frequency ** 2 / 2 - frequency * f_m1 + (f_M1 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f > f_m1 and frequency + delta_f > f_M1:
        return P_total / B_t * (frequency * f_M1 - frequency ** 2 / 2 - (f_m1 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f < f_m2 and frequency + delta_f > f_M2:
        return P_total * frequency
    elif frequency - delta_f > f_m2 and frequency + delta_f < f_M2:
        return 0.0
    elif frequency - delta_f < f_m2 and frequency + delta_f < f_M2:
        return P_total / B_t * (frequency ** 2 / 2 - frequency * f_m2 + (f_M2 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f > f_m2 and frequency + delta_f > f_M2:
        return P_total / B_t * (frequency * f_M2 - frequency ** 2 / 2 - (f_m2 ** 2 - delta_f ** 2) / 2)
    else:
        return 0.0


@njit(parallel=True, nogil=True, cache=True)
def get_Pi_z(
        distance: float, P_total: float,
        frequencies: NDArray[np.float64], Power: NDArray[np.float64],
        i: int, channels: int
) -> float:  # Changed return type from List[float] to float
    """
    计算经过distance传输后各信道的功率值
    """
    C_r = 0.028 / 1e3 / 1e12
    alpha = get_alpha(frequencies[i])
    L_eff = (1 - np.exp(-alpha * distance)) / alpha
    r_f = get_r_f(frequencies[i], P_total)
    contribution_sum = 0.0

    for j in prange(channels):
        alpha_j = get_alpha(frequencies[j])
        L_eff_j = (1 - np.exp(-alpha_j * distance)) / alpha_j
        r_f_j = get_r_f(frequencies[j], P_total)
        # 确保每一步都是标量操作
        power_j = Power[j]
        exp_term = np.exp(-C_r * L_eff_j * r_f_j)
        contribution = power_j * exp_term  # 将乘积转换为标量

        contribution_sum += contribution

    pi_z_result = Power[i] * (np.exp(-C_r * L_eff * r_f) * P_total) / contribution_sum
    return pi_z_result


@njit(nogil=True, cache=True)
def lin2db(value: float) -> float:
    '''
    linear -> dB
    '''
    return 10 * np.log10(value)


def watt2dbm(value: float) -> float:  # Not njit, but calls njit lin2db
    '''
    W -> dBm
    '''
    return lin2db(value * 1e3)


@njit(parallel=True, nogil=True, cache=True)
def calculate_ASE_noise(
        Att: NDArray[np.float64], fi: NDArray[np.float64],
        Bch: NDArray[np.float64], distance: float
) -> NDArray[np.float64]:
    channels, n = fi.shape
    c = 3e8
    n_sp = 1.41  # n_sp = NF(4.5dB)/2
    h = 6.62607015e-34
    Ref_Lambda = 1575e-9
    num_of_spans = int(np.ceil(distance / 100e3))
    Length = 100 * 1e3 * np.ones(num_of_spans)
    single_ASE = np.zeros((channels, n))

    for j in prange(n):
        for i in prange(channels):
            a_i = Att[i, j]  # \alpha of COI in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            length_i = Length[j]
            single_ASE[i, j] = 2 * n_sp * h * (f_i + c / Ref_Lambda) * B_i * (np.exp(a_i * length_i) - 1)

    return np.sum(single_ASE, axis=1)


@njit(parallel=True, nogil=True, cache=True)
def _numba_one_link_transmission(
        distance: float, channels: int, Power: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    计算链路 ASE 噪声和调整后的 Power（考虑信号再分配）。
    输入的 Power 不会被修改。
    """
    # 复制 Power 避免副作用
    Power_out = np.copy(Power)
    P_total = np.sum(Power_out)

    for i in prange(channels):
        Power_out[i] = get_Pi_z(100e3, P_total, frequencies, Power, i, channels)

    num_of_spans = int(np.ceil(distance / 100e3))
    RefFreq = 190.5e12
    # fi here is only (channels, 1) and will be tiled later in one_link_transmission for calculate_ASE_noise
    fi_for_ase = np.array([(f - RefFreq) for f in frequencies]).reshape(-1, 1)  # This fi is temporary

    Bch = 150e9 * np.ones((channels, num_of_spans))
    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))

    # TODO: Check if the fi need tiling
    # # This `fi` is for `calculate_ASE_noise`
    # fi_reshaped_for_ase = (frequencies - RefFreq).reshape(-1, 1)  # (channels, 1)
    # # Tile it to (channels, num_of_spans)
    # fi_tiled_for_ase = tile_implementation(fi_reshaped_for_ase, num_of_spans)

    # Now call calculate_ASE_noise with the correctly shaped fi_tiled_for_ase
    ASE_noise = calculate_ASE_noise(Att, fi_for_ase, Bch, distance)

    return ASE_noise, Power_out


@njit(nogil=True, cache=True)
def cal_eps(B_i: float, f_i: float, a_i: float, mean_L: float, beta2: NDArray[np.float64],
            beta3: NDArray[np.float64]) -> float:
    # beta2 and beta3 are expected to be 1D arrays here from the caller, so np.mean is appropriate.
    return (3 / 10) * np.log(1 + (6 / a_i) / (
            mean_L * np.arcsinh(
        np.pi ** 2 / 2 * abs(np.mean(beta2) + 2 * np.pi * np.mean(beta3) * f_i) / a_i * B_i ** 2)))


@njit(nogil=True, cache=True)
def cal_SPM(phi_i: float, T_i: float, B_i: float, a: float, a_bar: float, gamma: float) -> float:
    return 4 / 9 * gamma ** 2 / B_i ** 2 * np.pi / (phi_i * a_bar * (2 * a + a_bar)) \
        * ((T_i - a ** 2) / a * np.arcsinh(phi_i * B_i ** 2 / a / np.pi) + ((a + a_bar) ** 2 - T_i) / (
                a + a_bar) * np.arcsinh(
            phi_i * B_i ** 2 / (a + a_bar) / np.pi))


@njit(nogil=True, cache=True)
def cal_XPM(Pi: float, Pk: NDArray[np.float64], phi_ik: NDArray[np.float64], T_k: NDArray[np.float64],
            B_i: float, B_k: NDArray[np.float64], a: NDArray[np.float64], a_bar: NDArray[np.float64],
            gamma: float) -> float:
    if Pi == 0:
        return 0
    else:
        # Changed sum(...) to np.sum(...) for Numba efficiency
        return 32 / 27 * np.sum((Pk / Pi) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar * (2 * a + a_bar))
                                * ((T_k - a ** 2) / a * np.arctan(phi_ik * B_i / a)
                                   + ((a + a_bar) ** 2 - T_k) / (a + a_bar) * np.arctan(phi_ik * B_i / (a + a_bar)))
                                )


@njit(parallel=True, nogil=True, cache=True)
def calculate_NLI_noise(
        Att: NDArray[np.float64], Att_bar: NDArray[np.float64], Cr: NDArray[np.float64],
        Pch: NDArray[np.float64], fi: NDArray[np.float64], Bch: NDArray[np.float64],
        Length: NDArray[np.float64], D: NDArray[np.float64], S: NDArray[np.float64],
        gamma: NDArray[np.float64], RefLambda: float
) -> NDArray[np.float64]:
    """
    Returns nonlinear interference power and coefficient for each WDM
    channel.
    """
    channels, n = Att.shape

    c = 3e8

    a = Att
    a_bar = Att_bar
    L = Length
    P_ij = Pch
    Ptot = np.sum(P_ij, axis=0)

    beta2 = -D * RefLambda ** 2 / (2 * np.pi * c)
    beta3 = RefLambda ** 2 / (2 * np.pi * c) ** 2 * (RefLambda ** 2 * S + 2 * RefLambda * D)

    # Average Coherence Factor
    mean_att_i = np.empty(channels, dtype=np.float64)
    for i in range(channels):
        mean_att_i[i] = np.mean(a[i, :])
    mean_L = np.mean(L)  # average fiber length

    eta_SPM = np.zeros((channels, n))
    eta_XPM = np.zeros((channels, n))

    for j in prange(n):
        """ Calculation of nonlinear interference (NLI) power in fiber span j """
        for i in prange(channels):
            """ Compute the NLI of each COI """
            not_i_mask = np.arange(channels) != i  # Using a mask for clarity when slicing
            a_i = a[i, j]  # \alpha of COI in fiber span j
            a_k = a[not_i_mask, j]  # \alpha of INT in fiber span j
            a_bar_i = a_bar[i, j]  # \bar{\alpha} of COI in fiber span j
            a_bar_k = a_bar[not_i_mask, j]  # \bar{\alpha} of INT in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            f_k = fi[not_i_mask, j]  # f_k of INT in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            B_k = Bch[not_i_mask, j]  # B_k of INT in fiber span j
            Cr_i = Cr[i, j]  # Cr  of COI in fiber span j
            Cr_k = Cr[not_i_mask, j]  # Cr  of INT in fiber span j
            P_i = P_ij[i, j]  # P_i of COI in fiber span j
            P_k = P_ij[not_i_mask, j]  # P_k of INT in fiber span j

            phi_i = 3 / 2 * np.pi ** 2 * (beta2[j] + np.pi * beta3[j] * (f_i + f_i))  # \phi_i of COI in fiber span j
            phi_ik = 2 * np.pi ** 2 * (f_k - f_i) * (
                    beta2[j] + np.pi * beta3[j] * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j

            if Ptot[j] != 0:  # Avoid division by zero if total power is zero
                T_i = (a_i + a_bar_i - f_i * Ptot[j] * Cr_i) ** 2  # T_i of COI in fiber span j
                T_k = (a_k + a_bar_k - f_k * Ptot[j] * Cr_k) ** 2  # T_k of INT in fiber span j
            else:
                T_i = (a_i + a_bar_i) ** 2
                T_k = (a_k + a_bar_k) ** 2

            eta_SPM[i, j] = cal_SPM(
                phi_i, T_i, B_i, a_i, a_bar_i, gamma[j]
            ) * n ** cal_eps(B_i, f_i, mean_att_i[i], mean_L, beta2,
                             beta3)  # computation of SPM contribution in fiber span j

            eta_XPM[i, j] = cal_XPM(P_i, P_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k,
                                    gamma[j])  # computation of XPM contribution in fiber span j

    nonzero_mask = P_ij[:, 0] != 0  # Create a boolean mask for channels with non-zero initial power
    eta_n = np.zeros(channels)  # Fixed typo: np.np.zeros to np.zeros
    eta_n[nonzero_mask] = np.sum(eta_SPM[nonzero_mask] + eta_XPM[nonzero_mask], axis=1)

    # computation of NLI normalized to transmitter power, see Ref. [1, Eq. (5)]
    NLI = P_ij[:, 0] ** 3 * eta_n  # Ref. [1, Eq. (1)]

    # print("NLI:", NLI)
    return NLI


@njit(parallel=True, nogil=True, cache=True)
def calculate_GSNR(
        Power: NDArray[np.float64], noise: NDArray[np.float64], channels: int
) -> NDArray[np.float64]:
    GSNR = np.zeros(channels)
    for i in prange(channels):
        if Power[i] != 0 and noise[i] != 0:  # Added check for noise[i] != 0
            GSNR[i] = lin2db(Power[i] / noise[i])
        else:
            GSNR[i] = 0
    return GSNR


# Numba-optimized tiling function
@njit(parallel=True, nogil=True, cache=True)
def tile_implementation(arr: NDArray[np.float64], num_columns: int) -> NDArray[np.float64]:
    rows = arr.shape[0]
    expanded = np.empty((rows, num_columns))
    for i in prange(rows):
        for j in prange(num_columns):
            expanded[i, j] = arr[i, 0]  # Copy the value from the first column for each row
    return expanded


def one_link_transmission(
        distance: float, channels: int, Power: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    单链路传输计算，返回链路上每个信道的 GSNR。
    不修改原始 Power 数组，避免副作用。
    '''
    # Get initial channel power adjusted for SRS equivalent and ASE noise from one Numba call
    # This also handles the fi tiling required for calculate_ASE_noise internally now.
    ASE_noise, adjusted_Power = _numba_one_link_transmission(
        distance, channels, Power, frequencies
    )

    # Physical parameters setting
    num_of_spans = math.ceil(distance / 100e3)
    Ref_Freq = 190.5e12  # Hz

    # Optimization: Direct NumPy array creation for fi
    fi_temp_unshaped = frequencies - Ref_Freq
    fi_for_nli = tile_implementation(fi_temp_unshaped.reshape(-1, 1), num_of_spans)  # shape: (channels, num_of_spans)

    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans), dtype=np.float64)
    Att_bar = Att
    Cr = 0.028 / 1e3 / 1e12 * np.ones((channels, num_of_spans), dtype=np.float64)

    # Using adjusted_Power to generate Pch
    # This converts channel power != 0 to 5 dBm (0.0031622776602 W) for NLI calculation if active
    tmpPower_NLI = 0.0031622776602 * ((adjusted_Power != 0).astype(np.float64))
    Pch_for_nli = tile_implementation(tmpPower_NLI.reshape(channels, 1), num_of_spans)

    Bch = np.full((channels, num_of_spans), 150e9, dtype=np.float64)
    Length = 100e3 * np.ones(num_of_spans, dtype=np.float64)
    D = 17e-12 / 1e-9 / 1e3 * np.ones(num_of_spans, dtype=np.float64)
    S = 0.067e-12 / 1e-9 / 1e3 / 1e-9 * np.ones(num_of_spans, dtype=np.float64)
    gamma = 1.21 / 1e3 * np.ones(num_of_spans, dtype=np.float64)
    RefLambda = 1575e-9

    # Calculate Nonlinear Interference noise
    NLI_noise = calculate_NLI_noise(Att, Att_bar, Cr, Pch_for_nli, fi_for_nli, Bch, Length, D, S, gamma, RefLambda)

    # Total noise
    noise = NLI_noise + ASE_noise

    # Calculate GSNR
    GSNR = calculate_GSNR(adjusted_Power, noise, channels)

    return adjusted_Power, GSNR
