import numpy as np

from rlquantopt.utils.analysis_utils import get_pulse_data, get_pulse_spectrum

pulse_file = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/results/clip_best/rl_model_12566528_steps_ZCQPEE_pl-1000_T-50ns_delta_mode_ep0_clip_best.csv'
tlist, amps = get_pulse_data(pulse_file, verbose=True)


# Define the bandpass filter around 6.66 GHz with a specified width
def bandpass_filter(freqs, amplitudes, center_freq, width):
    low_cutoff = center_freq - width
    high_cutoff = center_freq + width
    filtered_amplitudes = np.where((freqs >= low_cutoff) & (freqs <= high_cutoff), amplitudes, 0)
    return filtered_amplitudes


# Compute FFT to filter frequencies
def compute_filtered_pulse(tlist, amplist, center_freq, width):
    # Perform FFT
    freq_domain = np.fft.fft(amplist)
    freq_domain_shifted = np.fft.fftshift(freq_domain)

    # Get frequency axis
    freq_axis = np.fft.fftfreq(len(freq_domain), d=(tlist[1] - tlist[0]))
    freq_axis_shifted = np.fft.fftshift(freq_axis)

    # Apply bandpass filter
    filtered_freq_domain = bandpass_filter(freq_axis_shifted, freq_domain_shifted, center_freq, width)

    # Inverse FFT to get time-domain signal
    filtered_freq_domain_unshifted = np.fft.ifftshift(filtered_freq_domain)
    filtered_time_signal = np.fft.ifft(filtered_freq_domain_unshifted).real

    return tlist, filtered_time_signal

