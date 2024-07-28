import numpy as np
import matplotlib.pyplot as plt
# import librosa
import soundfile as sf


class PLL:
    def __init__(self, gamma, T, filter_type='gain', gain=1, integrator_gain=0, initial_freq=1):
        self.gamma = gamma
        self.T = T
        self.filter_type = filter_type
        self.gain = gain
        self.integrator_gain = integrator_gain
        self.phase = 0
        self.freq = initial_freq
        self.integrator_state = 0

    def vco(self, t):
        return np.sin(2 * np.pi * self.freq * t + self.phase)

    def phase_comparator(self, O, x):
        return np.mean(O * x)

    def loop_filter(self, phase_error):
        if self.filter_type == 'gain':
            return self.gain * phase_error
        elif self.filter_type == 'integrator':
            self.integrator_state += self.integrator_gain * phase_error
            return self.gain * phase_error + self.integrator_state

    def update(self, O, t, dt):
        x = self.vco(t)
        phase_error = self.phase_comparator(O, x)
        freq_adjustment = self.loop_filter(phase_error)
        self.freq += self.gamma * freq_adjustment
        self.phase += 2 * np.pi * self.freq * dt
        return x


def onset_detection(signal):
    energy = signal.astype(np.float64) ** 2
    energy_diff = np.diff(energy)
    onset_env = np.maximum(energy_diff, 0)

    max_onset = np.max(onset_env)
    if max_onset > 0:
        onset_env = onset_env / max_onset
    else:
        onset_env = np.zeros_like(onset_env, dtype=np.float64)

    onset_env -= np.mean(onset_env)
    onset_env = np.interp(np.linspace(0, 1, len(signal)), np.linspace(0, 1, len(onset_env)), onset_env)
    return onset_env


def run_pll(onset_env, sr, pll, duration):
    t = np.arange(0, duration, 1 / sr)
    dt = 1 / sr
    phase_errors = np.zeros_like(t)
    vco_outputs = np.zeros_like(t)

    for i in range(len(t)):
        vco_outputs[i] = pll.update(onset_env[i], t[i], dt)
        phase_errors[i] = pll.loop_filter(pll.phase_comparator(onset_env[i], vco_outputs[i]))

    return phase_errors, vco_outputs


def plot_results(t, onset_env, phase_errors, vco_outputs, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    ax1.plot(t, onset_env)
    ax1.set_title('Onset Envelope')
    ax1.set_xlabel('Time (s)')
    ax2.plot(t, phase_errors)
    ax2.set_title('Phase Error')
    ax2.set_xlabel('Time (s)')
    ax3.plot(t, vco_outputs)
    ax3.set_title('VCO Output')
    ax3.set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_sine_wave(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def generate_square_wave(freq, duration, sr, duty_cycle=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.where((t * freq) % 1 < duty_cycle, 1, -1)


def test_pll(input_signal, sr, pll, duration, title):
    onset_env = onset_detection(input_signal)
    t = np.arange(0, duration, 1 / sr)
    phase_errors, vco_outputs = run_pll(onset_env, sr, pll, duration)
    plot_results(t, onset_env, phase_errors, vco_outputs, title)
    return phase_errors, vco_outputs


# Parameters
sr = 1000
duration = 5
base_freq = 2

# Generate test signals
sine_wave = generate_sine_wave(base_freq, duration, sr)
square_wave_50 = generate_square_wave(base_freq, duration, sr, duty_cycle=0.5)
square_wave_10 = generate_square_wave(base_freq, duration, sr, duty_cycle=0.1)

# Create PLL instances with initial frequency
pll_gain = PLL(gamma=0.1, T=0.1, filter_type='gain', gain=0.5, initial_freq=base_freq)
pll_integrator = PLL(gamma=0.1, T=0.1, filter_type='integrator', gain=0.5, integrator_gain=0.1, initial_freq=base_freq)

# Test with sine wave
print("Testing with Sine Wave:")
test_pll(sine_wave, sr, pll_gain, duration, 'PLL with Gain Filter - Sine Wave Input')
test_pll(sine_wave, sr, pll_integrator, duration, 'PLL with Integrator Filter - Sine Wave Input')

# Test with square wave (50% duty cycle)
print("Testing with Square Wave (50% duty cycle):")
test_pll(square_wave_50, sr, pll_gain, duration, 'PLL with Gain Filter - Square Wave 50% Input')
test_pll(square_wave_50, sr, pll_integrator, duration, 'PLL with Integrator Filter - Square Wave 50% Input')

# Test with square wave (10% duty cycle)
print("Testing with Square Wave (10% duty cycle):")
test_pll(square_wave_10, sr, pll_gain, duration, 'PLL with Gain Filter - Square Wave 10% Input')
test_pll(square_wave_10, sr, pll_integrator, duration, 'PLL with Integrator Filter - Square Wave 10% Input')

# Test with real audio file
# audio_file = 'path_to_your_audio_file.wav'
# audio, sr = librosa.load(audio_file)
#
# print("Testing with real audio:")
# phase_errors, vco_outputs = test_pll(audio, sr, pll_gain, len(audio) / sr, 'PLL with Gain Filter - Real Audio Input')
# test_pll(audio, sr, pll_integrator, len(audio) / sr, 'PLL with Integrator Filter - Real Audio Input')
#
# # Generate metronome clicks
# clicks = librosa.clicks(times=np.where(vco_outputs > 0.9)[0] / sr, sr=sr, length=len(audio))
#
# # Mix original audio with metronome clicks
# output = audio + clicks
#
# # Save output
# sf.write('output_with_metronome.wav', output, sr)