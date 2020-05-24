import math as m
import random as r
import numpy as np
import matplotlib.pyplot as plt

def generate_random_signal(max_omega, harmonics):
    amplitudes = [r.randint(0, 10) for _ in range(harmonics)]
    phases = [r.random() * 2 * np.pi for _ in range(harmonics)]
    def s(t):
        x = 0
        for k in range(harmonics):
            x += amplitudes[k] * m.sin(max_omega * (k + 1) / harmonics * t + phases[k])
        return x

    return np.vectorize(s)

def dft(values):
    n = len(values)
    p = np.arange(n)
    k = p.reshape((n, 1))
    w = np.exp(-2j * np.pi * p * k / n)
    return np.dot(w, values)

def fft(values):
    signal = np.asarray(values, dtype=float)
    N = len(signal)
    if N <= 2:
        return dft(signal)
    else:
        signal_even = fft(signal[::2])
        signal_odd = fft(signal[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([signal_even + terms[:N // 2] * signal_odd,
                               signal_even + terms[N // 2:] * signal_odd])

signal = generate_random_signal(1100, 12)
t = np.linspace(0, 5, 64)
x = signal(t)
x_t = list(range(0, 64))
x_dft = dft(x)
x_fft = fft(x)
x_dft_re = x_dft.real
x_dft_im = x_dft.imag
x_fft_re = x_fft.real
x_fft_im = x_fft.imag

_, (chart_signal, chart_DFT, chart_FFT) = plt.subplots(3, 1, figsize=(20, 20))

chart_signal.set_title("Random signal")
chart_signal.plot(t, x, 'k')
chart_DFT.set_title("DFT")
chart_DFT.plot(t, x_dft_re, 'g', t, x_dft_im, 'k')
chart_FFT.set_title("FFT")
chart_FFT.plot(t, x_fft_re, 'g', t, x_fft_im, 'k')

plt.show()
