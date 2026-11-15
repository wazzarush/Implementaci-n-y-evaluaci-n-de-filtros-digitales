"""
utils_filtros.py
Funciones auxiliares compartidas por los scripts del proyecto:
espectro de amplitud, SNR, atenuación en frecuencias específicas y estilo de gráficas.
"""
import numpy as np
import matplotlib.pyplot as plt

# Estilo general de las gráficas
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
AZUL, NARANJA, VERDE, ROJO, GRIS = "#1f77b4", "#d95f02", "#2ca02c", "#c0392b", "#7f7f7f"


def espectro_amplitud(x, fs):
    """Devuelve (frecuencias, amplitud) del espectro unilateral usando la FFT.
    Se normaliza para que un seno de amplitud A aparezca con altura A."""
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    amp = 2 * np.abs(X) / np.sum(np.hanning(n))   # corrección de la ventana
    return np.fft.rfftfreq(n, 1 / fs), amp


def amplitud_en(x, fs, f0, ancho=2.0):
    """Amplitud estimada de la componente cercana a f0 (máximo en f0 ± ancho Hz)."""
    f, a = espectro_amplitud(x, fs)
    return a[(f >= f0 - ancho) & (f <= f0 + ancho)].max()


def snr_db(referencia, senal):
    """Relación señal-ruido (dB) de 'senal' respecto de una referencia limpia."""
    ruido = senal - referencia
    return 10 * np.log10(np.sum(referencia ** 2) / np.sum(ruido ** 2))


def rmse(referencia, senal):
    """Error cuadrático medio (raíz) entre la señal y la referencia limpia."""
    return np.sqrt(np.mean((senal - referencia) ** 2))
