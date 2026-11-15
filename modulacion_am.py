#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actividad formativa 4 - Implementación y análisis de un sistema de modulación
en amplitud (AM) con Python (NumPy, SciPy, Matplotlib).

Estructura:
  1. Definición de la señal de entrada (mensaje) y la portadora
  2. Implementación de la modulación AM (DSB-LC, con portadora)
  3. Visualización en tiempo y frecuencia
  4. Análisis de ruido, atenuación, distorsión y sobremodulación
  5. Demodulación (detector de envolvente) y cálculo de métricas

Ejecución:  python modulacion_am.py
Genera las gráficas en la carpeta ./figuras
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # permite correr sin pantalla
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)            # semilla fija: resultados reproducibles

# ----------------------------------------------------------------------------
# 1. PARÁMETROS Y SEÑAL DE ENTRADA
# ----------------------------------------------------------------------------
fs = 100_000          # frecuencia de muestreo [Hz]
T = 0.02              # duración [s]
t = np.arange(0, T, 1 / fs)

fm = 500              # frecuencia del mensaje [Hz]  (baja frecuencia)
fc = 10_000           # frecuencia de la portadora [Hz] (alta frecuencia)
Am = 1.0              # amplitud del mensaje
Ac = 2.0              # amplitud de la portadora
# Índice de modulación mu = Am/Ac = 0.5  (modulación 50 %)

m = Am * np.cos(2 * np.pi * fm * t)        # señal de información
c = Ac * np.cos(2 * np.pi * fc * t)        # portadora


# ----------------------------------------------------------------------------
# 2. FUNCIONES DEL SISTEMA
# ----------------------------------------------------------------------------
def modular_am(m, Ac, fc, t, mu=None):
    """s(t) = Ac [1 + mu * m_n(t)] cos(2 pi fc t), con m_n normalizado a |m|<=1."""
    m_n = m / np.max(np.abs(m))
    if mu is None:
        mu = np.max(np.abs(m)) / Ac
    return Ac * (1 + mu * m_n) * np.cos(2 * np.pi * fc * t), mu


def espectro(x, fs):
    """Espectro de amplitud de un lado (FFT con ventana de Hann)."""
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x * w) / (np.sum(w) / 2)
    f = np.fft.rfftfreq(N, 1 / fs)
    return f, np.abs(X)


def agregar_ruido(s, snr_db):
    """Ruido blanco gaussiano aditivo (AWGN) para una SNR dada en dB."""
    p_senal = np.mean(s ** 2)
    p_ruido = p_senal / (10 ** (snr_db / 10))
    return s + rng.normal(0, np.sqrt(p_ruido), len(s))


def demodular_envolvente(s, fs, fc_corte=2_000):
    """Detector de envolvente: |señal analítica| -> filtro pasa-bajas -> quitar DC."""
    env = np.abs(hilbert(s))
    b, a = butter(4, fc_corte / (fs / 2))
    env = filtfilt(b, a, env)
    return env - np.mean(env)


def distorsion_no_lineal(s, a1=1.0, a2=0.0, a3=-0.03):
    """Amplificador no lineal: y = a1 s + a2 s^2 + a3 s^3 (distorsión armónica)."""
    return a1 * s + a2 * s ** 2 + a3 * s ** 3


def correlacion(m, m_rec):
    """Coeficiente de correlación entre mensaje original y recuperado."""
    return np.corrcoef(m, m_rec)[0, 1]


def snr_recuperada_db(m, m_rec):
    """SNR a la salida del demodulador (ajustando la ganancia óptima)."""
    g = np.dot(m, m_rec) / np.dot(m_rec, m_rec)
    err = m - g * m_rec
    return 10 * np.log10(np.sum(m ** 2) / np.sum(err ** 2))


# ----------------------------------------------------------------------------
# 3. MODULACIÓN BASE Y GRÁFICAS (tiempo y frecuencia)
# ----------------------------------------------------------------------------
s, mu = modular_am(m, Ac, fc, t)
print(f"Índice de modulación mu = {mu:.2f}")

# Fig 1: señal de entrada (mensaje) y portadora
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(t * 1e3, m, color="tab:blue")
ax[0].set_title(f"Señal de entrada (mensaje) m(t): fm = {fm} Hz")
ax[0].set_ylabel("Amplitud [V]")
ax[1].plot(t * 1e3, c, color="tab:orange", lw=0.8)
ax[1].set_title(f"Portadora c(t): fc = {fc/1e3:.0f} kHz")
ax[1].set_xlabel("Tiempo [ms]"); ax[1].set_ylabel("Amplitud [V]")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/01_mensaje_portadora.png", dpi=150); plt.close()

# Fig 2: señal modulada en el tiempo con envolvente
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t * 1e3, s, lw=0.8, label="s(t) modulada")
ax.plot(t * 1e3, Ac * (1 + mu * m / Am), "r--", lw=1.5, label="Envolvente")
ax.plot(t * 1e3, -Ac * (1 + mu * m / Am), "r--", lw=1.5)
ax.set_title(f"Señal AM en el tiempo (mu = {mu:.2f})")
ax.set_xlabel("Tiempo [ms]"); ax.set_ylabel("Amplitud [V]")
ax.legend(loc="upper right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/02_am_tiempo.png", dpi=150); plt.close()

# Fig 3: espectros (mensaje y AM)
fm_, Xm = espectro(m, fs)
fs_, Xs = espectro(s, fs)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(fm_ / 1e3, Xm); ax[0].set_xlim(0, 3)
ax[0].set_title("Espectro del mensaje"); ax[0].set_xlabel("Frecuencia [kHz]")
ax[0].set_ylabel("Amplitud [V]")
ax[1].plot(fs_ / 1e3, Xs); ax[1].set_xlim(7, 13)
ax[1].set_title("Espectro de la señal AM")
ax[1].set_xlabel("Frecuencia [kHz]")
for f0, lab in [(fc - fm, "fc−fm"), (fc, "fc"), (fc + fm, "fc+fm")]:
    ax[1].annotate(lab, (f0 / 1e3, Xs[np.argmin(np.abs(fs_ - f0))]),
                   textcoords="offset points", xytext=(0, 6), ha="center")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/03_espectros.png", dpi=150); plt.close()

# Potencias teóricas: Pc = Ac^2/2 ; Pbl = Pc*mu^2/2 ; eficiencia = mu^2/(2+mu^2)
Pc = Ac ** 2 / 2
eficiencia = mu ** 2 / (2 + mu ** 2)
print(f"Potencia portadora = {Pc:.2f} W | eficiencia = {100*eficiencia:.1f} %")

# ----------------------------------------------------------------------------
# 4. ANÁLISIS DE ESCENARIOS
# ----------------------------------------------------------------------------
# 4.1 Efecto del índice de modulación (sub, crítica, sobremodulación)
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
resultados_mu = []
for a_, mu_i, nombre in zip(ax, [0.5, 1.0, 1.5],
                            ["Submodulación (mu=0.5)", "Modulación 100 % (mu=1.0)",
                             "Sobremodulación (mu=1.5)"]):
    s_i, _ = modular_am(m, Ac, fc, t, mu=mu_i)
    a_.plot(t * 1e3, s_i, lw=0.7)
    a_.plot(t * 1e3, Ac * (1 + mu_i * m / Am), "r--", lw=1.2)
    a_.set_title(nombre); a_.set_ylabel("V"); a_.grid(alpha=.3)
    r = demodular_envolvente(s_i, fs)
    r_ok = correlacion(m, r)
    # En sobremodulación la envolvente real es |1+mu*m|, no 1+mu*m -> distorsión
    resultados_mu.append((mu_i, r_ok, snr_recuperada_db(m, r)))
ax[-1].set_xlabel("Tiempo [ms]")
plt.tight_layout(); plt.savefig(f"{OUT}/04_indice_modulacion.png", dpi=150); plt.close()

# 4.2 Ruido AWGN con distintas SNR
snrs = [30, 15, 5]
fig, ax = plt.subplots(len(snrs), 2, figsize=(12, 7))
resultados_ruido = []
for i, snr in enumerate(snrs):
    sn = agregar_ruido(s, snr)
    f_, Xn = espectro(sn, fs)
    ax[i, 0].plot(t * 1e3, sn, lw=0.5)
    ax[i, 0].set_title(f"Señal AM con ruido, SNR = {snr} dB")
    ax[i, 0].set_ylabel("V")
    ax[i, 1].plot(f_ / 1e3, 20 * np.log10(Xn + 1e-9), lw=0.6)
    ax[i, 1].set_xlim(0, 25); ax[i, 1].set_ylim(-80, 10)
    ax[i, 1].set_title("Espectro [dB]"); ax[i, 1].set_ylabel("dB")
    r = demodular_envolvente(sn, fs)
    resultados_ruido.append((snr, correlacion(m, r), snr_recuperada_db(m, r)))
ax[-1, 0].set_xlabel("Tiempo [ms]"); ax[-1, 1].set_xlabel("Frecuencia [kHz]")
for a_ in ax.ravel(): a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/05_ruido.png", dpi=150); plt.close()

# 4.3 Atenuación del canal
atenuaciones = [1.0, 0.5, 0.1]   # factor de amplitud del canal
fig, ax = plt.subplots(figsize=(9, 3.8))
resultados_att = []
for k in atenuaciones:
    # Ruido térmico FIJO en el receptor (no depende de la atenuación)
    sigma = np.sqrt(np.mean(s ** 2) / 10 ** (20 / 10))
    sk = k * s + rng.normal(0, sigma, len(s))
    r = demodular_envolvente(sk, fs)
    snr_in = 10 * np.log10(np.mean((k * s) ** 2) / sigma ** 2)
    resultados_att.append((k, snr_in, correlacion(m, r), snr_recuperada_db(m, r)))
    ax.plot(t * 1e3, sk, lw=0.6, label=f"Atenuación x{k}  (SNR entrada {snr_in:.0f} dB)")
ax.set_title("Efecto de la atenuación del canal (ruido de receptor fijo)")
ax.set_xlabel("Tiempo [ms]"); ax.set_ylabel("V"); ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/06_atenuacion.png", dpi=150); plt.close()

# 4.4 Distorsión no lineal (amplificador con compresión)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
sd = distorsion_no_lineal(s, a3=-0.03)
f_, Xd = espectro(sd, fs)
ax[0].plot(t * 1e3, s, lw=0.6, label="Original")
ax[0].plot(t * 1e3, sd, lw=0.6, label="Distorsionada")
ax[0].set_title("Distorsión no lineal (cúbica, a3=-0.03) en el tiempo")
ax[0].set_xlabel("Tiempo [ms]"); ax[0].legend()
ax[1].plot(fs_ / 1e3, 20 * np.log10(Xs + 1e-9), lw=0.8, label="Original")
ax[1].plot(f_ / 1e3, 20 * np.log10(Xd + 1e-9), lw=0.8, label="Distorsionada", alpha=.8)
ax[1].set_xlim(0, 45); ax[1].set_ylim(-100, 10)
ax[1].set_title("Espectro [dB]: aparecen armónicos (3fc, fc±2fm...)")
ax[1].set_xlabel("Frecuencia [kHz]"); ax[1].legend()
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/07_distorsion.png", dpi=150); plt.close()
r = demodular_envolvente(sd, fs)
res_dist = (correlacion(m, r), snr_recuperada_db(m, r))

# 4.5 Demodulación: mensaje original vs recuperado (caso con SNR = 15 dB)
sn = agregar_ruido(s, 15)
r = demodular_envolvente(sn, fs)
g = np.dot(m, r) / np.dot(r, r)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t * 1e3, m, label="Mensaje original", lw=2)
ax.plot(t * 1e3, g * r, "--", label="Recuperado (SNR canal 15 dB)", lw=1.3)
ax.set_title("Demodulación por detector de envolvente")
ax.set_xlabel("Tiempo [ms]"); ax.set_ylabel("V"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/08_demodulacion.png", dpi=150); plt.close()

# Curva completa SNR canal vs SNR recuperada
rango = np.arange(0, 41, 2)
rec = []
for snr in rango:
    vals = [snr_recuperada_db(m, demodular_envolvente(agregar_ruido(s, snr), fs))
            for _ in range(10)]
    rec.append(np.mean(vals))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rango, rec, "o-")
ax.set_xlabel("SNR en el canal [dB]"); ax.set_ylabel("SNR del mensaje recuperado [dB]")
ax.set_title("Desempeño del sistema AM frente al ruido"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/09_curva_snr.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 5. RESUMEN EN CONSOLA
# ----------------------------------------------------------------------------
print("\n--- Índice de modulación ---")
for mu_i, c_, sn_ in resultados_mu:
    print(f"mu={mu_i:.1f}  correlación={c_:.3f}  SNR rec={sn_:.1f} dB")
print("\n--- Ruido AWGN ---")
for snr, c_, sn_ in resultados_ruido:
    print(f"SNR canal={snr:2d} dB  correlación={c_:.3f}  SNR rec={sn_:.1f} dB")
print("\n--- Atenuación ---")
for k, sin, c_, sn_ in resultados_att:
    print(f"x{k:.1f}  SNR entrada={sin:.1f} dB  correlación={c_:.3f}  SNR rec={sn_:.1f} dB")
print(f"\n--- Distorsión no lineal ---\ncorrelación={res_dist[0]:.3f}  SNR rec={res_dist[1]:.1f} dB")
print(f"\nGráficas guardadas en: {OUT}")
