"""
aplicacion_ecg.py
Aplicación práctica: limpieza de un electrocardiograma (ECG) contaminado.

Un ECG real suele tener tres perturbaciones típicas:
  - Deriva de línea base (respiración/movimiento): muy baja frecuencia (< 0.5 Hz)
  - Interferencia de la red eléctrica: 60 Hz en México (50 Hz en Europa)
  - Ruido muscular / electrónico: banda ancha, de alta frecuencia

Como no se usan datos de pacientes, el ECG se SIMULA con una suma de ondas gaussianas
(P, Q, R, S, T) por latido. Esto permite conocer la señal limpia y medir el error.

Cadena de filtrado:
  1) Pasa altos Butterworth 0.5 Hz (orden 4) -> elimina la deriva de línea base
  2) Notch (rechaza banda) 60 Hz     -> elimina la interferencia de la red
  3) Pasa bajos Butterworth 100 Hz   -> reduce el ruido muscular de alta frecuencia

Uso:  python aplicacion_ecg.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils_filtros import (espectro_amplitud, snr_db, rmse, amplitud_en,
                           AZUL, NARANJA, VERDE, ROJO, GRIS)

# ---------------- 1) Definición de la señal de entrada ----------------
fs = 500                         # Hz (valor típico en electrocardiógrafos)
T = 10                           # segundos
t = np.arange(0, T, 1 / fs)
fc_cardiaca = 72 / 60            # 72 latidos por minuto -> 1.2 Hz
rng = np.random.default_rng(7)

def latido(tt):
    """Un latido: suma de 5 gaussianas (amplitud mV, centro s, ancho s)."""
    ondas = [(0.15, -0.20, 0.025), (-0.12, -0.04, 0.010), (1.20, 0.0, 0.012),
             (-0.25, 0.04, 0.010), (0.30, 0.25, 0.045)]       # P, Q, R, S, T
    return sum(a * np.exp(-0.5 * ((tt - c) / w) ** 2) for a, c, w in ondas)

ecg_limpio = np.zeros_like(t)
for k in np.arange(0.5, T, 1 / fc_cardiaca):                  # posición de cada onda R
    ecg_limpio += latido(t - k)

deriva = 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.2 * np.sin(2 * np.pi * 0.1 * t)   # línea base
red = 0.25 * np.sin(2 * np.pi * 60 * t)                                          # 60 Hz
emg = rng.normal(0, 0.06, t.size)                                                # ruido muscular
ecg_ruidoso = ecg_limpio + deriva + red + emg

# ---------------- 2) Diseño de los filtros ----------------
sos_hp = signal.butter(4, 0.5, btype="high", fs=fs, output="sos")     # deriva
b_notch, a_notch = signal.iirnotch(60, Q=30, fs=fs)                   # red eléctrica
sos_notch = signal.tf2sos(b_notch, a_notch)
sos_lp = signal.butter(4, 100, btype="low", fs=fs, output="sos")      # ruido muscular
sos_total = np.vstack([sos_hp, sos_notch, sos_lp])                    # cadena en cascada

# ---------------- 3) Aplicación (fase cero para no deformar el QRS) ----------------
paso1 = signal.sosfiltfilt(sos_hp, ecg_ruidoso)
paso2 = signal.sosfiltfilt(sos_notch, paso1)
ecg_filtrado = signal.sosfiltfilt(sos_lp, paso2)

# Métricas (se omiten 1 s al inicio y al final para evitar efectos de borde)
m = slice(fs, -fs)
res = {
    "rmse_antes_mV": round(rmse(ecg_limpio[m], ecg_ruidoso[m]), 3),
    "rmse_despues_mV": round(rmse(ecg_limpio[m], ecg_filtrado[m]), 3),
    "snr_antes_db": round(snr_db(ecg_limpio[m], ecg_ruidoso[m]), 2),
    "snr_despues_db": round(snr_db(ecg_limpio[m], ecg_filtrado[m]), 2),
    "amp_60Hz_antes": round(amplitud_en(ecg_ruidoso, fs, 60), 3),
    "amp_60Hz_despues": round(amplitud_en(ecg_filtrado, fs, 60), 4),
    "amp_R_limpio_mV": round(float(ecg_limpio[m].max()), 3),
    "amp_R_filtrado_mV": round(float(ecg_filtrado[m].max()), 3),
}

# ---------------- 4) Visualización ----------------
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ver = (t >= 2) & (t <= 6)
ax[0].plot(t[ver], ecg_ruidoso[ver], color=GRIS); ax[0].set_title("ECG contaminado (deriva + 60 Hz + ruido muscular)")
ax[1].plot(t[ver], ecg_filtrado[ver], color=NARANJA, label="ECG filtrado")
ax[1].plot(t[ver], ecg_limpio[ver], color=AZUL, ls="--", lw=1, label="ECG limpio (referencia)")
ax[1].set_title("ECG después de la cadena de filtrado"); ax[1].legend(loc="upper right", fontsize=8)
ax[2].plot(t[ver], ecg_ruidoso[ver] - ecg_limpio[ver], color=ROJO, lw=0.8, label="Ruido antes")
ax[2].plot(t[ver], ecg_filtrado[ver] - ecg_limpio[ver], color=VERDE, lw=0.8, label="Error residual")
ax[2].set_title("Error respecto de la señal limpia"); ax[2].legend(loc="upper right", fontsize=8)
for a in ax:
    a.set_ylabel("Amplitud (mV)")
ax[2].set_xlabel("Tiempo (s)")
fig.tight_layout(); fig.savefig("figuras/08_ecg_tiempo.png"); plt.close(fig)

# Espectro antes/después y respuesta en frecuencia de la cadena
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
f1, A1 = espectro_amplitud(ecg_ruidoso, fs); f2, A2 = espectro_amplitud(ecg_filtrado, fs)
ax[0].semilogy(f1, A1 + 1e-6, color=GRIS, label="Antes"); ax[0].semilogy(f2, A2 + 1e-6, color=NARANJA, label="Después")
ax[0].set_xlim(0, 250); ax[0].set_ylim(1e-4, 1); ax[0].axvline(60, color=ROJO, ls="--", lw=1)
ax[0].set_title("Espectro de amplitud (escala logarítmica)"); ax[0].set_xlabel("Frecuencia (Hz)"); ax[0].set_ylabel("Amplitud (mV)"); ax[0].legend()
w, h = signal.sosfreqz(sos_total, worN=8192, fs=fs)
ax[1].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=AZUL)
ax[1].set_xscale("log"); ax[1].set_xlim(0.05, 250); ax[1].set_ylim(-80, 5)
ax[1].set_title("Respuesta en frecuencia de la cadena completa"); ax[1].set_xlabel("Frecuencia (Hz, escala log)"); ax[1].set_ylabel("Magnitud (dB)")
for fr in (0.5, 60, 100):
    ax[1].axvline(fr, color=ROJO, ls="--", lw=1)
fig.tight_layout(); fig.savefig("figuras/09_ecg_frecuencia.png"); plt.close(fig)

with open("resultados_ecg.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
