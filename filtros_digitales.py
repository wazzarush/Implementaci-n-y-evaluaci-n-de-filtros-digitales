"""
filtros_digitales.py
Actividad formativa 3: Implementación y evaluación de filtros digitales.

Se diseñan y aplican filtros pasa bajos, pasa altos y pasa bandas (FIR e IIR)
sobre una señal compuesta de varias frecuencias contaminada con ruido blanco.

Estructura de cada experimento:
  1) Definición de la señal de entrada
  2) Diseño del filtro
  3) Aplicación del filtro a la señal
  4) Visualización de los resultados

Uso:  python filtros_digitales.py      (genera las figuras en ./figuras y resultados.json)
"""
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # avisos de retardo de grupo en singularidades
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils_filtros import (espectro_amplitud, amplitud_en, snr_db, rmse,
                           AZUL, NARANJA, VERDE, ROJO, GRIS)

# ----------------------------------------------------------------------
# 1) DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ----------------------------------------------------------------------
fs = 1000                       # frecuencia de muestreo (Hz)  -> Nyquist = 500 Hz
T = 2.0                         # duración (s)
t = np.arange(0, T, 1 / fs)     # vector de tiempo
rng = np.random.default_rng(42) # semilla fija para reproducibilidad

# Tres componentes sinusoidales: baja, media y alta frecuencia
f_baja, f_media, f_alta = 10, 60, 200
x_baja = 1.0 * np.sin(2 * np.pi * f_baja * t)
x_media = 0.8 * np.sin(2 * np.pi * f_media * t)
x_alta = 0.6 * np.sin(2 * np.pi * f_alta * t)
limpia = x_baja + x_media + x_alta

ruido = rng.normal(0, 0.5, size=t.size)   # ruido blanco gaussiano (sigma = 0.5)
x = limpia + ruido                         # señal de entrada con ruido

resultados = {"fs": fs, "ruido_sigma": 0.5, "snr_entrada_db": round(snr_db(limpia, x), 2)}


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"figuras/{nombre}.png")
    plt.close(fig)


# --- Figura 01: la señal de entrada (tiempo y frecuencia) ---
fig, ax = plt.subplots(3, 1, figsize=(9, 8))
ax[0].plot(t[:400], limpia[:400], color=AZUL); ax[0].set_title("Señal limpia (10 + 60 + 200 Hz)")
ax[1].plot(t[:400], x[:400], color=GRIS); ax[1].set_title("Señal con ruido blanco (entrada al filtro)")
for a in ax[:2]:
    a.set_xlabel("Tiempo (s)"); a.set_ylabel("Amplitud")
f, A = espectro_amplitud(x, fs)
ax[2].plot(f, A, color=NARANJA); ax[2].set_xlim(0, 500)
ax[2].set_title("Espectro de amplitud de la señal con ruido"); ax[2].set_xlabel("Frecuencia (Hz)"); ax[2].set_ylabel("Amplitud")
for fr in (f_baja, f_media, f_alta):
    ax[2].annotate(f"{fr} Hz", (fr, amplitud_en(x, fs, fr)), textcoords="offset points", xytext=(5, 6), fontsize=9)
guardar(fig, "01_senal_entrada")


# ----------------------------------------------------------------------
# Función genérica: evalúa y grafica un filtro (respuesta en frecuencia,
# tiempo y espectro, antes y después)
# ----------------------------------------------------------------------
def evaluar(nombre, archivo, w, h_resp, y, objetivo, cortes, titulo_filtro):
    """w, h_resp: respuesta en frecuencia (Hz, complejo). y: señal filtrada.
    objetivo: componente que debería conservarse. cortes: frecuencias de corte a marcar."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    # (a) Magnitud de la respuesta en frecuencia
    ax[0, 0].plot(w, 20 * np.log10(np.maximum(np.abs(h_resp), 1e-6)), color=AZUL)
    ax[0, 0].set_ylim(-100, 5); ax[0, 0].set_xlim(0, fs / 2)
    for c in cortes:
        ax[0, 0].axvline(c, color=ROJO, ls="--", lw=1)
    ax[0, 0].axhline(-3, color=GRIS, ls=":", lw=1)
    ax[0, 0].set_title(f"Respuesta en frecuencia: {titulo_filtro}")
    ax[0, 0].set_xlabel("Frecuencia (Hz)"); ax[0, 0].set_ylabel("Magnitud (dB)")
    # (b) Fase
    ax[0, 1].plot(w, np.unwrap(np.angle(h_resp)), color=VERDE)
    ax[0, 1].set_xlim(0, fs / 2); ax[0, 1].set_title("Respuesta de fase")
    ax[0, 1].set_xlabel("Frecuencia (Hz)"); ax[0, 1].set_ylabel("Fase (rad)")
    # (c) Dominio del tiempo
    ax[1, 0].plot(t[:400], x[:400], color=GRIS, alpha=0.6, label="Entrada con ruido")
    ax[1, 0].plot(t[:400], y[:400], color=NARANJA, label="Salida filtrada")
    ax[1, 0].plot(t[:400], objetivo[:400], color=AZUL, ls="--", lw=1, label="Componente objetivo")
    ax[1, 0].set_title("Dominio del tiempo"); ax[1, 0].set_xlabel("Tiempo (s)"); ax[1, 0].set_ylabel("Amplitud")
    ax[1, 0].legend(loc="upper right", fontsize=8)
    # (d) Dominio de la frecuencia
    f1, A1 = espectro_amplitud(x, fs); f2, A2 = espectro_amplitud(y, fs)
    ax[1, 1].plot(f1, A1, color=GRIS, alpha=0.7, label="Antes")
    ax[1, 1].plot(f2, A2, color=NARANJA, label="Después")
    ax[1, 1].set_xlim(0, fs / 2); ax[1, 1].set_title("Espectro de amplitud")
    ax[1, 1].set_xlabel("Frecuencia (Hz)"); ax[1, 1].set_ylabel("Amplitud"); ax[1, 1].legend(fontsize=8)
    fig.suptitle(nombre, fontweight="bold")
    guardar(fig, archivo)


def metricas(y, objetivo):
    """Métricas de efectividad del filtro respecto del componente que se quiere conservar."""
    return {
        "snr_salida_db": round(snr_db(objetivo, y), 2),
        "rmse": round(rmse(objetivo, y), 3),
        "amp_10Hz": round(amplitud_en(y, fs, 10), 3),
        "amp_60Hz": round(amplitud_en(y, fs, 60), 3),
        "amp_200Hz": round(amplitud_en(y, fs, 200), 3),
    }


resultados["amplitudes_entrada"] = {"10Hz": round(amplitud_en(x, fs, 10), 3),
                                    "60Hz": round(amplitud_en(x, fs, 60), 3),
                                    "200Hz": round(amplitud_en(x, fs, 200), 3)}

# ----------------------------------------------------------------------
# 2) EXPERIMENTO A: PASA BAJOS IIR - Butterworth orden 6, fc = 30 Hz
#    Objetivo: conservar los 10 Hz y eliminar 60 Hz, 200 Hz y el ruido.
# ----------------------------------------------------------------------
sos_lp = signal.butter(6, 30, btype="low", fs=fs, output="sos")   # diseño (secciones de 2.º orden)
y_lp = signal.sosfiltfilt(sos_lp, x)                               # filtrado de fase cero
w, h = signal.sosfreqz(sos_lp, worN=4096, fs=fs)
evaluar("Experimento A: pasa bajos Butterworth (orden 6, fc = 30 Hz)", "02_pasa_bajos_butterworth",
        w, h, y_lp, x_baja, [30], "Butterworth orden 6")
resultados["A_pasa_bajos"] = metricas(y_lp, x_baja)

# ----------------------------------------------------------------------
# 3) EXPERIMENTO B: PASA ALTOS IIR - Chebyshev tipo I, orden 6, rizado 1 dB, fc = 150 Hz
#    Objetivo: conservar los 200 Hz y eliminar 10 Hz y 60 Hz.
# ----------------------------------------------------------------------
sos_hp = signal.cheby1(6, 1, 150, btype="high", fs=fs, output="sos")
y_hp = signal.sosfiltfilt(sos_hp, x)
w, h = signal.sosfreqz(sos_hp, worN=4096, fs=fs)
evaluar("Experimento B: pasa altos Chebyshev I (orden 6, rizado 1 dB, fc = 150 Hz)", "03_pasa_altos_chebyshev",
        w, h, y_hp, x_alta, [150], "Chebyshev I orden 6")
resultados["B_pasa_altos"] = metricas(y_hp, x_alta)

# ----------------------------------------------------------------------
# 4) EXPERIMENTO C: PASA BANDAS FIR con ventana de Hamming, 101 coeficientes, 40-80 Hz
#    Objetivo: conservar los 60 Hz y eliminar 10 Hz, 200 Hz y el ruido fuera de banda.
# ----------------------------------------------------------------------
fir_bp = signal.firwin(101, [40, 80], pass_zero=False, window="hamming", fs=fs)
y_bp = signal.filtfilt(fir_bp, [1.0], x)        # FIR: denominador = 1
w, h = signal.freqz(fir_bp, worN=4096, fs=fs)
evaluar("Experimento C: pasa bandas FIR con ventana Hamming (101 coef., 40-80 Hz)", "04_pasa_bandas_fir",
        w, h, y_bp, x_media, [40, 80], "FIR Hamming 101 coeficientes")
resultados["C_pasa_bandas"] = metricas(y_bp, x_media)

# ----------------------------------------------------------------------
# 5) COMPARACIÓN DE FAMILIAS (todas pasa bajos con fc = 30 Hz):
#    Butterworth, Chebyshev I, Chebyshev II y FIR-Hamming
# ----------------------------------------------------------------------
disenos = {
    "Butterworth (n=4)":        ("iir", signal.butter(4, 30, "low", fs=fs, output="sos")),
    "Chebyshev I (n=4, 1 dB)":  ("iir", signal.cheby1(4, 1, 30, "low", fs=fs, output="sos")),
    "Chebyshev II (n=4, 40 dB)": ("iir", signal.cheby2(4, 40, 40, "low", fs=fs, output="sos")),
    "FIR Hamming (101 coef.)":  ("fir", signal.firwin(101, 30, window="hamming", fs=fs)),
}
colores = [AZUL, NARANJA, VERDE, ROJO]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
comp = {}
for (nombre, (tipo, d)), col in zip(disenos.items(), colores):
    if tipo == "iir":
        w, h = signal.sosfreqz(d, worN=4096, fs=fs)
        b, a = signal.sos2tf(d)
        wg, gd = signal.group_delay((b, a), w=4096, fs=fs)
        y_ = signal.sosfilt(d, x)                # filtrado causal para medir el retardo real
        orden = 2 * d.shape[0]
    else:
        w, h = signal.freqz(d, worN=4096, fs=fs)
        wg, gd = signal.group_delay((d, [1.0]), w=4096, fs=fs)
        y_ = signal.lfilter(d, [1.0], x)
        orden = len(d) - 1
    ax[0].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=col, label=nombre)
    ax[1].plot(w, np.unwrap(np.angle(h)), color=col)
    ax[2].plot(wg, gd, color=col)
    idx = lambda f0: np.argmin(np.abs(w - f0))
    comp[nombre] = {"orden": int(orden),
                    "atenuacion_60Hz_db": round(float(20 * np.log10(abs(h[idx(60)]))), 1),
                    "atenuacion_200Hz_db": round(float(20 * np.log10(abs(h[idx(200)]))), 1),
                    "retardo_grupo_10Hz_muestras": round(float(gd[idx(10)]), 1)}
ax[0].set_ylim(-100, 5); ax[0].set_xlim(0, 250); ax[0].axvline(30, color=GRIS, ls="--", lw=1)
ax[0].set_title("Magnitud (dB)"); ax[0].legend(fontsize=8)
ax[1].set_xlim(0, 250); ax[1].set_title("Fase (rad)")
ax[2].set_xlim(0, 100); ax[2].set_ylim(0, 60); ax[2].set_title("Retardo de grupo (muestras)")
for a_ in ax:
    a_.set_xlabel("Frecuencia (Hz)")
fig.suptitle("Experimento D: comparación de familias de filtros pasa bajos (fc = 30 Hz)", fontweight="bold")
guardar(fig, "05_comparacion_familias")
resultados["D_comparacion_familias"] = comp

# ----------------------------------------------------------------------
# 6) EFECTO DEL ORDEN: Butterworth pasa bajos fc = 30 Hz con n = 2, 4, 8
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
orden_res = {}
for n, col in zip((2, 4, 8), (AZUL, NARANJA, VERDE)):
    s = signal.butter(n, 30, "low", fs=fs, output="sos")
    w, h = signal.sosfreqz(s, worN=4096, fs=fs)
    ax[0].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=col, label=f"n = {n}")
    yy = signal.sosfiltfilt(s, x)
    orden_res[f"n={n}"] = {"snr_salida_db": round(snr_db(x_baja, yy), 2),
                           "atenuacion_60Hz_db": round(float(20 * np.log10(abs(h[np.argmin(np.abs(w - 60))]))), 1)}
    ax[1].plot(t[:300], yy[:300], color=col, label=f"n = {n}")
ax[0].set_ylim(-120, 5); ax[0].set_xlim(0, 200); ax[0].axvline(30, color=GRIS, ls="--", lw=1)
ax[0].set_title("Magnitud según el orden"); ax[0].set_xlabel("Frecuencia (Hz)"); ax[0].set_ylabel("dB"); ax[0].legend()
ax[1].plot(t[:300], x_baja[:300], color=GRIS, ls="--", label="Objetivo (10 Hz)")
ax[1].set_title("Señal filtrada"); ax[1].set_xlabel("Tiempo (s)"); ax[1].legend(fontsize=8)
guardar(fig, "06_efecto_del_orden")
resultados["E_efecto_orden"] = orden_res

# ----------------------------------------------------------------------
# 7) FILTRADO CAUSAL (lfilter) vs FASE CERO (filtfilt): retardo y distorsión de fase
# ----------------------------------------------------------------------
s = signal.butter(6, 30, "low", fs=fs, output="sos")
y_causal = signal.sosfilt(s, x)
y_cero = signal.sosfiltfilt(s, x)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t[:300], x_baja[:300], color=GRIS, ls="--", label="Objetivo (10 Hz)")
ax.plot(t[:300], y_causal[:300], color=ROJO, label="Causal (sosfilt): con retardo")
ax.plot(t[:300], y_cero[:300], color=VERDE, label="Fase cero (sosfiltfilt): sin retardo")
ax.set_title("Filtrado causal frente a fase cero (Butterworth n = 6, fc = 30 Hz)")
ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud"); ax.legend(fontsize=8)
guardar(fig, "07_causal_vs_fase_cero")
# retardo estimado por correlación cruzada
lag = np.argmax(np.correlate(y_causal[200:1800], x_baja[200:1800], "full")) - (1600 - 1)
resultados["F_retardo_causal_muestras"] = int(lag)
resultados["F_retardo_causal_ms"] = round(1000 * lag / fs, 1)

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)
print(json.dumps(resultados, indent=2, ensure_ascii=False))
