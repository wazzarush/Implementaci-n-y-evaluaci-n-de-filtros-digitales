# Implementación y evaluación de filtros digitales (Python)

Actividad formativa 3 · Diseño de filtros digitales pasa bajos, pasa altos y pasa bandas con **SciPy** y **Matplotlib**, y análisis de su efecto en el dominio del tiempo y de la frecuencia.

> Autor: Abigail Ramos · Materia: Señales y Sistemas · Fecha: 15 de Noviembre del 2026

## Propósito

1. Generar una **señal compuesta** (10 Hz + 60 Hz + 200 Hz) contaminada con **ruido blanco gaussiano**.
2. Diseñar filtros **IIR** (Butterworth, Chebyshev I y II) y **FIR** (ventana de Hamming).
3. Aplicarlos y comparar las señales **antes y después**, en tiempo y en frecuencia.
4. Evaluar la efectividad con métricas objetivas (SNR, RMSE, atenuación en dB).
5. Aplicar lo aprendido a un caso práctico: **limpieza de un electrocardiograma (ECG)** simulado.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `filtros_digitales.py` | Experimentos A–F: pasa bajos, pasa altos, pasa bandas, comparación de familias, efecto del orden y filtrado causal vs. fase cero |
| `aplicacion_ecg.py` | Aplicación práctica: ECG con deriva de línea base, interferencia de 60 Hz y ruido muscular |
| `utils_filtros.py` | Funciones auxiliares (espectro, SNR, RMSE, estilo de gráficas) |
| `figuras/` | Gráficas generadas por los scripts |
| `resultados.json`, `resultados_ecg.json` | Métricas numéricas de cada ejecución |
| `requirements.txt` | Dependencias |

## Cómo ejecutarlo

```bash
git clone https://github.com/<tu-usuario>/filtros-digitales.git
cd filtros-digitales
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
python filtros_digitales.py
python aplicacion_ecg.py
```

Las gráficas se guardan en `figuras/`. Se usa una semilla aleatoria fija, así que los resultados son reproducibles.

## Filtros diseñados

| Exp. | Tipo | Método | Parámetros | Objetivo |
|---|---|---|---|---|
| A | Pasa bajos | IIR Butterworth | orden 6, fc = 30 Hz | Conservar 10 Hz |
| B | Pasa altos | IIR Chebyshev I | orden 6, rizado 1 dB, fc = 150 Hz | Conservar 200 Hz |
| C | Pasa bandas | FIR ventana Hamming | 101 coeficientes, 40–80 Hz | Conservar 60 Hz |
| D | Comparación | Butterworth, Cheby I, Cheby II, FIR | fc = 30 Hz | Magnitud, fase y retardo |
| E | Orden | Butterworth | n = 2, 4, 8 | Pendiente de atenuación |
| F | Causal vs. fase cero | `sosfilt` vs. `sosfiltfilt` | Butterworth n = 6 | Retardo y distorsión de fase |

## Resultados principales (fs = 1000 Hz, ruido σ = 0.5)

| Filtro | SNR de entrada | SNR de salida | Atenuación de las componentes no deseadas |
|---|---|---|---|
| Pasa bajos (A) | 5.99 dB | **15.28 dB** | 60 Hz y 200 Hz eliminadas por completo |
| Pasa altos (B) | 5.99 dB | 1.21 dB | 10 Hz y 60 Hz eliminadas; el ruido por encima de 150 Hz permanece |
| Pasa bandas (C) | 5.99 dB | 13.37 dB | 10 Hz y 200 Hz eliminadas |

Aplicación ECG: el error RMSE baja de **0.408 mV a 0.091 mV** y la interferencia de 60 Hz pasa de 0.252 mV a 0.003 mV. Los detalles y su interpretación están en el informe.

## Notas

- Los filtros IIR se implementan en **secciones de segundo orden (SOS)** para garantizar estabilidad numérica.
- Para comparar sin retardo se usa filtrado de **fase cero** (`filtfilt`), que aplica el filtro dos veces (adelante y atrás). Solo es posible fuera de línea; en tiempo real se usa filtrado causal.
- El ECG es **simulado** (ondas gaussianas P, Q, R, S, T). No contiene datos de pacientes.

## Referencias

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-time signal processing* (3.ª ed.). Pearson.
- Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2
