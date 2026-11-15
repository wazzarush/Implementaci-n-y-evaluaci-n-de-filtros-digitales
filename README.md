# Implementación y análisis de un sistema de modulación en amplitud (AM)

Actividad formativa 4 – Sistemas de comunicación.
Simulación en **Python** (NumPy, SciPy, Matplotlib) de un sistema AM con portadora (DSB-LC)
y análisis de su desempeño con **ruido, atenuación, distorsión y sobremodulación**.

## ¿Qué hace el código?
1. Define un mensaje de baja frecuencia (500 Hz) y una portadora de alta frecuencia (10 kHz).
2. Modula en amplitud: `s(t) = Ac [1 + μ·m(t)] cos(2π fc t)`.
3. Grafica mensaje, portadora y señal modulada (tiempo y frecuencia).
4. Evalúa cuatro escenarios: índice de modulación (μ = 0.5, 1.0, 1.5), ruido AWGN (30/15/5 dB),
   atenuación del canal (×1, ×0.5, ×0.1) y distorsión no lineal.
5. Demodula con detector de envolvente y calcula métricas (correlación y SNR recuperada).

## Cómo ejecutarlo
```bash
git clone https://github.com/<tu-usuario>/am-modulacion.git
cd am-modulacion
pip install -r requirements.txt
python modulacion_am.py
```
Las gráficas se guardan en `figuras/`.

## Parámetros principales
| Parámetro | Valor |
|---|---|
| Frecuencia de muestreo | 100 kHz |
| Mensaje (fm, Am) | 500 Hz, 1 V |
| Portadora (fc, Ac) | 10 kHz, 2 V |
| Índice de modulación base | μ = 0.5 |

## Resultados resumidos
| Escenario | Correlación mensaje–recuperado | SNR recuperada |
|---|---|---|
| μ = 0.5 / 1.0 (sin ruido) | 1.000 | ≈ 50 dB |
| μ = 1.5 (sobremodulación) | 0.967 | ≈ 12 dB |
| Ruido SNR 30 / 15 / 5 dB | 1.000 / 0.996 / 0.957 | 34 / 20 / 11 dB |
| Atenuación ×0.5 / ×0.1 (ruido fijo) | 0.992 / 0.825 | 18 / 5 dB |
| Distorsión cúbica | 0.995 | ≈ 20 dB |

## Estructura
```
am-modulacion/
├── modulacion_am.py   # código comentado
├── requirements.txt
├── README.md
└── figuras/           # gráficas generadas
```

## Aplicación real
Radiodifusión AM comercial (530–1700 kHz) y comunicaciones aeronáuticas VHF (118–137 MHz),
donde el detector de envolvente simple permite receptores de bajo costo.
