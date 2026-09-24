# CC3092 – Laboratorio 7: ETTh1 Forecasting

**Dataset:** ETTh1 (Electricity Transformer Temperature – hourly, ~2 years)  
**Variable objetivo:** `OT` (Oil Temperature)  
**Horizontes de predicción:** h = 24 h y h = 48 h  
**Arquitecturas:** LSTM many-to-one y Transformer encoder con self-attention temporal (ambas con PyTorch puro)

### Estructura del repositorio

```
├── Lab7_ETTh1.ipynb    # notebook principal (Tasks 1–4)
├── README.md           # respuestas a las preguntas de cada task
├── requirements.txt
├── data/               # ETTh1.csv, descargado por el notebook (ignorado en git)
└── outputs/            # figuras generadas por el notebook
```

Para reproducir: `pip install -r requirements.txt` y ejecutar el notebook completo.

---

## Task 1

### 1.2.a) Segundo pico más alto de la ACF e interpretación de periodicidad

Calculada la ACF sobre `OT_train` (10 452 muestras horarias) con `max_lag = 96`, los resultados obtenidos son:

| Lag (h) | ρ(k)   | Interpretación              |
|--------:|-------:|:----------------------------|
| 0       | 1.0000 | **primer pico** (autocorrelación perfecta) |
| 22      | 0.9262 | **segundo pico más alto** (máximo local)   |
| 24      | 0.9250 | un día completo             |
| 48      | 0.8763 | dos días                    |
| 96      | 0.8489 | cuatro días (= L)           |

El **primer pico** de la ACF es siempre el lag 0 (ρ = 1.0). El **segundo pico más alto** de la ACF (el primer máximo local después del lag 0) se produce en el **lag 22** con ρ = 0.9262.
Este máximo local apunta a una periodicidad de aproximadamente **22–24 horas**, consistente con
el ciclo diario de carga de la red eléctrica: la demanda sube durante el día y cae de noche,
modulando la temperatura del aceite de forma casi periódica cada 24 horas.
El leve desfase de 2 h respecto al lag 24 se debe a que los picos y valles del ciclo diario
no coinciden exactamente con la hora exacta de muestreo.

La ACF revela que la temperatura del aceite exhibe **persistencia de corto plazo
muy fuerte** (alta correlación para lags pequeños) y un **patrón de repetición diario** de ≈ 24 h,
propio de la operación circadiana de los transformadores.

---

### 1.2.b) Justificación de L = 96 como longitud de ventana

Con los valores reales de la ACF obtenidos:

- ρ(24) = **0.9250** — la correlación con 1 día atrás sigue siendo muy alta.
- ρ(48) = **0.8763** — la correlación con 2 días atrás sigue siendo sustancial.
- ρ(96) = **0.8489** — la correlación con 4 días atrás (= L) todavía supera 0.84.

**L = 96 es una elección apropiada** por las siguientes razones:

1. **Captura el ciclo diario completo más tres repeticiones:** puesto que la periodicidad
   dominante es de ≈ 24 h, una ventana de 96 h (4 días) incluye cuatro ciclos completos,
   lo que le da al modelo información suficiente para identificar el patrón periódico.

2. **La ACF se mantiene alta hasta lag 96 (ρ = 0.849):** esto indica que el pasado a
   4 días de distancia sigue siendo informativo para predecir el valor actual. Reducir
   L a, por ejemplo, 24 h (un solo ciclo) privaría al modelo de correlaciones
   aún significativas en lags 48–96.

3. **Hay ganancia marginal decreciente más allá de 96:** a partir de lag 96 la ACF
   continúa descendiendo (valores < 0.85). Ampliar L a 168 h (1 semana) podría
   aportar algo, pero incrementaría el costo computacional y la dimensión de entrada sin
   una ganancia proporcional en señal predictiva.

**Conclusión:** L = 96 es un equilibrio razonable entre capacidad de modelado
y eficiencia computacional, respaldado empíricamente porque ρ(96) = 0.849 confirma
que aún hay dependencia estadística relevante en ese rango y que la señal de la
periodicidad diaria (≈ 4 ciclos) queda completamente cubierta.

---

## Task 2

### 2.1) `LSTMForecaster`

Celda LSTM implementada paso a paso con `nn.Parameter` (`Wx (4d_h, 1)`, `Wh (4d_h, d_h)`, `b (4d_h,)`,
`W_out (1, d_h)`, `b_out (1,)`), sin `nn.LSTM`. En cada paso se calculan las 4 compuertas
`[i, f, g, o] = x_t Wxᵀ + h_{t-1} Whᵀ + b`, y luego `c_t = f⊙c_{t-1} + i⊙g` y `h_t = o⊙tanh(c_t)`.
Solo `h_L` se proyecta a la predicción (many-to-one).

**Normalización relativa al último valor.** La ventana entra al LSTM como `x̃ = x − x_t` y el modelo
predice el cambio: `ŷ = x_t + Δ̂`. Esto hizo falta porque la serie cambia de nivel entre splits
(media de `y` en train ≈ +0.45 y en validación ≈ −0.76, en escala normalizada). Con la entrada en
valores absolutos, el LSTM le ganaba al naive en train (MAE 0.272 vs 0.282) pero perdía en
validación (MAE 0.298 vs 0.200), porque sus predicciones se iban hacia el nivel de train.
Con la entrada relativa el modelo ya no depende del nivel y sí supera al naive en validación.
La idea viene de NLinear (Zeng et al., 2023) y RevIN (Kim et al., 2022). El Transformer
(Task 3.2) usa la misma normalización para que la comparación del Task 4 sea justa.

### 2.2) Entrenamiento y resultados

Configuración: `d_h = 32`, Adam (`lr = 1e-3`), 20 épocas, `batch_size = 64`, semilla 42.
Se entrenó un modelo para cada horizonte.

**Función de pérdida: MSE.**
- La serie está normalizada (z-score) y el error de pronóstico es aproximadamente gaussiano,
  así que minimizar el MSE equivale a la estimación de máxima verosimilitud.
- El MSE penaliza los errores grandes de forma cuadrática. Para un transformador, fallar un pico de
  temperatura (riesgo de falla térmica) es más grave que varios errores pequeños, y el MAE los
  pesaría a todos igual.
- Optimiza directamente el RMSE que se reporta como métrica.

**Métricas en validación** (escala normalizada; el error en °C es el error normalizado × σ = 8.567):

| Modelo | Horizonte | MAE | RMSE | MAE (°C) | RMSE (°C) | Ratio vs naive |
|---|---|---|---|---|---|---|
| LSTM  | 24h | 0.1918 | 0.2592 | 1.643 | 2.221 | **0.961** |
| LSTM  | 48h | 0.2547 | 0.3243 | 2.182 | 2.778 | **0.978** |
| Naive | 24h | 0.1996 | 0.2672 | 1.710 | 2.289 | 1.000 |
| Naive | 48h | 0.2604 | 0.3359 | 2.231 | 2.878 | 1.000 |

En ambos horizontes el LSTM supera al baseline de persistencia (ratio < 1). La ventaja es modesta
porque la persistencia ya es un predictor muy fuerte: la ACF indica ρ(24) = 0.925.
El error crece de H = 24 a H = 48 tanto para el LSTM como para el naive.

**Curvas de pérdida de entrenamiento:**

![Curvas de pérdida LSTM](outputs/lstm_loss_curves.png)

Con H = 24 la pérdida baja de forma casi monótona (0.148 → 0.137). Con H = 48 la pérdida se
estanca alrededor de 0.22 y oscila entre épocas. Predecir a 48 h tiene más incertidumbre
irreducible, y con `lr = 1e-3` fijo el optimizador rebota alrededor del mínimo.

---

## Task 3

### 3.1) Positional encoding y proyección de entrada

`make_pe(max_len, d_model)` implementa `PE(t, 2i) = sin(t / 10000^{2i/d_model})` y
`PE(t, 2i+1) = cos(·)` de forma vectorizada. Retorna un tensor `(L, d_model)` que se registra con
`register_buffer`, así que no es aprendible. Cada `x_t` escalar se proyecta con
`nn.Linear(1, d_model)` y luego se le suma el PE.

### 3.2) `TransformerForecaster`

Encoder de una capa implementado sin `nn.MultiheadAttention`:
- Multi-head self-attention manual: `Q, K, V = X W_{Q,K,V}`, luego reshape a `(B, h, L, d_k)`,
  `softmax(QKᵀ/√d_k)`, `@ V`, concatenar las cabezas y multiplicar por `W_O`.
- LayerNorm manual con `γ, β` aprendibles.
- FFN `ReLU(xW₁ + b₁)W₂ + b₂`.
- Dos bloques Add & Norm (post-norm). La predicción sale de la posición 0 usada como CLS.

La celda de verificación del enunciado pasa: `attn_w` tiene forma `(4, 2, 96, 96)` y las filas
suman 1.

### 3.3)

> *Pendiente (otro integrante).* Se reutilizan `train_model`, `predict` y `metrics` del Task 2.2.

---

## Uso de IA generativa

Siguiendo la instrucción del laboratorio, se documenta el prompt usado para los Tasks 2 y 3.1–3.2
(asistente: Claude Code):

> *"necesito que me ayudes a dividir las tareas, parece que José Ruíz ya hizo la primera parte en el
> notebook, déjame la parte a mí José Auyón hasta la 3.2. ¿Me ayudas implementando? hagamos el plan
> primero"*

**Por qué funcionó:** el asistente tenía como contexto el enunciado completo (`Instrucciones.md`) y
el notebook con el Task 1 ya hecho. Así pudo reutilizar las variables existentes (`X_tr_24`,
`naive_mae`, `sigma`, …) y respetar las restricciones del enunciado (sin `nn.LSTM` ni
`nn.MultiheadAttention`). Pedir primero un plan permitió revisar la división de tareas y el enfoque
antes de escribir código. Al ejecutar el notebook, la verificación del Task 2 falló
(LSTM MAE 0.298 > naive 0.200). Se diagnosticó el cambio de nivel entre train y validación y se
corrigió con la normalización relativa descrita en 2.1.

---

## Task 4

> *Pendiente de implementación.*
