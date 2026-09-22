# CC3092 – Laboratorio 7: ETTh1 Forecasting

**Dataset:** ETTh1 (Electricity Transformer Temperature – hourly, ~2 years)  
**Variable objetivo:** `OT` (Oil Temperature)  
**Horizontes de predicción:** h = 24 h y h = 48 h  
**Arquitecturas:** LSTM many-to-one y Transformer encoder con self-attention temporal (ambas con PyTorch puro)

---

## Task 1

### 1.2.a) Segundo pico más alto de la ACF e interpretación de periodicidad

Calculada la ACF sobre `OT_train` (10 452 muestras horarias) con `max_lag = 96`, los resultados obtenidos son:

| Lag (h) | ρ(k)   | Interpretación              |
|--------:|-------:|:----------------------------|
| 0       | 1.0000 | autocorrelación perfecta    |
| 1       | 0.9923 | **mayor pico** (persistencia inmediata) |
| 22      | 0.9262 | **segundo pico más alto**   |
| 24      | 0.9250 | un día completo             |
| 48      | 0.8769 | dos días                    |
| 96      | 0.8489 | cuatro días (= L)           |

El **primer pico significativo tras lag 0** se ubica en **lag 1** (ρ = 0.9923), lo cual refleja
la fuerte dependencia temporal inmediata: la temperatura del aceite de un transformador cambia
lentamente, por lo que el valor de la siguiente hora es casi idéntico al actual.

El **segundo pico más alto** (excluyendo lag 0) se produce en **lag 22** con ρ = 0.9262.
Este máximo local apunta a una periodicidad de aproximadamente **22–24 horas**, consistente con
el ciclo diario de carga de la red eléctrica: la demanda sube durante el día y cae de noche,
modulando la temperatura del aceite de forma casi periódica cada 24 horas.
El leve desfase de 2 h respecto al lag 24 se debe a que los picos y valles del ciclo diario
no coinciden exactamente con la hora exacta de muestreo.

En resumen, la ACF revela que la temperatura del aceite exhibe **persistencia de corto plazo
muy fuerte** (alta correlación para lags pequeños) y un **patrón de repetición diario** de ≈ 24 h,
propio de la operación circadiana de los transformadores.

---

### 1.2.b) Justificación de L = 96 como longitud de ventana

Con los valores reales de la ACF obtenidos:

- ρ(24) = **0.9250** — la correlación con 1 día atrás sigue siendo muy alta.
- ρ(48) = **0.8769** — la correlación con 2 días atrás sigue siendo sustancial.
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

> *Pendiente de implementación.*

---

## Task 3

> *Pendiente de implementación.*

---

## Task 4

> *Pendiente de implementación.*
