# Prompts utilizados – Laboratorio 7

## Task 1 – Preprocesamiento y ACF

### Prompt utilizado

Construye el pipeline usando únicamente `urllib.request`, NumPy y tensores PyTorch: descarga el CSV desde la URL indicada, extrae `OT`, normalízala con z-score guardando `mu` y `sigma`, y separa cronológicamente 60/20/20 sin mezclar. Implementa `make_windows(series, L, H)` con entrada `(N, L)` y objetivo `(N,)` para `L=96`, `H=24` y `H=48`. Después implementa manualmente `acf_manual` hasta lag 96 con la fórmula del enunciado, grafica el correlograma con bandas `±1.96/sqrt(N)`.
### ¿Por qué funcionó este prompt?

Delimita datos, fórmulas, shapes, partición temporal y pruebas de aceptación, de modo que la IA no tiene que adivinar ni el protocolo de evaluación ni la convención de las ventanas. También preserva el contexto del notebook y exige que la justificación use resultados calculados, evitando una interpretación genérica de la ACF.

## Task 2 – LSTM manual y evaluación

### Prompt utilizado

Sobre las ventanas y splits ya creados, implementa `LSTMForecaster(d_in=1, d_h=32)` en PyTorch puro, sin `nn.LSTM`: declara con `nn.Parameter` `Wx`, `Wh`, `b`, `W_out` y `b_out`, procesa los 96 pasos en un bucle y aplica las compuertas `[i, f, g, o]` exactamente como define el enunciado. Entrena modelos independientes para H=24 y H=48 con Adam, 20 épocas y batch size 64; reutiliza funciones de entrenamiento y predicción cuando sea útil. Reporta MAE, RMSE y `MAE_modelo / MAE_naive` sobre validación, dibuja ambas curvas de pérdida y ejecuta la celda de verificación. Documenta en el README la pérdida elegida y justifícala en términos de la serie normalizada y la métrica final.

### ¿Por qué funcionó este prompt?

Especifica tanto el contrato matemático de la celda como la configuración experimental, lo que impide sustituir la implementación requerida por una capa de alto nivel. Al pedir métricas, gráfica, baseline y verificación en el mismo encargo, mantiene trazabilidad entre entrenamiento, evaluación y explicación escrita.

## Task 3 – Transformer y atención

### Prompt utilizado

Implementa los Tasks 3.1–3.3 respetando que no podemos usar `nn.MultiheadAttention`. Crea `make_pe(L, d_model)` sinusoidal no aprendible y registra el PE como buffer. En `TransformerForecaster(d_model=32, n_heads=2, d_ff=64, L=96)`, proyecta la entrada escalar, implementa manualmente Q/K/V, separación y concatenación de 2 cabezas, `softmax(QKᵀ/sqrt(d_k))`, FFN, LayerNorm aprendible y las dos conexiones Add & Norm. El `forward` debe devolver `(pred, attn_w)` si `return_attn=True`, con `pred` de forma `(B,)` y atención `(B,2,96,96)`. Entrena un modelo por horizonte con las mismas condiciones del LSTM, calcula MAE/RMSE/ratio, y para cinco ejemplos de validación promedia la atención sobre ejemplos y cabezas, genera heatmaps y convierte las posiciones con mayor peso desde CLS a lags, donde lag 1 es el valor más reciente. Ejecuta los asserts del enunciado antes de cerrar la tarea.

### ¿Por qué funcionó este prompt?

Fija las restricciones arquitectónicas, las formas intermedias y el significado de cada eje de atención, que son los puntos donde una implementación aparentemente correcta suele fallar. La instrucción conecta explícitamente entrenamiento, extracción de pesos y conversión posición-lag, por lo que las figuras quedan listas para el análisis comparativo posterior.

## Task 4 – Análisis final

### Prompt utilizado

Con los resultados realmente impresos por el notebook, redacta el Task 4 en el README sin inventar cifras. Compara los picos locales de la ACF con los lags de mayor atención desde CLS, incluyendo para cada afirmación los valores numéricos de ACF y atención; distingue coincidencia aproximada de coincidencia exacta y plantea una hipótesis verificable si hay desplazamientos. Explica qué dependencias condicionales y no lineales puede seleccionar atención frente a la ACF lineal, sin afirmar causalidad solo a partir de un mapa promedio. Completa la tabla LSTM/Transformer/naive para 24 y 48 h, explica el aumento del error mediante acumulación iterativa y, para la predicción directa implementada, mediante incertidumbre condicional. Finalmente compara el flujo de gradiente recurrente con las rutas directas de atención/residual y relaciónalo cautelosamente con los resultados observados.

### ¿Por qué funcionó este prompt?

Obliga a basar la interpretación en evidencia numérica del experimento y separa observación, hipótesis y conclusión arquitectónica. Además, señala las dos sutilezas del análisis —que el modelo implementado es directo por horizonte y que atención promedio no prueba causalidad—, lo que mantiene la respuesta técnicamente precisa y alineada con el enunciado.
