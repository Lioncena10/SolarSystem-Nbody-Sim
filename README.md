# Simulación Numérica de N-Cuerpos: Perturbación Urano-Neptuno

Este proyecto implementa una **simulación numérica de N-Cuerpos** en Python para analizar la dinámica gravitacional del Sistema Solar, centrándose en la histórica perturbación orbital que Neptuno ejerce sobre Urano.

## 🔭 Sobre el Proyecto

El descubrimiento de Neptuno en 1846 marcó un hito en la mecánica celeste, al ser el primer planeta hallado mediante predicciones matemáticas basadas en las irregularidades orbitales de Urano. 

El presente trabajo tiene como objetivo analizar esta perturbación gravitacional empleando herramientas computacionales modernas. Para ello, se desarrolló una simulación que integra las ecuaciones de movimiento durante el periodo **1800–1965**.

## ⚙️ Metodología

* **Modelo Físico:** Problema de los $N$-Cuerpos (Newtoniano).
* **Integrador Numérico:** Método de Runge-Kutta de cuarto orden (RK4).
* **Lenguaje:** Python.
* **Paso Temporal:** $\Delta t = 1$ día.
* **Condiciones Iniciales:** Basadas en efemérides del JPL (NASA).

## 📊 Resultados Clave

El modelo mostró una alta consistencia con las efemérides reales, validando la precisión del integrador RK4:

* **Precisión:** Errores relativos menores al **0.03%** en las posiciones finales de los planetas exteriores.
* **Desviación Máxima:** La interacción con Neptuno generó un desplazamiento en Urano de **$5.16 \times 10^9$ m** hacia 1947.
* **Desplazamiento Angular:** Esta discrepancia equivale a aproximadamente **$391''$** (segundos de arco).

> **Conclusión:** Este valor es dos órdenes de magnitud superior a la incertidumbre instrumental del siglo XIX, confirmando que las anomalías históricas de Urano no fueron aleatorias, sino la manifestación determinista de las aceleraciones inducidas por la masa de Neptuno.

## ✒️ Autor
**León Rodríguez** - Estudiante de la Licenciatura en física, CUCEI (UDG).
