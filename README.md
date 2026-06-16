#  NeuroScan AI - Clasificación de Tumores Cerebrales mediante Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)

##  Descripción del Proyecto
**NeuroScan AI** es una plataforma interactiva de diagnóstico médico asistido por computadora (CAD). Utiliza una Red Neuronal Convolucional (CNN) optimizada y entrenada en **PyTorch** para analizar secuencias de Resonancia Magnética (IRM) y clasificar el tejido cerebral en cuatro categorías clínicas:
- **Glioma** (Márgenes mal definidos, edema vasogénico extenso).
- **Meningioma** (Lesiones extraaxiales, signo de la cola dural).
- **Tumor Pituitario** (Masas selares, signo del muñeco de nieve).
- **No Tumor** (Tejido cerebral estructuralmente sano).

El sistema integra el backend de Deep Learning con una interfaz web en **Streamlit**, permitiendo a profesionales de la salud o docentes cargar una imagen médica en tiempo real, visualizar los mapas de activación y obtener métricas estadísticas del diagnóstico junto con fundamentos teóricos-radiológicos.

##  Métricas de Rendimiento Obtenidas
El modelo (`BestCNN`) fue entrenado localmente con aceleración por GPU utilizando un split de datos 80/20 y optimizaciones de hiperparámetros basadas en la literatura científica de Özkaraca:

* **Precisión Global (Accuracy):** 94.44% (Alcanzado de forma óptima en la Época 48).
* **F1-Score Combinado (Macro):** 0.9436 (Garantiza un balance de alta fidelidad diagnóstica).
* **Robustez del Paper Original:** 96.00% empleando Validación Cruzada de 10 pliegues (*10-Fold Cross-Validation*).

##  Tecnologías Utilizadas
- **Lenguaje:** Python 3.9+
- **Framework de IA:** PyTorch / Torchvision (Procesamiento de imágenes y tensores).
- **Interfaz Web:** Streamlit (Componentes interactivos y renderizado HTML/CSS personalizado).
- **Procesamiento Numérico y Datos:** NumPy, Pillow (PIL), OS.

##  Cómo Ejecutar el Proyecto Localmente

1. Clona este repositorio:
```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
