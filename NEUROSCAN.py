import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import google.generativeai as genai

# 1. ARQUITECTURA CNN COMPACTA Y DENSA (Basada en Özkaraca et al., 2023)
class BestCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BestCNN, self).__init__()
       
        # Extractor de Características jerárquicas locales
        self.features = nn.Sequential(
            # Bloque Convolucional 1: Detección de gradientes y bordes crudos
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reducción de resolución a la mitad
            nn.Dropout(0.25),
           
            # Bloque Convolucional 2: Formas geométricas anatómicas iniciales
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
           
            # Bloque Convolucional 3: Texturas tisulares y asimetrías complejas
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
           
            # Bloque Convolucional 4: Abstracción de alto nivel patológico
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
           
            # Pooling Adaptativo: Concede invarianza de resolución espacial
            nn.AdaptiveAvgPool2d((8, 8))
        )
       
        # Clasificador Denso Multiclase totalmente conectado
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularización extrema contra sobreajuste
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

NOMBRES_CLASES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# SELECCIÓN DINÁMICA DE HARDWARE DE ACELERACIÓN LÓGICA
if torch.cuda.is_available():
    device = torch.device("cuda")
    estado_hardware = "SISTEMA ACELERADO EN NVIDIA CUDA GPU"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    estado_hardware = "SISTEMA ACELERADO EN APPLE SILICON MPS"
else:
    device = torch.device("cpu")
    estado_hardware = "SISTEMA BAJO PROCESAMIENTO CENTRAL STANDARD (CPU)"

# CONFIGURACIÓN DEL ENTORNO WEB DE STREAMLIT
st.set_page_config(
    page_title="NeuroScanAI Core Gateway",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INYECCIÓN DE HOJAS DE ESTILO CSS PARA ENTORNO CLARO CLÍNICO
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

.header-container {
    padding: 24px 32px;
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    border-bottom: 4px solid #3B82F6;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

.main-title {
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #FFFFFF !important;
    margin-bottom: 6px;
}

.subtitle {
    font-size: 14px;
    color: #93C5FD;
    font-weight: 400;
    line-height: 1.5;
}

.hardware-badge {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #38BDF8;
    background-color: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 8px;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #E2E8F0;
    padding: 4px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    font-size: 13px;
    font-weight: 500;
    background-color: transparent;
    color: #475569 !important;
    border-radius: 6px;
    padding: 0px 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #1E3A8A !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.pro-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.01);
}

.metric-box {
    background: #F1F5F9;
    border-left: 4px solid #2563EB;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 10px;
    font-size: 13px;
    color: #334155;
}

div.stButton > button:first-child {
    background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 10px 24px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    width: 100%;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.15) !important;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%) !important;
    transform: translateY(-0.5px);
}

.clinical-alert-tumor {
    padding: 14px;
    background-color: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-radius: 6px;
    color: #991B1B;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 15px;
}

.clinical-alert-healthy {
    padding: 14px;
    background-color: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 6px;
    color: #065F46;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 15px;
}

.theo-subtitle {
    font-size: 20px;
    color: #1E3A8A;
    font-weight: 600;
    margin-top: 28px;
    margin-bottom: 14px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 6px;
}

.math-display {
    background-color: #F8FAFC;
    padding: 24px;
    border-radius: 8px;
    border-left: 5px solid #2563EB;
    margin: 20px 0;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
    text-align: center;
}

.edu-link {
    color: #2563EB !important;
    text-decoration: underline !important;
    font-weight: 600;
}

.simulated-image-box {
    background-color: #F1F5F9;
    border: 2px dashed #CBD5E1;
    border-radius: 6px;
    padding: 20px;
    text-align: center;
    color: #64748B;
    font-size: 12px;
    font-style: italic;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# CONFIGURACIÓN LATERAL PARA CLAVE DE GEMINI GRATUITA
with st.sidebar:
    st.markdown("### Credenciales del Sistema")
    api_key_gemini = st.text_input(
        "Google Gemini API Key:", 
        type="password", 
        help="Consigue tu clave de desarrollo gratuita en aistudio.google.com"
    )
    st.markdown("---")
    if api_key_gemini:
        genai.configure(api_key=api_key_gemini)
        st.success("Enlace a la API de Google activo.")
    else:
        st.warning("Operando en modo local desconectado.")

st.markdown(f"""
    <div class='header-container'>
        <div class='main-title'>NEUROSCAN AI GATEWAY</div>
        <div class='subtitle'>Entorno Unificado Clínico-Académico para la Evaluación de Arquitecturas Convolucionales Dedicadas (Dense CNN) Entrenadas desde Cero en Diagnóstico de Resonancia Magnética (MRI).</div>
        <div class='hardware-badge'>{estado_hardware}</div>
    </div>
""", unsafe_allow_html=True)

tab_teoria, tab_tutor, tab_quiz, tab_laboratorio = st.tabs([
    "Teoría", 
    "Chat", 
    "Quizz", 
    "Laboratorio"
])


# PESTAÑA 1: ENCICLOPEDIA TEÓRICA Y CIENTÍFICA


with tab_teoria:
    col_t1, col_t2 = st.columns([7, 3])
   
    with col_t1:
        st.markdown("""
        <div class='pro-card'>
        <h2>¿Cómo aprende la Red Neuronal?</h2>
        <p>El propósito fundamental de este software es resolver un problema de <strong>clasificación multiclase</strong> en el ámbito médico, logrando discriminar entre 4 estados biológicos independientes utilizando imágenes de Resonancia Magnética Cerebral (MRI).</p>
        <p>A diferencia de los enfoques comerciales que utilizan <em>Transfer Learning</em> (reutilizar redes entrenadas con fotos cotidianas de la base de datos ImageNet), la investigación científica de referencia de <strong>Özkaraca, Bağrıaçık, Gürüler et al. (2023)</strong> demostró que las texturas y escalas de grises analógicas de la medicina radiológica requieren un diseño especializado. Por lo tanto, esta arquitectura <strong>BestCNN se ha estructurado y entrenado completamente desde cero</strong>, garantizando un aprendizaje preciso de la firma anatómica humana.</p>
       
        <div class='theo-subtitle'>1. Taxonomía del Dataset de M. Nickparvar (Kaggle)</div>
        <p>El modelo procesa un volumen de <strong>7,021 imágenes médicas totales</strong> distribuidas de forma equilibrada para entrenar filtros específicos sobre cada patología:</p>
        <ul>
            <li><strong>Glioma (1,621 imágenes):</strong> Neoplasias infiltrantes derivadas de células gliales. Poseen contornos sumamente difusos que se mezclan con el tejido cerebral sano.</li>
            <li><strong>Meningioma (1,645 imágenes):</strong> Tumores extracraneales encapsulados que crecen en las meninges. Ejercen un efecto de masa que desplaza la corteza sin infiltrarla, mostrando bordes muy definidos.</li>
            <li><strong>Tumor Pituitario / Adenoma de Hipófisis (1,757 imágenes):</strong> Neoplasias localizada en la fosa de la silla turca basal, comprimiendo Estructuras Ópticas.</li>
            <li><strong>No Tumor / Cerebro Sano (2,000 imágenes):</strong> Grupo de control que exhibe perfecta simetría interhemisférica y ventrículos normoconfigurados.</li>
        </ul>


        <div class='theo-subtitle'>2. Preprocesamiento e Invarianza Espacial de Entrada</div>
        <p>Las imágenes crudas poseen diferentes tamaños lumínicos y resoluciones. Para estandarizar el flujo del tensor, el script realiza un rediseño de dimensiones a <strong>128x128 píxeles</strong> (<code>IMG_SIZE = 128</code>). Reducir de las dimensiones originales de 224 a 128 permite acelerar el cómputo local <strong>3 veces más rápido</strong> conservando intactas las estructuras morfológicas críticas.</p>
        <p>Para evitar que el modelo sufra de "hiperconfianza" o memorice de forma estática las posiciones (sobreajuste), se implementan técnicas de <strong>Aumento de Datos (Data Augmentation)</strong> como rotaciones aleatorias, volteo de espejo y un escalado estadístico que mapea los píxeles (0 a 255) a rangos continuos uniformes.</p>
       
        <div class='theo-subtitle'>3. El Extractor Convolucional: Operación Matemática de Ventana</div>
        <p>A nivel matemático, cada una de las 4 etapas convolucionales secuenciales de la red desliza un filtro pequeño o matriz numérica de 3x3 (llamado <em>kernel</em>) sobre la superficie de la imagen. La operación computa la suma del producto de elementos correlacionados espacialmente para extraer características visuales como bordes, límites óseos y densidades tumorales:</p>
        </div>
        """, unsafe_allow_html=True)
       
        # ECUACIÓN 1: OPERACIÓN CONVOLUCIONAL (NATIVA STREAMLIT - GIGANTE)
        st.markdown("<div class='math-display'>", unsafe_allow_html=True)
        st.latex(r"\Large S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} \sum_{c} I(i - m, j - n, c) K(m, n, c)")
        st.markdown("<br><a class='edu-link' href='https://es.wikipedia.org/wiki/Red_neuronal_convolucional' target='_blank'>Ver Guía de Operaciones Convolucionales</a></div>", unsafe_allow_html=True)
       
        st.markdown("""
        <div class='pro-card'>
        <div class='theo-subtitle'>4. Batch Normalization (Estabilización Intercapa)</div>
        <p>A medida que la red se hace profunda, los valores de los gradientes sufren desviaciones drásticas en cada lote de imágenes (fenómeno llamado <em>Cambio de Covarianza Interna</em>). Para solucionarlo, la capa de normalización recalcula la media y la varianza de cada grupo de 32 imágenes (Batch Size = 32) para estabilizar los datos:</p>
        </div>
        """, unsafe_allow_html=True)


        # ECUACIÓN 2: BATCH NORMALIZATION (NATIVA STREAMLIT - GIGANTE)
        st.markdown("<div class='math-display'>", unsafe_allow_html=True)
        st.latex(r"\Large \mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i \quad \Longleftrightarrow \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2")
        st.markdown("<br><a class='edu-link' href='https://en.wikipedia.org/wiki/Batch_normalization' target='_blank'>Ver Documentación de Batch Normalization</a></div>", unsafe_allow_html=True)


        st.markdown("""
        <div class='pro-card'>
        <div class='theo-subtitle'>5. Funciones de Activación: El uso de ReLU</div>
        <p>Los modelos antiguos utilizaban la función <em>Sigmoide</em>, la cual comprime los datos en un rango de [0,1]. Sin embargo, la función Sigmoide sufre de saturación: para valores muy altos o bajos su derivada es casi cero, deteniendo por completo el aprendizaje. Para resolverlo, implementamos la <strong>Unidad Lineal Rectificada (ReLU)</strong>. Si la entrada es negativa, se apaga (0); si es positiva, deja pasar el gradiente intacto con una derivada constante de 1:</p>
        </div>
        """, unsafe_allow_html=True)


        # ECUACIÓN 3: RELU (NATIVA STREAMLIT - GIGANTE)
        st.markdown("<div class='math-display'>", unsafe_allow_html=True)
        st.latex(r"\Large f(x) = \max(0, x)")
        st.markdown("<br><a class='edu-link' href='https://es.wikipedia.org/wiki/Rectificador_(redes_neuronales)' target='_blank'>Ver Explicación Gráfica de la función ReLU</a></div>", unsafe_allow_html=True)


        st.markdown("""
        <div class='pro-card'>
        <div class='theo-subtitle'>6. Clasificación Final y Distribución Exponencial Softmax</div>
        <p>En la última etapa del programa, la matriz tridimensional se convierte en un vector plano mediante <code>nn.Flatten()</code>. Las neuronas densas finales calculan unas puntuaciones crudas conocidas como <em>logits</em> (z). Para convertir estas puntuaciones en porcentajes comprensibles para el médico que sumen exactamente 100% (1.0), se procesan a través de la ecuación de <strong>Softmax</strong>:</p>
        </div>
        """, unsafe_allow_html=True)


        # ECUACIÓN 4: SOFTMAX (NATIVA STREAMLIT - GIGANTE)
        st.markdown("<div class='math-display'>", unsafe_allow_html=True)
        st.latex(r"\Large P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^{4} e^{z_j}}")
        st.markdown("<br><a class='edu-link' href='https://es.wikipedia.org/wiki/Funci%C3%B3n_Softmax' target='_blank'>Ver Guía de Distribuciones de Probabilidad Softmax</a></div>", unsafe_allow_html=True)


        st.markdown("""
        <div class='pro-card'>
        <div class='theo-subtitle'>7. Métodos de Control de Calidad</div>
        <ul>
            <li><strong>Dropout (Desactivación Aleatoria):</strong> El clasificador apaga aleatoriamente el 50% de sus conexiones neuronales en cada paso del entrenamiento. Esto obliga a la red a aprender caminos alternativos robustos, impidiendo la memorización estática de datos. Explora visualmente este comportamiento en el <a class='edu-link' href='https://poloclub.github.io/cnn-explainer/' target='_blank'>CNN Explainer Interactivo</a>.</li>
            <li><strong>Label Smoothing (Suavizado de Etiquetas):</strong> Al calcular la penalización de errores con la función de <em>Entropía Cruzada Categórica</em>, suavizamos los objetivos reales de [0, 1] inyectando un margen de incertidumbre épsilon = 0.1. Esto enseña al programa a no clasificar con un 100% de confianza ciega, tolerando ruidos y variaciones del equipamiento médico de resonancia.</li>
        </ul>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div class='pro-card'>
        <div class='theo-subtitle'>8. Infraestructura y Fundamentos de la Interfaz del Usuario en Streamlit</div>
        <p>Para trasladar este modelo matemático a un entorno clínico interactivo utilizable por médicos, se seleccionó el framework de <strong>Streamlit</strong> debido a su ejecución síncrona reactiva y su capacidad de manipular tensores de PyTorch en tiempo real.</p>
        <p>La arquitectura de ejecución de la interfaz web se rige bajo los siguientes principios de ingeniería de software:</p>
        <ul>
            <li><strong>Gestión del Estado de Sesión (st.session_state):</strong> Streamlit reejecuta todo el script de arriba a abajo cada vez que el usuario interactúa con un botón. Para evitar que el historial de los chats o los resultados de la inferencia médica se borren en cada renderizado, los datos se almacenan en el búfer permanente de memoria del servidor mediante diccionarios de estado de sesión.</li>
            <li><strong>Optimización de Memoria Mediante Caché (st.cache_resource):</strong> La red neuronal posee millones de pesos sinápticos que se cargan desde el disco duro (<code>mejor_modelo.pth</code>). Realizar esta lectura en cada interacción congelaría la experiencia del usuario. Al envolver la función bajo el decorador de recursos en caché, el modelo se aloja en la memoria RAM/VRAM de forma persistentemente una única vez durante el ciclo de vida del proceso.</li>
            <li><strong>Estructura de Diseño Modular por Columnas y Pestañas:</strong> El diseño separa la lógica pesada del laboratorio de inferencia de los módulos teóricos y académicos mediante contenedores dinámicos (<code>st.tabs</code>) y rejillas proporcionales (<code>st.columns</code>). Esto permite que el médico examine la imagen médica y los histogramas de Softmax de forma paralela en una sola pantalla sin desbordamientos visuales.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


    with col_t2:
        st.markdown("""
        <div class='pro-card'>
        <h4>Métricas de Rendimiento Obtenidas</h4><br>
        <div class='metric-box'><strong>Dataset de Validación:</strong> Split de datos 80% Entrenamiento y 20% Prueba.</div>
        <div class='metric-box'><strong>Precisión Global (Accuracy):</strong> 94.44% alcanzado de forma óptima en la Época 48 de entrenamiento.</div>
        <div class='metric-box'><strong>F1-Score Combinado (Macro):</strong> 0.9436, indicando un balance de alta fidelidad diagnóstica.</div>
        <div class='metric-box'><strong>Tiempo de Ejecución Local:</strong> 24 minutos en entorno unificado con aceleración por GPU.</div>
        <div class='metric-box'><strong>Validación Científica del Paper:</strong> El equipo de Özkaraca utilizó el método de <strong>Validación Cruzada de 10 pliegues (10-Fold Cross-Validation)</strong>. Esta técnica divide todo el dataset en 10 partes idénticas y entrena el modelo 10 veces distintas alternando el set de testeo; de esta manera, se demuestra de forma estadística que la precisión final del <strong>96%</strong> es sólida y libre de sesgos por distribución aleatoria de carpetas.</div>
        </div>
        
        <div class='pro-card'>
        <h4>Tabla Comparativa de Modelos Literarios</h4>
        <p>Comparativa formal de rendimientos en la clasificación automática de tumores sobre el mismo dataset:</p>
        <table style='width:100%; font-size:12px; border-collapse: collapse; text-align: left;'>
            <thead>
                <tr style='background-color:#F1F5F9; border-bottom:2px solid #CBD5E1;'>
                    <th style='padding:6px;'>Modelo</th>
                    <th style='padding:6px; text-align:center;'>Exactitud</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom:1px solid #E2E8F0;'>
                    <td style='padding:6px;'>CNN Convencional Básica</td>
                    <td style='padding:6px; text-align:center;'>~72%</td>
                </tr>
                <tr style='border-bottom:1px solid #E2E8F0;'>
                    <td style='padding:6px;'>VGG16 (Transfer Learning)</td>
                    <td style='padding:6px; text-align:center;'>~85%</td>
                </tr>
                <tr style='border-bottom:1px solid #E2E8F0;'>
                    <td style='padding:6px;'>DenseNet (Transfer Learning)</td>
                    <td style='padding:6px; text-align:center;'>~88%</td>
                </tr>
                <tr style='border-bottom:1px solid #E2E8F0; font-weight:bold; color:#1E3A8A;'>
                    <td style='padding:6px;'>Código PyTorch (Split 80/20)</td>
                    <td style='padding:6px; text-align:center;'>94.44%</td>
                </tr>
                <tr style='font-weight:bold; color:#10B981;'>
                    <td style='padding:6px;'>Modelo Original (10-Fold)</td>
                    <td style='padding:6px; text-align:center;'>96.00%</td>
                </tr>
            </tbody>
        </table>
        </div>

        <div class='pro-card'>
        <h4>Fundamento Clínico de la Clasificación</h4>
        <p style='font-size:13px; color:#475569; line-height:1.4;'>La Red Neuronal analiza patrones radiológicos específicos en secuencias ponderadas de Resonancia Magnética para diferenciar las firmas patológicas:</p>
        
        <div style='background-color:#F8FAFC; padding:10px; border-radius:6px; border-left:3px solid #991B1B; margin-bottom:8px; font-size:12px;'>
            <strong style='color:#991B1B;'>• Gliomas:</strong> Tienden a mostrar márgenes mal definidos, edema vasogénico perilesional extenso y realce heterogéneo tras la administración de contraste debido a la ruptura de la barrera hematoencefálica.
        </div>
        
        <div style='background-color:#F8FAFC; padding:10px; border-radius:6px; border-left:3px solid #1E3A8A; margin-bottom:8px; font-size:12px;'>
            <strong style='color:#1E3A8A;'>• Meningiomas:</strong> Lesiones extraaxiales de base amplia en la duramadre. Se caracterizan por un realce homogéneo e intenso y el clásico signo de la "cola dural".
        </div>
        
        <div style='background-color:#F8FAFC; padding:10px; border-radius:6px; border-left:3px solid #D97706; margin-bottom:8px; font-size:12px;'>
            <strong style='color:#D97706;'>• Tumores Pituitarios:</strong> Masas ubicadas en la silla turca. Los macroadenomas pueden causar una deformación del quiasma óptico, conocida radiológicamente como el "signo del muñeco de nieve".
        </div>
        
        <div style='background-color:#F8FAFC; padding:10px; border-radius:6px; border-left:3px solid #065F46; margin-bottom:12px; font-size:12px;'>
            <strong style='color:#065F46;'>• No Tumor:</strong> Tejido cerebral normal con simetría estructural conservada, diferenciación clara entre sustancia gris y blanca, y sistema ventricular de dimensiones normales.
        </div>
        
        <div style='text-align: center; margin-top: 15px; padding-top: 10px; border-top: 1px solid #E2E8F0;'>
            <a class='edu-link' href='https://radiopaedia.org/articles/brain-tumors' target='_blank' style='font-size: 13px;'>
                🌐 Acceder al Atlas Radiológico Completo (Casos Clínicos e Imágenes)
            </a>
        </div>
        </div>
        """, unsafe_allow_html=True)


# PESTAÑA 2: CONSULTORÍA ACADÉMICA VIRTUAL (Gemini Gratuito)

with tab_tutor:
    st.markdown("<div class='pro-card'><h4>Consola de Diálogo Clínico e Ingeniería</h4>"
                "Asistente virtual potenciado por <strong>Google Gemini (Nivel Gratuito)</strong> para resolver dudas de código, arquitectura de la red o detalles patológicos de tumores de forma interactiva.</div>", unsafe_allow_html=True)

    if "historial_chat" not in st.session_state:
        st.session_state.historial_chat = [
            {"role": "assistant", "content": "Hola, soy tu asistente académico. Puedes preguntarme detalles técnicos del paper de Özkaraca, el funcionamiento de las capas convolucionales o las características de los tumores cerebrales."}
        ]

    for mensaje in st.session_state.historial_chat:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    if entrada_usuario := st.chat_input("Escribe tu duda aquí...", key="chat_general_input"):
        st.session_state.historial_chat.append({"role": "user", "content": entrada_usuario})
        with st.chat_message("user"):
            st.write(entrada_usuario)

        with st.chat_message("assistant"):
            if not api_key_gemini:
                respuesta_tutor = " Por favor, ingresa tu clave gratuita de Google Gemini en la barra lateral izquierda para activar la inteligencia artificial sin costo."
                st.write(respuesta_tutor)
                st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_tutor})
            else:
                try:
                    model_gemini = genai.GenerativeModel('gemini-2.5-flash') # O usa 'gemini-2.5-flash'
                    prompt_contexto = f"Actúa como un profesor experto. Responde académicamente y con rigurosidad: {entrada_usuario}"
                    respuesta_objeto = model_gemini.generate_content(prompt_contexto)
                    st.write(respuesta_objeto.text)
                    st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_objeto.text})
                except Exception as e:
                    st.error(f"Error al conectar con Gemini: {e}")


# PESTAÑA 3: ACREDITACIÓN DE CONOCIMIENTO (QUIZ DINÁMICO VIA GEMINI)

with tab_quiz:
    st.markdown("<div class='pro-card'><h4>Evaluación Generativa Aleatoria por IA</h4>", unsafe_allow_html=True)
    
    if "quiz_dinamico" not in st.session_state:
        st.session_state.quiz_dinamico = None
    if "quiz_indice" not in st.session_state:
        st.session_state.quiz_indice = 0
    if "quiz_aciertos" not in st.session_state:
        st.session_state.quiz_aciertos = 0
    if "quiz_evaluado" not in st.session_state:
        st.session_state.quiz_evaluado = False

    if st.button("Generar Cuestionario Aleatorio vía API"):
        if not api_key_gemini:
            st.error(" Ingrese su Gemini API Key en la barra lateral para generar el test dinámico.")
        else:
            prompt_quiz = (
                "Genera un examen técnico en formato JSON estricto sobre redes neuronales convolucionales y tumores cerebrales. "
                "Debe ser un objeto JSON con un arreglo llamado 'preguntas'. Cada pregunta debe tener: "
                "'pregunta' (string), 'opciones' (arreglo de 3 strings), 'correcta' (entero 0 al 2 que representa el índice correcto), "
                "y 'explicacion' (string científico). Genera exactamente 3 preguntas. No incluyas markdown fuera del JSON."
            )
            with st.spinner("La IA está redactando las preguntas en tiempo real..."):
                try:
                    model_gemini = genai.GenerativeModel('gemini-2.5-flash') # O usa 'gemini-2.5-flash'
                    res_raw = model_gemini.generate_content(prompt_quiz).text
                    if "```json" in res_raw:
                        res_raw = res_raw.split("```json")[1].split("```")[0].strip()
                    elif "```" in res_raw:
                        res_raw = res_raw.split("```")[1].split("```")[0].strip()
                    datos = json.loads(res_raw)
                    st.session_state.quiz_dinamico = datos["preguntas"]
                    st.session_state.quiz_indice = 0
                    st.session_state.quiz_aciertos = 0
                    st.session_state.quiz_evaluado = False
                    st.rerun()
                except Exception as parse_err:
                    st.error(f"Error al estructurar el examen: {str(parse_err)}")

    if st.session_state.quiz_dinamico is not None:
        preguntas_activas = st.session_state.quiz_dinamico
        total_q = len(preguntas_activas)

        if st.session_state.quiz_indice < total_q:
            q_actual = preguntas_activas[st.session_state.quiz_indice]
            st.write(f"**Progreso:** Evaluación {st.session_state.quiz_indice + 1} / {total_q}")
            st.write(q_actual["pregunta"])
            
            seleccion = st.radio(
                label="Seleccione el postulado correcto:",
                options=["Seleccione..."] + q_actual["opciones"],
                key=f"dq_{st.session_state.quiz_indice}"
            )
            
            if st.button("Validar Respuesta") and seleccion != "Seleccione...":
                indice_sel = q_actual["opciones"].index(seleccion)
                st.session_state.quiz_evaluado = True
                if indice_sel == q_actual["correcta"]:
                    st.success(f"¡Correcto! {q_actual['explicacion']}")
                    if f"dcont_{st.session_state.quiz_indice}" not in st.session_state:
                        st.session_state.quiz_aciertos += 1
                        st.session_state[f"dcont_{st.session_state.quiz_indice}"] = True
                else:
                    st.error(f"Incorrecto. {q_actual['explicacion']}")

            if st.session_state.quiz_evaluado:
                if st.button("Siguiente Pregunta"):
                    st.session_state.quiz_indice += 1
                    st.session_state.quiz_evaluado = False
                    st.rerun()
        else:
            st.success(f"¡Examen completado con éxito! Calificación final: {int((st.session_state.quiz_aciertos/total_q)*100)}%")
    st.markdown("</div>", unsafe_allow_html=True)


# PESTAÑA 4: LABORATORIO  

with tab_laboratorio:
    @st.cache_resource
    def cargar_modelo_en_gpu():
        modelo_red = BestCNN(num_classes=4)
        
        # OBTENER LA RUTA ABSOLUTA AUTOMÁTICA EN EL SERVIDOR DE STREAMLIT
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_archivo_pesos = os.path.join(directorio_actual, "mejor_modelo.pth")
        
        MODEL_DOWNLOAD_URL = "https://io.googleup.workers.dev/?id=1RaTULpeDLPxZ4H5Q338Xu8GyUyR-vxdN" 
        
        # Si el archivo NO existe en el servidor, se fuerza la descarga automatizada
        if not os.path.exists(ruta_archivo_pesos):
            with st.spinner("Descargando parámetros del modelo entrenado desde la nube... (Esto puede tardar un minuto)"):
                try:
                    import requests
                    response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
                    response.raise_for_status() # Verifica que el enlace no tire error 404 o 403
                    
                    with open(ruta_archivo_pesos, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success("¡Modelo de red neuronal descargado exitosamente!")
                except Exception as e:
                    return modelo_red, False, f"⚠️ Error de descarga. Corriendo en MODO DEMO. Detalle: {str(e)}"
        
        # Una vez que el archivo ya existe (local o recién descargado), se cargan los pesos
        if os.path.exists(ruta_archivo_pesos):
            try:
                estado_pesos = torch.load(ruta_archivo_pesos, map_location=device)
                if isinstance(estado_pesos, dict) and 'model' in estado_pesos:
                    estado_pesos = estado_pesos['model']
                modelo_red.load_state_dict(estado_pesos, strict=False)
                modelo_red.to(device).eval()
                return modelo_red, True, "Parámetros cargados correctamente en el hardware de inferencia."
            except Exception as e:
                return modelo_red, False, f"Modo Demostración activo. Error al estructurar pesos: {str(e)}"
                
        return modelo_red, False, "Archivo 'mejor_modelo.pth' no localizado."

    model, pesos_reales_listos, mensaje_estado_modelo = cargar_modelo_en_gpu()
    st.info(f"**Consola de Pesos:** {mensaje_estado_modelo}")
    
    col_panel_control, col_panel_imagen, col_panel_resultados = st.columns([3, 3, 4])
    
    with col_panel_control:
        st.markdown("<div class='pro-card'><h5>1. Entrada de Muestras</h5>", unsafe_allow_html=True)
        archivo_mri = st.file_uploader("Cargar Imagen MRI (PNG / JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        ejecutar_analisis = st.button("Ejecutar Análisis Diagnóstico")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_panel_imagen:
        st.markdown("<div class='pro-card' style='text-align: center;'><h5>2. Espécimen Médico</h5>", unsafe_allow_html=True)
        if archivo_mri is not None:
            imagen_entrada = Image.open(archivo_mri).convert('RGB')
            st.image(imagen_entrada, use_container_width=True)
        else:
            st.write("Esperando espécimen...")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_panel_resultados:
        st.markdown("<div class='pro-card'><h5>3. Vector de Salida Analítica</h5>", unsafe_allow_html=True)
        
        if "prediccion_hecha" not in st.session_state:
            st.session_state.prediccion_hecha = False
            st.session_state.clase_diagnosticada = ""
            st.session_state.porcentaje_confianza = 0.0
            st.session_state.mapa_probabilidades = None

        if archivo_mri is not None and ejecutar_analisis:
            transformacion_clinica = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_mri = transformacion_clinica(imagen_entrada).unsqueeze(0).to(device)
            with torch.no_grad():
                salida_logits = model(tensor_mri)
                mapa_probabilidades = torch.softmax(salida_logits, dim=1)[0].cpu().numpy()
                
            indice_prediccion = np.argmax(mapa_probabilidades)
            st.session_state.clase_diagnosticada = NOMBRES_CLASES[indice_prediccion]
            st.session_state.porcentaje_confianza = mapa_probabilidades[indice_prediccion] * 100
            st.session_state.mapa_probabilidades = mapa_probabilidades
            st.session_state.prediccion_hecha = True

        if st.session_state.prediccion_hecha:
            clase_diag = st.session_state.clase_diagnosticada
            confianza = st.session_state.porcentaje_confianza
            
            if clase_diag == "No Tumor":
                st.markdown(f"<div class='clinical-alert-healthy'>ESTUDIO COMPATIBLE CON NORMALIDAD: {clase_diag.upper()} ({confianza:.2f}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='clinical-alert-tumor'>ALERTA: ANOMALÍA DETECTADA — {clase_diag.upper()} ({confianza:.2f}%)</div>", unsafe_allow_html=True)
            
            for idx, nombre_item in enumerate(NOMBRES_CLASES):
                valor_porcentaje = st.session_state.mapa_probabilidades[idx] * 100
                st.write(f"<small>{nombre_item}: {valor_porcentaje:.1f}%</small>", unsafe_allow_html=True)
                st.progress(int(valor_porcentaje))
        st.markdown("</div>", unsafe_allow_html=True)

    # CHAT CLINICO DINÁMICO CON GEMINI
    if st.session_state.prediccion_hecha:
        st.markdown("<div class='pro-card'><h4>4. Consola de Discusión Clínica vía Gemini</h4></div>", unsafe_allow_html=True)
        
        if "chat_clinico_historial" not in st.session_state:
            st.session_state.chat_clinico_historial = [
                {"role": "assistant", "content": f"El modelo local clasificó el estudio como {st.session_state.clase_diagnosticada}. ¿Deseas hacerle alguna consulta a Gemini sobre esta patología?"}
            ]
            
        for msg in st.session_state.chat_clinico_historial:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        if pregunta_clinica := st.chat_input("Consulta a Gemini..."):
            st.session_state.chat_clinico_historial.append({"role": "user", "content": pregunta_clinica})
            with st.chat_message("user"):
                st.write(pregunta_clinica)
                
            with st.chat_message("assistant"):
                if not api_key_gemini:
                    st.write(" Ingrese la Gemini API Key en el panel lateral para procesar consultas clínicas.")
                else:
                    try:
                        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
                        prompt_clinico = f"El modelo local detectó {st.session_state.clase_diagnosticada} ({st.session_state.porcentaje_confianza:.2f}%). Responde rigurosamente a: {pregunta_clinica}"
                        res_clinica = model_gemini.generate_content(prompt_clinico).text
                        st.write(res_clinica)
                        st.session_state.chat_clinico_historial.append({"role": "assistant", "content": res_clinica})
                    except Exception as e:
                        st.error(f"Error de red: {e}")
