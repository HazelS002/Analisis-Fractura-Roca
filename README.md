# Análisis de Fracturas en Rocas - Guanajuato Capital

## Descripción del Proyecto
En este proyecto se analiza los patrones subyacentes de lozas del área de
Guanajuato Capital mediante técnicas de procesamiento digital de imágenes y
análisis computacional sobre calcos de las lozas hechos por estudiantes de la
Universidad de Guanajuato.

## Estructura del Proyecto

```
Analisis-Fractura-Roca/
├── src/                          # Código fuente principal
├── data/                        # Gestión de datos
│   ├── raw/                     # Datos originales
│   ├── processed/               # Datos procesados
│   └── sample-images/           # Muestra inicial de imágenes
├── report/                      # Reporte de resultados
├── scripts/                     # Scripts de utilidad
```

## Instalación y Configuración

### Prerrequisitos
```bash
# Clonar el repositorio
git clone https://github.com/HazelS002/Analisis-Fractura-Roca.git
cd Analisis-Fractura-Roca

# Crear entorno de Conda
conda env create -f environment.yml

# Activar entorno de Conda
conda activate imagen-denoising
```

### Dependencias
#### Dependencias principales
- **OpenCV**: Procesamiento de imágenes y filtros de denoising
- **scikit-image**: Algoritmos avanzados de eliminación de ruido
- **PyTorch**: Redes neuronales para denoising profundo
- **TensorFlow**: Alternativa para modelos de machine learning
- **NumPy**: Cálculos numéricos y operaciones con arrays de imágenes
- **Matplotlib**: Visualización de resultados antes/después
<!-- - **Pillow**: Manipulación básica de formatos de imagen -->
- **scipy**: Funciones matemáticas avanzadas para procesamiento

#### Dependencias secundarias
- **seaborn**: Estilos mejorados para visualización
- **scikit-learn**: Utilidades de preprocesamiento y métricas
- **imageio**: Lectura/escritura de múltiples formatos de imagen
<!-- - **torchvision**: Modelos preentrenados y datasets para PyTorch -->

#### Otras dependencias
<!-- - **pandas**: Análisis de métricas de calidad (PSNR, SSIM) -->
- **PyMuPDF**: Conversión de PDF a PNG (si se necesita procesar documentos)


## Uso del Proyecto

### Conversión de PDFs a Imágenes
```bash
# Convertir todos los PDFs de data/raw/pdfs a PNG
python scripts/convert_pdfs.py    # (especificar rutas en el script)
```

**Parámetros configurables:**
- Calidad de imagen (DPI)
- Formato de salida (PNG, JPG)
- Carpeta de entrada/salida

### Ejecución del código fuente en SRC
```bash
# Ejecutar 
python -m src.<file-name>
```

**Funcionalidades en desarrollo:**
- Detección de bordes de fracturas
- Análisis de patrones geométricos
- Controlar rutas desde un sólo archivo


## Estructura de Datos

### Formato de Entrada
- **PDFs**: Documentos escaneados en `data/raw/pdfs/`
- **Imágenes**: PNG en `data/raw/images/`

<!-- ### Formato de Salida
- Imágenes procesadas en `data/processed/`
- Resultados de análisis en `reports/figures/`
- Métricas y datos en `reports/` -->

## Desarrollo

### Agregar Nuevas Funcionalidades
1. Crear módulo en `src/` correspondiente
2. Actualizar `environment.yml` si es necesario

### Estructurar Nuevos Módulos
```python
# Ejemplo de estructura de módulo
nuevo_modulo/
├── __init__.py
├── . . .
└── helpers.py
```

## Contribución

1. Fork del proyecto
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## Mantenimiento

### Actualizar Dependencias
```bash
conda env update -f environment.yml --prune
```

### Compilar Reporte
```bash
# Generar reportes
cd reports/
pdflatex main.tex
```

## Troubleshooting

### Problemas Comunes

**PDFs no se convierten:**
- Verificar que PyMuPDF esté instalado
- Revisar permisos de archivos

**Imágenes no se cargan:**
- Verificar rutas en `src/utils/load_images.py`
- Confirmar formato de archivos

## Contacto

- **Autor**: Hazel Shamed
- **Repositorio**: [github.com/HazelS002/Analisis-Fractura-Roca](https://github.com/HazelS002/Analisis-Fractura-Roca)
- **Institución**: Universidad de Guanajuato

---

*Este proyecto está en desarrollo activo. La estructura y funcionalidades pueden cambiar.*