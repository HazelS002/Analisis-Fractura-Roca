# Análisis de Fracturas en Rocas - Guanajuato Capital

## Descripción del Proyecto
Este proyecto analiza los patrones subyacentes en los calcos de lozas del área
de Guanajuato Capital mediante técnicas de procesamiento digital de imágenes y
análisis computacional.

## Estructura del Proyecto

```
Analisis-Fractura-Roca/
├── src/                          # Código fuente principal
│   ├── analysis/                # Análisis de patrones de fractura
│   ├── denoising/               # Eliminación de ruido en imágenes
│   └── utils/                   # Utilidades compartidas
├── data/                        # Gestión de datos
│   ├── raw/                     # Datos originales (PDFs)
│   ├── processed/               # Datos procesados (imágenes)
│   └── sample-images/           # Muestra inicial de imágenes
├── notebooks/                   # Análisis exploratorio
├── reports/                     # Reportes y documentación
├── tests/                       # Pruebas automatizadas
├── scripts/                     # Scripts de utilidad
```

## Instalación y Configuración

### Prerrequisitos
```bash
# Clonar el repositorio
git clone https://github.com/HazelS002/Analisis-Fractura-Roca.git
cd Analisis-Fractura-Roca

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### Dependencias principales
- OpenCV: Procesamiento de imágenes
- PyMuPDF: Conversión PDF a PNG
- NumPy: Cálculos numéricos
- Matplotlib: Visualización

## Uso del Proyecto

### 1. Conversión de PDFs a Imágenes
```bash
# Convertir todos los PDFs de data/raw/pdfs a PNG
python scripts/convert_pdfs.py
```

**Parámetros configurables:**
- Calidad de imagen (DPI)
- Formato de salida (PNG, JPG)
- Carpeta de entrada/salida

### 2. Eliminación de Ruido en Imágenes
```bash
# Ejecutar proceso de denoising en imágenes muestrales
python src/denoising/main.py

# O ejecutar como módulo
python -m src.denoising.main
```

**Características:**
- Aplica filtros de suavizado
- Prepara imágenes para análisis

### 3. Carga y Visualización de Imágenes
```python
from src.utils.load_images import read_images
from src.utils.visualization import plot_comparison

# Cargar imágenes
images, names = read_images("data/sample-images")

# Visualizar resultados
plot_comparison(original_img, processed_img, "Comparación")
```

### 4. Análisis de Patrones de Fractura
```bash
# Ejecutar análisis (desarrollándose)
python src/analysis/fracture_analysis.py
```

**Funcionalidades en desarrollo:**
- Detección de bordes de fracturas
- Análisis de patrones geométricos
- Clasificación de tipos de fractura

## Flujo de Trabajo Recomendado

1. **Preparación de Datos**
   ```bash
   python scripts/convert_pdfs.py
   ```

2. **Preprocesamiento**
   ```bash
   python src/denoising/main.py
   ```

3. **Análisis Exploratorio**
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```

4. **Análisis Específico**
   ```python
   from src.analysis import fracture_analysis
   ```

## Estructura de Datos

### Formato de Entrada
- **PDFs**: Documentos escaneados en `data/raw/pdfs/`
- **Imágenes**: PNG/JPG en `data/processed/images/`

### Formato de Salida
- Imágenes procesadas en `data/processed/`
- Resultados de análisis en `reports/figures/`
- Métricas y datos en `reports/`

## Desarrollo

### Agregar Nuevas Funcionalidades
1. Crear módulo en `src/` correspondiente
2. Agregar tests en `tests/`
3. Actualizar `requirements.txt` si es necesario
4. Documentar en `docs/`

### Ejecutar Tests
```bash
python -m pytest tests/
```

### Estructurar Nuevos Módulos
```python
# Ejemplo de estructura de módulo
nuevo_modulo/
├── __init__.py
├── main.py
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
pip freeze > requirements.txt
```

### Generar Documentación
```bash
# Generar reportes (futuro)
cd reports/
pdflatex main.tex
```

## Troubleshooting

### Problemas Comunes

**Error de importación:**
```bash
# Asegurarse de estar en directorio raíz
pip install -e .
```

**PDFs no se convierten:**
- Verificar que PyMuPDF esté instalado
- Revisar permisos de archivos

**Imágenes no se cargan:**
- Verificar rutas en `src/utils/load_images.py`
- Confirmar formato de archivos

## Contacto y Soporte

- **Autor**: Hazel Shamed
- **Repositorio**: [github.com/HazelS002/Analisis-Fractura-Roca](https://github.com/HazelS002/Analisis-Fractura-Roca)
- **Institución**: Universidad de Guanajuato

---

*Este proyecto está en desarrollo activo. La estructura y funcionalidades pueden cambiar.*