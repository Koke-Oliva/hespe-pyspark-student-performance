# HESPE Student Performance — PySpark ML Pipeline

Proyecto de **Machine Learning distribuido con PySpark / Spark ML** para predecir el rendimiento académico final de estudiantes mediante un pipeline reproducible de clasificación multiclase.

> **Proyecto formativo de portafolio.** El dataset tiene 145 observaciones; Spark se utiliza para demostrar arquitectura y prácticas escalables, no para afirmar mejoras de rendimiento por volumen.

## Qué demuestra

- carga de datos con **schema explícito**;
- auditoría de nulos, duplicados y dominios;
- EDA ejecutado en Spark y visualización solo de agregados pequeños;
- tratamiento diferenciado de variables ordinales y nominales;
- split train/test **estratificado**;
- `Pipeline` completo de Spark ML;
- baseline con **Logistic Regression multinomial**;
- **Random Forest** con `CrossValidator` y tuning;
- Accuracy, Weighted F1, Macro F1 y métricas por clase;
- matriz de confusión y falsos positivos/negativos por categoría;
- métricas ordinales (`MAE`, exactitud ±1 y ±2 categorías);
- feature importance agregada a las variables originales;
- ejecución reproducible mediante GitHub Actions.

## Correcciones metodológicas respecto de la entrega original

La entrega académica original trataba `grade` como una regresión ordinal aproximada. Para la versión profesionalizada:

1. `grade` se modela como **clasificación multiclase**, coherente con la definición oficial de UCI y con la rúbrica de evaluación.
2. `course_id` ya no se describe como un identificador sin poder predictivo: UCI lo define como feature. Se excluye del modelo principal por riesgo de dependencia específica de curso y se documenta como limitación de generalización.
3. Se elimina el escalado innecesario del pipeline de árboles. En particular, se evita `StandardScaler(withMean=True)` sobre OHE, que puede densificar vectores sparse y es una mala práctica para un escenario Big Data.
4. Se añaden EDA real, matriz de confusión, métricas por clase, mejores hiperparámetros y feature importance interpretable.
5. Se usa un split estratificado para asegurar representación de las ocho clases en test.

## Dataset

**Higher Education Students Performance Evaluation** — UCI Machine Learning Repository.

- 145 instancias
- 31 features
- tarea oficial: Classification
- target: 8 categorías de nota final
- DOI: https://doi.org/10.24432/C51G82
- licencia: CC BY 4.0

Más detalles: [`data/README.md`](data/README.md).

## Pipeline

```text
CSV + schema explícito
        ↓
Data quality audit
        ↓
EDA distribuido
        ↓
Feature typing
  ├─ ordinales
  └─ nominales → StringIndexer → OneHotEncoder
        ↓
Split estratificado
        ↓
Logistic Regression baseline
        ↓
Random Forest + 3-fold CV + tuning
        ↓
Evaluación multiclase + ordinal
        ↓
Confusion matrix + feature importance
```

## Resultados

Las métricas finales se completarán a partir de una **ejecución validada por CI** del notebook profesionalizado. El README no duplica manualmente resultados antes de esa validación para evitar inconsistencias entre código y documentación.

## Reproducibilidad

```bash
git clone https://github.com/Koke-Oliva/hespe-pyspark-student-performance.git
cd hespe-pyspark-student-performance

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter notebook hespe_pyspark_student_performance.ipynb
```

Requiere **Java 17** para la ejecución local de Spark 4.x.

## Estructura

```text
.
├── .github/
│   └── workflows/
│       └── notebook-ci.yml
├── data/
│   ├── hespe-data.csv
│   └── README.md
├── figures/
│   └── README.md
├── hespe_pyspark_student_performance.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitaciones

- 145 observaciones no permiten demostrar ventajas de throughput o escalabilidad distribuida.
- las ocho clases presentan soportes distintos;
- la generalización a cursos/instituciones no observados requiere validación agrupada externa;
- las variables codificadas reflejan categorías discretas, no mediciones continuas;
- feature importance es predictiva, no causal.

## Contexto académico

Proyecto desarrollado a partir de la evaluación final de **Fundamentos de Big Data** del Bootcamp de Ciencia de Datos — IT Academy / Kibernum, Talento Digital para Chile.