# Financial Analysis Service

Servicio de análisis de series temporales financieras con un módulo RAG para consultar documentación en lenguaje natural.

## Sobre este proyecto

Trabajo de la asignatura **Análisis de Series Temporales** del **Máster en Big Data e Inteligencia Artificial** de la **Universitat Politècnica de València (UPV)**, realizado por **Adrián Sánchez** y **Juanfran Tomás**.

El repositorio combina dos piezas:

- Una API REST (`app/main.py`) que expone endpoints de análisis técnico y un sistema RAG sobre PDFs de documentación financiera.
- Una colección de notebooks (`notebooks/`) que recorren, de forma práctica, los conceptos de la asignatura: carga y manipulación de series temporales, gráficos, rentabilidades, volatilidad, medias móviles, Value at Risk y forecasting con SARIMAX.

## ¿Qué hace `app/main.py`?

`app/main.py` es el punto de entrada del servicio. Al arrancar:

1. Carga las variables de entorno desde `app/.env` (`load_dotenv`).
2. Inicializa una instancia **en memoria** de Qdrant (`QdrantClient(location=":memory:")`) con una colección `fin-docs` de dimensión 1536 (compatible con `text-embedding-3-small`).
3. Configura el *retriever* con `QdrantVectorStore` + `OpenAIEmbeddings` y un LLM `gpt-4o-mini` de OpenAI.
4. Define un prompt que obliga al modelo a responder en español, conciso y usando solo el contexto recuperado (con cláusula "si no sabes, di que no sabes").

### Endpoints expuestos

| Método | Ruta                  | Descripción |
|--------|-----------------------|-------------|
| `GET`  | `/health`             | Healthcheck simple. Devuelve `{"status": "ok"}`. |
| `POST` | `/technical-analysis` | Recibe `symbol` (ej. `aapl`, `msft`, `tsla`, `goog`). Calcula el **MACD** (EMA12 − EMA26) y la línea de señal (EMA9), genera un gráfico PNG con `matplotlib` y lo devuelve como `image/png`. El fichero se persiste en `./outputs/<symbol>macd<timestamp>.png`. |
| `POST` | `/index_data`         | Recorre los PDFs de `app/data/`, los trocea con `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` y los indexa en la colección Qdrant. Devuelve páginas cargadas, chunks indexados y los IDs de muestra. |
| `POST` | `/query`              | Recupera los **top-5** documentos más similares a la pregunta (`similarity_search_with_score`), construye el prompt con ese contexto, lo envía a `gpt-4o-mini` y devuelve la respuesta junto con las fuentes y sus scores. |

### Flujo RAG

```
PDFs en app/data/  ──►  PyPDFLoader  ──►  RecursiveCharacterTextSplitter
                                                       │
                                                       ▼
                                          OpenAIEmbeddings (1536-d)
                                                       │
                                                       ▼
                                         Qdrant (in-memory collection "fin-docs")
                                                       │
            pregunta del usuario ──► similarity_search (k=5)
                                                       │
                                                       ▼
                              ChatPromptTemplate  ──►  ChatOpenAI (gpt-4o-mini)
                                                       │
                                                       ▼
                                                respuesta + fuentes
```

## Cómo usar este repositorio

Leyendo esta guía deberíamos ser capaces de tener la versión 0.1.0 de la API lista para correr, con sus notebooks y dependencias instaladas.

---

## Qué vamos a instalar

| Herramienta | Para qué sirve |
|---|---|
| **Git** | Control de versiones — imprescindible en cualquier proyecto de software |
| **uv** | Gestiona Python y las dependencias del proyecto sin necesitar permisos de administrador |
| **Python 3.12** | Lo instala uv automáticamente, solo para este proyecto |

---

## 1. Instalar Git

### Windows (sin permisos de administrador)

1. Entra en https://git-scm.com/download/win
2. Descarga **"64-bit Git for Windows Portable"**
3. Ejecuta el `.exe` descargado — no pide permisos de administrador
4. Al terminar, abre el **Git Bash** que viene incluido y úsalo para todos los comandos de esta guía

Comprueba que funciona:
```bash
git --version
```

### Linux

La mayoría de máquinas Linux ya tienen Git instalado. Compruébalo primero:
```bash
git --version
```

Si aparece una versión, ya está listo. Si no, prueba las opciones siguientes en orden.

**Opción A — Sistema de módulos** (habitual en máquinas universitarias y servidores compartidos)
```bash
module load git
git --version
```
Si funciona, añádelo a tu `.bashrc` para no tener que ejecutarlo cada vez:
```bash
echo "module load git" >> ~/.bashrc
```

---

## 2. Instalar uv

`uv` se instala en tu carpeta de usuario — no necesita permisos de administrador.

### Windows

Abre **PowerShell** (no hace falta que sea como administrador) y ejecuta:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cierra PowerShell y ábrelo de nuevo para que el PATH se actualice. Luego comprueba:
```powershell
uv --version
```

> **¿Te dice que el comando no se encuentra?** Ejecuta esto para añadir uv al PATH de forma permanente:
> ```powershell
> [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:USERPROFILE\.local\bin", "User")
> ```
> Cierra y vuelve a abrir PowerShell.

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cierra y vuelve a abrir la terminal, o ejecuta:
```bash
source ~/.bashrc    # si usas bash
source ~/.zshrc     # si usas zsh
```

Comprueba que funciona:
```bash
uv --version
```

---

## 3. Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd financial-analysis-svc
```

> Si no tienes la URL, el profesor la compartirá en clase.

---

## 4. Instalar Python 3.12 y las dependencias

Dentro de la carpeta del proyecto, ejecuta:

```bash
uv sync
```

Eso es todo. `uv` leerá el archivo `.python-version` del proyecto, instalará Python 3.12 si no lo tienes, creará un entorno virtual aislado e instalará todas las dependencias. No toca tu Python del sistema.

Comprueba qué versión de Python está usando el proyecto:
```bash
uv run python --version
```

Deberías ver `Python 3.12.x`.

---

## 5. Arrancar el servicio

```bash
uv run uvicorn app.main:app --reload
```

Si todo va bien, verás algo así:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

---

## 6. Comprobar que funciona

Abre un navegador y entra en:

```
http://localhost:8000/health
```

Deberías ver:

```json
{"status": "ok"}
```

También tienes la documentación interactiva automática en:

```
http://localhost:8000/docs
```

---

## Estructura del proyecto

```
financial-analysis-svc/
├── app/
│   ├── __init__.py
│   └── main.py          # punto de entrada de la API
├── .python-version      # fija la versión de Python para este proyecto
├── pyproject.toml       # dependencias del proyecto
└── README.md
```

---

## Solución de problemas frecuentes

**"uv: command not found" después de instalarlo**
La terminal no ha cargado el nuevo PATH. Ciérrala y ábrela de nuevo.

**El puerto 8000 ya está en uso**
```bash
uv run uvicorn app.main:app --reload --port 8001
```
Y accede a `http://localhost:8001/health`.

**En Windows, PowerShell dice "la ejecución de scripts está deshabilitada"**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego vuelve a ejecutar el comando de instalación de uv.

---

## Para parar el servidor

Pulsa `Ctrl + C` en la terminal donde está corriendo.
