# 🔬 RAG Agent con Hybrid Retriever (RRF) + Gemini

Un agente de recuperación aumentada (RAG) que combina búsqueda semántica vectorial y búsqueda por palabras clave (BM25) mediante **Reciprocal Rank Fusion (RRF)**, potenciado por el modelo **Gemini 2.5 Flash** de Google.

---

## 📐 Arquitectura

```
URL (Wikipedia)
     │
     ▼
WebBaseLoader ──► RecursiveCharacterTextSplitter
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     ChromaDB (Vectores)         BM25Retriever
     HuggingFace Embeddings      (TF-IDF léxico)
              │                       │
              └───────────┬───────────┘
                          ▼
                  HybridRetriever (RRF)
                          │
                          ▼
                    research_tool
                          │
                          ▼
                   Agente LangChain
                  (Gemini 2.5 Flash)
```

---

## 🧩 Componentes principales

### 1. `HybridRetriever` — Fusión RRF
Combina los resultados del retriever vectorial y BM25 aplicando la fórmula de **Reciprocal Rank Fusion**:

```
score(doc) = Σ  1 / (rrf_k + rank_i + 1)
```

Esto permite que documentos bien posicionados en ambos rankings reciban una puntuación mayor, mejorando la relevancia final sin necesidad de un modelo cross-encoder.

### 2. `KnowledgeEngine` — Motor de conocimiento
- Carga y limpia contenido web desde una URL dada.
- Divide el texto en chunks de 1000 caracteres con solapamiento de 200.
- Indexa los chunks en **ChromaDB** con embeddings `all-mpnet-base-v2`.
- Crea un retriever BM25 sobre los mismos chunks.
- Expone un `HybridRetriever` con `k=4` documentos finales.

### 3. `research_tool` — Herramienta del agente
Función decorada con `@tool` que el agente invoca para consultar la base de conocimiento híbrida.

### 4. Agente LangChain
Agente ReAct que utiliza `ChatGoogleGenerativeAI` (Gemini 2.5 Flash) y la herramienta de búsqueda para responder preguntas con evidencia factual.

---

## 🛠️ Instalación

```bash
pip install langchain langchain-community langchain-google-genai \
            langchain-huggingface langchain-chroma \
            sentence-transformers beautifulsoup4 rank_bm25 python-dotenv
```

---

## ⚙️ Configuración

El proyecto requiere una API Key de Google Gemini. Puedes configurarla como variable de entorno:

```bash
export GOOGLE_API_KEY="tu_api_key_aqui"
```

O bien, el script la solicitará de forma interactiva al ejecutarse por primera vez.

---

## 🚀 Uso

```python
# 1. Instanciar el motor apuntando a una URL
engine = KnowledgeEngine("https://es.wikipedia.org/wiki/Toxina")

# 2. El agente usa la herramienta automáticamente
result = agent.invoke({
    "messages": [
        HumanMessage(content="¿Por quién fue introducido el término toxina?")
    ]
})

# 3. Imprimir los mensajes del agente
for message in result["messages"]:
    message.pretty_print()
```

Para apuntar a otra fuente de conocimiento, simplemente cambia la URL al instanciar `KnowledgeEngine`:

```python
engine = KnowledgeEngine("https://es.wikipedia.org/wiki/Penicilina")
```

---

## 📦 Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Gemini 2.5 Flash (`langchain-google-genai`) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Framework | LangChain |
| Web Scraping | BeautifulSoup4 + WebBaseLoader |

---

## 📁 Estructura del proyecto

```
.
├── HybridRank_RAG.ipynb   # Notebook principal con todo el pipeline
└── README.md              # Este archivo
```

---

## 📝 Notas

- El contenido de Wikipedia se trunca en la sección "Véase también" para evitar ruido.
- Se recuperan **10 candidatos** de cada retriever antes de aplicar RRF, devolviendo los **4 mejores**.
- El parámetro `rrf_k=60` es el valor estándar recomendado en la literatura para RRF.
