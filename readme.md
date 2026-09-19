# 🎓 University Knowledge Base Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about university policies, fees, scholarships, examinations, admissions, and the academic calendar.

The assistant retrieves relevant information from a curated collection of official university PDF documents and uses **Groq's `openai/gpt-oss-120b`** model to generate grounded answers with document, section, and page references.

Built with **Streamlit, LangChain, FAISS, Sentence Transformers, pdfplumber, and Groq**.

---

## ✨ Features

* 📄 **Multi-document knowledge base** — works with six university policy and academic PDFs.
* 📊 **Table-aware PDF extraction** — uses `pdfplumber` to preserve fee structures, grade tables, schedules, and other tabular information.
* 🧠 **Section-aware chunking** — splits documents using numbered headings such as `4.2 Hostel and Accommodation` instead of blindly splitting text at fixed intervals.
* 🔍 **Local embeddings** — uses `sentence-transformers/all-MiniLM-L6-v2`, so no API key is required for embeddings.
* ⚡ **FAISS vector search** — provides fast semantic retrieval over the university knowledge base.
* 🤖 **Groq LLM** — uses `openai/gpt-oss-120b` to generate natural-language responses.
* 🔐 **User-provided API key** — the Groq API key is entered at runtime and is not hardcoded into the application.
* 📚 **Source citations** — answers include the source document, section, and page number where available.
* 🎨 **Streamlit chat interface** — simple and clean UI for interacting with the knowledge base.
* 🛡️ **Grounded responses** — the model is instructed to answer using retrieved university documents and avoid inventing unsupported information.

---

## 🏗️ Architecture

The application follows a standard RAG pipeline:

```text
                    OFFLINE INGESTION
┌───────────────────────────────┐
│      University PDF Files     │
│                               │
│  • Student Handbook           │
│  • Scholarship Policy         │
│  • Fee Policy                 │
│  • Examination Rules          │
│  • Admission Guidelines       │
│  • Academic Calendar          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│          ingest.py            │
│                               │
│  1. Extract PDF text          │
│  2. Extract tables            │
│  3. Detect sections           │
│  4. Create chunks             │
│  5. Generate embeddings       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        FAISS Vector Store     │
│                               │
│  index.faiss                  │
│  index.pkl                    │
└───────────────┬───────────────┘
                │
                │
              ONLINE
                │
                ▼
┌───────────────────────────────┐
│       User Question           │
│       Streamlit UI            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│           app.py              │
│                               │
│  1. Embed question            │
│  2. Retrieve top-k chunks     │
│  3. Build grounded context    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       Groq LLM                │
│   openai/gpt-oss-120b         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Grounded Answer          │
│                               │
│  Answer + Source Citations    │
└───────────────────────────────┘
```

### RAG Flow

```text
PDFs
  ↓
PDF Extraction
  ↓
Section-aware Chunking
  ↓
Local Embeddings
  ↓
FAISS Vector Store
  ↓
User Question
  ↓
Semantic Retrieval
  ↓
Top Relevant Chunks
  ↓
Groq LLM
  ↓
Grounded Response + Sources
```

---

## 📁 Project Structure

```text
university_knowledge_base/
│
├── knowledge_base/
│   ├── Student handbook.pdf
│   ├── Scholarship policy.pdf
│   ├── Fee policy.pdf
│   ├── Examination rules.pdf
│   ├── Admission guidelines.pdf
│   └── Academic calender.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── ingest.py
├── requirements.txt
└── README.md
```

> `faiss_index/` is generated automatically by `ingest.py`.

> A local `venv/` directory can be created for development but should not be committed to GitHub.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python **3.10+**
* Git
* Internet access for the initial embedding-model download
* A Groq API key

You can create a Groq API key from:

[Groq API Keys](https://console.groq.com/keys?utm_source=chatgpt.com)

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd university_knowledge_base
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Prepare the Knowledge Base

Place the six university PDFs inside:

```text
knowledge_base/
```

The expected documents are:

```text
knowledge_base/
├── Student handbook.pdf
├── Scholarship policy.pdf
├── Fee policy.pdf
├── Examination rules.pdf
├── Admission guidelines.pdf
└── Academic calender.pdf
```

The application uses these documents as its primary knowledge source.

---

## 5. Build the FAISS Index

Run the ingestion pipeline:

```bash
python ingest.py
```

The ingestion process will:

1. Read the university PDFs.
2. Extract text using `pdfplumber`.
3. Extract tables where available.
4. Preserve important document structure.
5. Detect numbered sections and headings.
6. Create meaningful document chunks.
7. Generate local embeddings using `all-MiniLM-L6-v2`.
8. Store the resulting vectors in FAISS.
9. Save the index and metadata to:

```text
faiss_index/
├── index.faiss
└── index.pkl
```

### First Run

The first execution downloads the:

```text
sentence-transformers/all-MiniLM-L6-v2
```

embedding model.

The model is approximately **90 MB**, so an internet connection is required during the initial download.

After that, embeddings can be generated locally.

---

## 6. Start the Application

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Enter your Groq API key in the sidebar and start asking questions.

---

# 📦 Requirements

The project uses the following Python packages:

```text
streamlit
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
langchain-classic
faiss-cpu
pdfplumber
sentence-transformers
```

### Why `langchain-classic`?

Recent LangChain versions moved several legacy chain implementations into the `langchain-classic` package.

For example, if the application uses:

```python
from langchain_classic.chains import create_retrieval_chain
```

then `langchain-classic` must be installed.

---

# 💬 Example Questions

Once the application is running, try questions such as:

### 📌 Policy Questions

```text
What is the attendance policy?
```

```text
What is the minimum GPA required to avoid academic probation?
```

### 💰 Fees and Tables

```text
Show me the tuition fee structure.
```

```text
What are the fees for hostel accommodation?
```

### 📅 Academic Calendar

```text
What are the important dates in the academic calendar?
```

```text
When do final examinations begin?
```

### 🎓 Scholarships

```text
What are the eligibility requirements for scholarships?
```

```text
Can a student with a low GPA apply for financial assistance?
```

### 🔄 Multi-Document Questions

```text
I missed my midterm because I was sick. What are the relevant examination rules and could there be any fee involved?
```

```text
What scholarship options are available and what academic requirements do they have?
```

### 🚫 Out-of-Scope Questions

Try asking:

```text
What is the Wi-Fi password?
```

or:

```text
Who won the World Cup?
```

For information that does not exist in the knowledge base, the assistant should indicate that the information is not available rather than inventing an answer.

---

# 🧠 How the RAG Pipeline Works

The application separates **knowledge retrieval** from **answer generation**.

### Step 1 — Document Ingestion

The PDFs are processed by `ingest.py`.

```text
PDF
 ↓
Text + Tables
 ↓
Sections
 ↓
Chunks
```

### Step 2 — Embedding Generation

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text.

### Step 3 — Vector Storage

The embeddings are stored in a FAISS index.

This allows the application to find chunks that are semantically similar to a user's question.

### Step 4 — Query Retrieval

When a user asks a question:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top Relevant Chunks
```

The application retrieves the most relevant sections from the university documents.

### Step 5 — Grounded Generation

The retrieved context is passed to:

```text
Groq
openai/gpt-oss-120b
```

The model then generates an answer using the retrieved university content.

```text
Retrieved Context
       +
User Question
       ↓
     LLM
       ↓
Grounded Answer
       +
Source References
```

---

# 📚 Why Section-Aware Chunking?

Traditional RAG systems often split documents using a fixed character or token size.

For policy documents, this can be problematic.

For example:

```text
4.2 Hostel and Accommodation

Students residing in university accommodation
must follow the following rules...
```

A fixed-size splitter could divide the heading from its associated policy.

This project instead attempts to preserve the document's natural structure:

```text
Section Heading
      ↓
Section Content
      ↓
Subsection
      ↓
Related Tables
```

This helps keep policy clauses and their context together during retrieval.

---

# 📊 Table-Aware PDF Extraction

University documents frequently contain important information in tables:

* Tuition fees
* Scholarship criteria
* Grade boundaries
* Examination schedules
* Academic dates
* Admission requirements

The ingestion pipeline uses `pdfplumber` to extract these structures where possible.

For example:

```text
| Program | Tuition Fee |
|---------|-------------|
| BSCS    | Rs. XXXXX   |
| BSIT    | Rs. XXXXX   |
```

This information can then become searchable through the RAG pipeline.

---

# 🔐 Security Notes

### API Key

The Groq API key is:

* Entered by the user at runtime.
* Not hardcoded into the source code.
* Not included in the repository.
* Not required for the local embedding model.

### FAISS Deserialization

If the application loads the FAISS index using:

```python
allow_dangerous_deserialization=True
```

the index should only be loaded when it comes from a trusted source.

Do **not** blindly load FAISS index files obtained from unknown or untrusted sources.

---

# ☁️ Deployment — Streamlit Cloud

The application can be deployed using Streamlit Community Cloud.

[Streamlit Community Cloud](https://share.streamlit.io/?utm_source=chatgpt.com)

## 1. Push the Project to GitHub

Your repository should contain:

```text
app.py
ingest.py
requirements.txt
README.md
knowledge_base/
faiss_index/
.streamlit/
```

If you want the cloud application to start immediately without rebuilding the index, include the pre-built:

```text
faiss_index/
```

directory.

## 2. Create the Streamlit App

On Streamlit Community Cloud:

1. Create a new app.
2. Select your GitHub repository.
3. Select the appropriate branch.
4. Set the main file to:

```text
app.py
```

5. Deploy the application.

## 3. API Key

This implementation allows users to enter their own Groq API key through the application sidebar.

Therefore, a repository-level Groq secret is not required for this particular setup.

> For a public production application, consider using server-side secrets instead of asking every user for an API key.

---

# 🛠️ Troubleshooting

| Problem                                                   | Solution                                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'langchain.chains'` | Install `langchain-classic` and update legacy imports to the corresponding `langchain_classic` modules. |
| `FAISS index not found`                                   | Run `python ingest.py` before starting the application.                                                 |
| First run is slow                                         | The Sentence Transformers embedding model is being downloaded.                                          |
| PDF content is missing                                    | Check whether the PDF contains selectable text. Scanned/image-only PDFs may require OCR.                |
| Answers are not relevant                                  | Check retrieval settings such as `k` and ensure the PDFs contain the requested information.             |
| Answers contain unsupported information                   | Strengthen the application's grounding prompt and reduce the retrieval context to relevant chunks.      |
| Streamlit file-watcher warnings appear                    | Add the `.streamlit/config.toml` configuration described below.                                         |
| `Permission denied` when deleting `venv`                  | Close VS Code, terminals, or Python processes using the environment and retry.                          |

---

# ⚙️ Streamlit Configuration

If Streamlit's file watcher produces unnecessary warnings, create:

```text
.streamlit/config.toml
```

with:

```toml
[server]
fileWatcherType = "none"

[global]
developmentMode = false
```

On Windows PowerShell, you can create it with:

```powershell
New-Item -ItemType Directory -Force -Path .streamlit | Out-Null

@'
[server]
fileWatcherType = "none"

[global]
developmentMode = false
'@ | Out-File -FilePath .streamlit\config.toml -Encoding utf8
```

---

# 🔄 Re-ingesting the Knowledge Base

Whenever you modify, replace, or add PDFs inside:

```text
knowledge_base/
```

rebuild the vector index:

```bash
python ingest.py
```

This generates a fresh:

```text
faiss_index/
├── index.faiss
└── index.pkl
```

If the application is deployed using a committed FAISS index, push the updated index to the repository after re-ingestion.

---

# 🔮 Possible Future Improvements

The current version focuses on reliable document-grounded question answering.

Potential future improvements include:

* 🔎 Hybrid keyword + semantic search
* 🧠 Cross-encoder reranking
* 📑 Better OCR support for scanned PDFs
* 🗂️ Metadata filtering by document type
* 📌 More precise citation extraction
* 💬 Conversation memory
* 🧪 Retrieval evaluation using a question-answer benchmark
* 📈 Retrieval and response-quality monitoring
* 🔄 Automatic document re-indexing
* 🌐 Live university website ingestion
* 👤 Student profile-based scholarship matching
* 🔐 Authentication and role-based access
* 📊 Admin dashboard for knowledge-base management

---

# 📜 License

This project is provided for **educational purposes**.

The university documents used as the knowledge base may be copyrighted and remain the property of their respective institutions. Do not redistribute or publicly expose university documents without appropriate permission.

---

# 🙌 Acknowledgements

This project was built using the following open-source technologies:

* **LangChain** — RAG orchestration and document processing
* **FAISS** — vector similarity search
* **Groq** — fast LLM inference
* **Streamlit** — interactive web application framework
* **pdfplumber** — PDF text and table extraction
* **Sentence Transformers** — local semantic embeddings

---

## ⭐ Project Goal

The goal of this project is to demonstrate how **RAG can turn static university policy documents into an interactive knowledge assistant**.

Instead of manually searching through multiple PDFs, students can ask natural-language questions and receive answers grounded in the university's documented policies and academic information.


App link: https://university-knowledge-base.streamlit.app/