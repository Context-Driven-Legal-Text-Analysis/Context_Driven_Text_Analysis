# DocuAI - Intelligent Document Assistant

DocuAI is a Flask-based web application that enables users to upload, process, and interact with documents (PDF, DOCX, PPTX, TXT) using natural language. It leverages advanced retrieval-augmented generation (RAG) techniques, semantic search, and large language models (LLMs) to provide accurate, context-aware answers to user queries about their documents. The system supports multilingual queries, voice input, and a modern chat interface.

---

## Features

- **Document Upload:** Supports PDF, DOCX, PPTX, and TXT files.
- **Semantic Chunking:** Splits documents into meaningful chunks using spaCy.
- **Embeddings & Vector Search:** Generates embeddings (DeepInfra/OpenAI API) and stores them in Pinecone for semantic search.
- **Hybrid Retrieval:** Combines vector similarity (Pinecone) and BM25 keyword search, reranked by a cross-encoder for best relevance.
- **LLM-Powered Q&A:** Uses Groq API to rewrite queries and generate grounded, context-aware answers.
- **Multilingual Support:** Detects and translates queries/responses using `langdetect` and `deep-translator`.
- **Voice Input:** Users can ask questions via speech, transcribed and translated as needed.
- **Modern Web UI:** Responsive chat interface with document upload, language selection, and voice controls.

---

## System Architecture

```
User (Web UI)
   │
   ▼
Flask Backend (Python)
   │
   ├─ Document Extraction (PyPDF2, python-docx, python-pptx)
   ├─ Semantic Chunking (spaCy)
   ├─ Embedding Generation (DeepInfra/OpenAI)
   ├─ Vector Storage & Search (Pinecone)
   ├─ Hybrid Retrieval (BM25, CrossEncoder)
   ├─ Query Rewriting & Q&A (Groq API)
   ├─ Multilingual & Voice Support (langdetect, deep-translator, SpeechRecognition)
   ▼
LLM APIs / Pinecone
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/parthjha03/DocuAI.git
   cd docuai
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root with the following keys:
     ```
     MODEL=your_groq_model_name
     GROQ_API_KEY=your_groq_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     INDEX_NAME=your_pinecone_index_name
     DEEPINFRA_API_KEY=your_deepinfra_api_key
     ```

4. **Download spaCy model:**
   ```sh
   python -m spacy download en_core_web_sm
   ```

5. **Run the application:**
   ```sh
   python app.py
   ```

6. **Access the app:**
   - Open your browser and go to `http://localhost:5000`

---

## Usage

- **Upload Documents:** Use the sidebar to upload PDF, DOCX, PPTX, or TXT files.
- **Ask Questions:** Type or speak your question in the chat interface.
- **Language Support:** Select your preferred language from the dropdown.
- **Voice Input:** Click the microphone button to record your question.

---

## Technologies Used

- **Backend:** Flask, Python
- **Frontend:** HTML, CSS, JavaScript
- **NLP & Embeddings:** spaCy, NLTK, DeepInfra/OpenAI, sentence-transformers
- **Vector Database:** Pinecone
- **LLM APIs:** Groq
- **Translation & Language Detection:** deep-translator, langdetect
- **Voice Recognition:** SpeechRecognition, Google Speech API

---

## Project Structure

```
.
├── app2.py                # Main Flask backend
├── utils.py               # Utility functions (e.g., token counting)
├── templates/
│   └── index.html         # Main frontend template
├── uploads/               # Uploaded documents
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (not committed)
```

---

## Example Query Flow

1. User uploads a document.
2. User asks a question (text or voice, any language).
3. System translates and rewrites the query for optimal retrieval.
4. Relevant document chunks are retrieved and reranked.
5. LLM generates a grounded answer using the retrieved context.
6. Answer is translated back to the user's language and displayed.

---

## License

MIT License

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Pinecone](https://www.pinecone.io/)
- [DeepInfra](https://deepinfra.com/)
- [Groq](https://groq.com/)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [sentence-transformers](https://www.sbert.net/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [deep-translator](https://pypi.org/project/deep-translator/)

---

*DocuAI brings the power of LLMs and semantic search to your documents, making them truly interactive and accessible.*
