# Mini Project: PDF Knowledge Base with RAG

Build a small question-answering system over a collection of PDF documents. The app extracts page text, splits it into overlapping chunks, stores Gemini embeddings in a local Chroma database, retrieves relevant passages, and asks Gemini to answer only from those passages. Answers include filename and page citations. A short conversation history supports follow-up questions.

## Requirements

- Windows with Python 3.10 or newer.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- Text-based PDFs. Scanned image-only files need OCR before indexing.

## Run on Windows

1. Open PowerShell in this project folder.
2. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, replace `your_gemini_api_key` with your key, and save. `GOOGLE_API_KEY` is the expected variable. The chat and embedding model names can be changed in `.env` if your account has access to different models.
4. Start the application:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

5. Upload one or more PDFs in the sidebar and choose **Build / refresh knowledge base**. Wait for the indexed page and chunk counts, then ask a question in the chat box. Answers include text citations in the form `[filename.pdf, p. 2]`; expand **Retrieved passages** to inspect the matching text.

## What is included

- `app.py`: PDF ingestion, page-aware chunking, Chroma vector storage, retrieval, conversational Q&A, and source citations.
- `requirements.txt`: Python dependencies.
- `.env.example`: configuration template. Keep your real `.env` and API key private.
- `uploaded_pdfs/` and `chroma_db/`: created locally when the app is used; excluded from version control.

## Notes

This demo uses Gemini for both embeddings and answer generation. Gemini API availability, model access, free-tier quotas, and billing depend on Google's current account and model rules; check the [Gemini API pricing and free-tier details](https://ai.google.dev/gemini-api/docs/pricing) before use. Free-tier availability is limited to eligible models and quotas. The app stores uploaded PDFs and its Chroma database locally in this project folder. Each indexing action replaces the previous set of PDFs and rebuilds the index. Retrieved text is treated as evidence rather than instructions, and the model is asked to acknowledge when the excerpts do not support an answer. Verify important answers against the cited page.
