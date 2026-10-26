# Generative AI and LLM Development - Week 2

This folder contains three deliverables based on the Week 2 assignment PDF and the mini-project brief shown in the screenshot:

1. **Assignment 2 - Language Model Training**: written explanation, training notebook and script, saved model, train/validation losses, and before/after samples.
2. **Mini Project 1 - Character-Level Text Generator**: starter-character generator with notebook, saved model, loss results, and ten generated samples.
3. **Mini Project 2 - RAG Knowledge Base**: Streamlit PDF Q&A app using LangChain, Chroma, source citations, and short conversational history.

The PDF names a character-level text generator as its mini project. The screenshot describes a separate PDF knowledge-base RAG project, so both are included here.

The two character-model projects use Python's standard library and run with `python train.py` from their respective folders. The RAG app requires Python 3.10 or newer, the packages in its `requirements.txt`, and a Gemini API key configured as `GOOGLE_API_KEY`. See its README for setup and run instructions.
