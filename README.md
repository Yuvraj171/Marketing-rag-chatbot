# Marketing RAG Chatbot

A query-based Retrieval-Augmented Generation (RAG) chatbot prototype designed to help Sales and Marketing teams answer customer queries using internal manufacturing documents and LLM-generated explanations.

---

## Overview

In many manufacturing companies, Sales and Marketing teams often receive customer questions that require technical product knowledge.  
This project aims to build a simple but scalable chatbot that can:

- accept user questions in chatbot format
- search internal company documents
- retrieve relevant information using RAG
- use an LLM to generate a simple, helpful final response
- explain answers in layman-friendly language

This prototype is being designed in a way that can later scale into a larger internal application with better UI, filters, citations, analytics, and admin capabilities.

---

## Business Context

The chatbot is intended for internal use in a manufacturing company with four major business units:

- Unit-1
- Unit-3
- Unit-5
- Bestec

Each unit has its own internal documents such as:

- product catalogues
- technical catalogues
- machine information documents
- supporting PDFs and docs

These documents will form the internal knowledge base of the chatbot.

---

## Why RAG is Suitable Here

RAG (Retrieval-Augmented Generation) is a good fit for this use case because:

- the chatbot needs to answer from internal documents
- product information may differ across business units
- internal knowledge can change over time
- the system should remain grounded in company documents
- the final answer should still be easy to understand for non-technical users

Instead of training a model from scratch, RAG allows us to:

1. retrieve the most relevant document chunks
2. pass those chunks to the LLM
3. generate a grounded response based on retrieved knowledge

---

## Planned V1 Features

- chat-based question answering
- PDF and document retrieval
- internal document-based answers
- LLM-generated simplified explanations
- basic source awareness
- simple local deployment
- beginner-friendly codebase

---

## Planned Tech Stack

- **Python**
- **LangChain**
- **Google Gemini API**
- **FAISS or ChromaDB**
- **Streamlit**
- **PyPDF**
- **python-dotenv**

---

## Project Structure

```text
Marketing-rag-chatbot/
│
├── knowledge_base/
│   ├── bestec/
│   ├── unit1/
│   ├── unit3/
│   └── unit5/
│
├── src/
│   ├── ingestion/
│   │   └── ingest_documents.py
│   │
│   ├── llm/
│   │   └── gemini_client.py
│   │
│   ├── prompts/
│   │   └── system_prompt.txt
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   └── utils/
│       └── helpers.py
│
├── tests/
├── ui/
│   └── app.py
│
├── vector_store/
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt