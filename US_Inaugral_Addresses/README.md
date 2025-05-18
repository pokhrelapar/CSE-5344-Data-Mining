# TF-IDF Search Engine on U.S. Inaugural Addresses

This project implements a lightweight **search engine** over U.S. Presidential Inaugural Addresses using the **TF-IDF** (Term Frequency–Inverse Document Frequency) model. It processes raw text data, builds normalized TF-IDF vectors, and ranks documents based on **cosine similarity** to user queries.

---

## 📘 What is This Project About?

This project:

- Reads all `.txt` files from a folder of inaugural speeches.
- Tokenizes and preprocesses the text (lowercase, remove stopwords, stem).
- Calculates term frequency (TF) and inverse document frequency (IDF).
- Builds **TF-IDF vectors** for each document.
- Enables users to input a query string and returns the **most relevant document** based on cosine similarity.
- Shows top-10 documents per query term using **postings lists**.

---

## 📦 Dependencies

- Python 3.12.3
- `nltk` 3.9.1

Install required packages using pip:

```bash
pip install nltk
```
