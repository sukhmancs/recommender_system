<div id="header" align="center">
  <h1>
    🚀 Recommender System
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2g0M2owejB2MHAxbnluN21sZnp3eG1taGNyYXc5dTc0OHA1Y2FqcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/lOgu1OnjYF2GHBfRU4/giphy.gif" width="40px"/>
  </h1>
</div>

This repository contains an implementation of a text document recommender system using Python. The system recommends similar documents based on vector representations and similarity calculations.
Overview

Recommender systems such as this are a core application of statistical AI. At the heart of recommender systems is a similarity calculation. In this implementation, we use vector representations of documents and a document similarity calculation to recommend relevant articles to users.

# Features

- Load text documents from various datasets (e.g., BBC news articles, scientific abstracts, Wikipedia articles).
- Preprocess text data to remove stopwords and perform lemmatization.
- Vectorize documents using TF-IDF vectorization.
- Calculate document similarity using cosine similarity.
- Generate recommendations based on the most similar documents to a selected document, while also including some less similar documents to provide diversity.
- Avoid recommending the same document or documents with the same title as the selected document.

# Usage

Clone the repository:
```bash

git clone https://github.com/your_username/text-document-recommender.git
```

Run the recommender system:
```bash
python recommender.py
```

# Dataset
The datasets used in this project include:

- BBC news articles
- Scientific abstracts
- Wikipedia articles

These datasets have been adapted for this task.

# License

This project is licensed under the MIT License. See the [LICENSE file](./LICENSE) for details.
