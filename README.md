# RC-System: Hybrid Recommender System

This project builds a hybrid recommender system using **content-based filtering** and **collaborative filtering (CF)**. It combines textual movie metadata with user rating embeddings to provide movie recommendations.

## Main Components

### 1. **Data Loading and Preprocessing**

* Loads the following datasets from the MovieLens dataset:

  * `movies_metadata.csv`
  * `credits.csv`
  * `keywords.csv`
  * `ratings.csv`
* Merges metadata, parses relevant fields (`cast`, `crew`, `genres`, `keywords`), and creates a **combined text feature** for each movie.

### 2. **Content-Based Filtering**

* Uses `CountVectorizer` to convert movie features (genre, cast, keywords, director) into a vector form.
* Computes **cosine similarity** between movie vectors.
* Allows recommendations based on textual/movie metadata.

### 3. **Collaborative Filtering (CF)**

* Trains a **neural network** using TensorFlow/Keras to learn user and movie embeddings from rating data.
* Uses embedding layers for users and movies.
* Predicts ratings using dense layers and minimizes MSE loss.

### 4. **Hybrid Embedding Fusion**

* Identifies movies with both metadata and rating data.
* Combines:

  * Content-based vectors (from `CountVectorizer`)
  * CF-based learned embeddings (from the trained model)
* Concatenates both into a **hybrid feature vector** for each movie.

### 5. **Hybrid Recommendation Function**

```python
get_hybrid_recommendations(title, top_n=10)
```

* Takes a movie title.
* Computes similarity based on hybrid vectors.
* Returns top `n` similar movies as recommendations.

## How to Run

1. Download all datasets from the [MovieLens dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
2. Place files in the appropriate input folder or adjust the file paths in the notebook.
3. Open and run `rc-system.ipynb` cell by cell.

## Dependencies

* Python 3.x
* pandas, numpy, scikit-learn
* TensorFlow/Keras
* Jupyter Notebook


