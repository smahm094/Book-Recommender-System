#### *GNG5125 Data Science Applications*
#### *Spring-Summer 2022*
#### *Yinruo Jiang (300274815), Rasheeq Mohammad (6849734) , Shahin Mahmud (300274789)*
#### *Final Project: Book Recommender System (BRS)*

---

# Final Project: Book Recommender System (BRS)
The book recommender system (BRS) includes a cosine-similarity-based search engine to look up books, a cosine-similarity-based collaborative filter to recommend books based on user ratings, a clustering-based exploratory data analysis (EDA) approach to explore the datasets used, a classification-based content filter to recommend books based on user descriptions, and a DialogFlow-and-Flask-based frontend. The Goodreads dataset used was scraped from Goodreads users’ public shelves (freely available on the web without login) by a group of researchers from University of California San Diego (UCSD) in 2017. The Wikipedia dataset used was extracted from Wikipedia articles on different books by a researcher from Carnegie Mellon University (CMU). 

## Files

* app.py (Flask-based web application; fulfills DialogFlow requests)
* clustering.ipynb (clustering algorithm; generates insights for Goodreads dataset)
* collaborative_filtering.ipynb (collaborative filter; produces recommendations based on other users with similar likes)
* content_filter_classifiers.ipynb (content filter; multi-label classification using standard classifiers)
* content_filter_lstm_innovation.ipynb (content filter; multi-label classification using LSTM)
* dataset1_prep.ipynb (Goodreads dataset cleaning and preparation; saves book metadata table in a JSON object)
* dataset2_prep.ipynb (Wikipedia dataset cleaning and preparation; saves book metadata table in a CSV file)
* search.ipynb (search engine; yields book metadata)

## Requirements

Software:
* Python 3.9.12 (https://www.python.org/)
* Google Colab (https://research.google.com/colaboratory/)
* LocalToNet (https://localtonet.com/)
* DialogFlow (https://cloud.google.com/dialogflow)

Python libraries:
* collections
* contractions
* nltk
* pandas
* random
* re
* seaborn
* sklearn
* unicodedata
* wordcloud
* flask
* gzip
* matplotlib
* json
* time
* numpy
* scipy

Datasets:
* Goodreads: Download from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
    * 1. good_reads_interactions.csv (4.02 GB)
    * 2. goodreads_books.json.gz (1.93 GB)
    * 3. book_id_map.csv (36 MB)
* Wikipedia: Download from https://www.cs.cmu.edu/~dbamman/booksummaries.html
    * 1. booksummaries.txt (41.4 MB)

## Usage

To reproduce this code, you will need to have the Python libraries mentioned above as well as download the datasets mentioned above. We executed the provided code beforehand so you do not even have to run the code if you want to just observe the outputs.

You can run the code with your choice of IDE.
