from flask import Flask, request, abort, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from scipy.sparse import coo_matrix
import json

app = Flask(__name__)

print("Start setup")

book_info_df = pd.read_json('book_info_df.json')
print("Ended loading book info")

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(book_info_df["mod_title"])
print("Ended creating vectorizer")

my_books = pd.read_csv("my_books.csv", index_col=0)
my_books["book_id"] = my_books["book_id"].astype(str)
print("Loaded read books list")

csv_book_mapping = {}
with open("book_id_map.csv", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        csv_id, book_id = line.strip().split(",")
        csv_book_mapping[csv_id] = book_id
print("Loaded book ID mapping")

my_books_set = set(my_books["book_id"])

overlap_users = {}
with open("goodreads_interactions.csv", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        user_id, csv_id, _, rating, _ = line.split(",")

        book_id = csv_book_mapping.get(csv_id)

        if book_id in my_books_set:
            if user_id not in overlap_users:
                overlap_users[user_id] = 1
            else:
                overlap_users[user_id] += 1
print("Found overlap users")

my_num_books = my_books.shape[0]
filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > my_num_books / 5])

interactions_list = []

with open("goodreads_interactions.csv", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        user_id, csv_id, _, rating, _ = line.split(",")

        if user_id in filtered_overlap_users:
            book_id = csv_book_mapping[csv_id]
            interactions_list.append([user_id, book_id, rating])
print("Found filtered overlap users")

interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
interactions = pd.concat([my_books[["user_id", "book_id", "rating"]], interactions])

interactions["book_id"] = interactions["book_id"].astype(str)
interactions["user_id"] = interactions["user_id"].astype(str)

interactions["rating"] = pd.to_numeric(interactions["rating"])

interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
interactions["book_index"] = interactions["book_id"].astype("category").cat.codes

ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))

ratings_mat = ratings_mat_coo.tocsr()
print("Created sparse matrix")

my_index = 0

similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()

indices = np.argpartition(similarity, -15)[-15:]

similar_users = interactions[interactions["user_index"].isin(indices)].copy()

similar_users = similar_users[similar_users["user_id"] != "-1"]

book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])

book_info_df["book_id"] = book_info_df["book_id"].astype(str)
book_recs = book_recs.merge(book_info_df, how="inner", on="book_id")

book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["num_ratings"])
book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]
book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]
my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", " ", regex=True)
book_recs = book_recs[~book_recs["mod_title"].isin(my_books["mod_title"])]
book_recs = book_recs[book_recs["mean"] >= 4]
book_recs = book_recs[book_recs["count"]> 2]

top_recs = book_recs.sort_values("mean", ascending=False)
print("Determined recommendations")

def search(query):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = book_info_df.iloc[indices]
    results = results.sort_values("num_ratings", ascending=False)

    key_info = ["book_id", "title", "num_ratings", "average_rating", "num_pages", "publication_year"]
    counter = 0
    rsp_list = []
    for index, row in results.iterrows():
        counter += 1
        rsp_list.append("Result " + str(counter) +
                        "\nbook_id: " + str(row['book_id']) +
                        "\ntitle: " + row['title'] +
                        "\nnum_ratings: " + str(row['num_ratings']) +
                        "\naverage_rating: " + str(row['average_rating']) +
                        "\nnum_pages: " + str(row['num_pages']) +
                        "\npublication_year: " + str(row['publication_year']) + "\n")
    rsp = ''.join(rsp_list)
    return rsp

@app.route('/webhook', methods=['POST'])
def webhook():
    fulfillmentText = 'Failed'
    if request.method == 'POST':
        req = request.get_json(silent=False, force=True)
        query_result = req.get('queryResult')
        if query_result.get('action') == 'BookSearchAction':
            parameters = query_result.get('parameters')
            book_name = parameters.get('BookNameForSeaEntity')
            res_str = search(book_name)
            fulfillmentText = res_str
        elif query_result.get('action') == 'BookRecAction':
            parameters = query_result.get('parameters')
            books_string = parameters.get('BookNameForRecEntity')
            # In production, will use books_string, but for demo use top_recs
            
            # Use top_recs for demo
            key_info = ["book_id", "mean", "title", "num_ratings", "average_rating", "num_pages", "publication_year"]
            counter = 0
            rsp_list = []
            for index, row in top_recs.iterrows():
                counter += 1
                rsp_list.append("Result " + str(counter) +
                                "\nbook_id: " + str(row['book_id']) +
                                "\nmean: " + str(row['mean']) +
                                "\ntitle: " + row['title'] +
                                "\nnum_ratings: " + str(row['num_ratings']) +
                                "\naverage_rating: " + str(row['average_rating']) +
                                "\nnum_pages: " + str(row['num_pages']) +
                                "\npublication_year: " + str(row['publication_year']) + "\n")
            rsp = ''.join(rsp_list) 
            fulfillmentText = rsp
    else:
        abort(400)

    return {"fulfillmentText": fulfillmentText}

if __name__ == '__main__':
    print("Started application")
    app.debug = False
    app.run()
