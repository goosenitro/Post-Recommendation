import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

posts = []

with open('data.json', 'r') as f:
    posts = json.load(f)

user_liked_posts = [1, 3]
user_search_terms = ["luxury cars", "boat for sale", "car 2023"]

df = pd.DataFrame(posts)
df['text'] = df['title'] + ' ' + df['description']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])
cosine_sim_posts = cosine_similarity(tfidf_matrix, tfidf_matrix)

search_terms = user_search_terms
search_tfidf = vectorizer.transform(search_terms)
cosine_sim_search = cosine_similarity(search_tfidf, tfidf_matrix)

def hybrid_recommendation(user_likes, cosine_sim_posts, cosine_sim_search, df, n=3):
    sim_scores = {}
    
    liked_post_indices = df.index[df['post_id'].isin(user_likes)].tolist()

    for idx in liked_post_indices:
        similar_posts = list(enumerate(cosine_sim_posts[idx]))
        for post_idx, score in similar_posts:
            if df.iloc[post_idx]['post_id'] not in user_likes:
                if df.iloc[post_idx]['post_id'] not in sim_scores:
                    sim_scores[df.iloc[post_idx]['post_id']] = score
                else:
                    sim_scores[df.iloc[post_idx]['post_id']] += score

    for idx, term_sim in enumerate(cosine_sim_search):
        for post_idx, score in enumerate(term_sim):
            if df.iloc[post_idx]['post_id'] not in user_likes:
                if df.iloc[post_idx]['post_id'] not in sim_scores:
                    sim_scores[df.iloc[post_idx]['post_id']] = score
                else:
                    sim_scores[df.iloc[post_idx]['post_id']] += score

    recommended_posts = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_recommendations = [post[0] for post in recommended_posts[:n]]

    return top_recommendations

recommended_posts = hybrid_recommendation(user_liked_posts, cosine_sim_posts, cosine_sim_search, df, n=3)

print("Recommended Posts for the User:")
for post_id in recommended_posts:
    post = df[df['post_id'] == post_id].iloc[0]
    print(f"Post ID: {post_id}, Title: {post['title']}")