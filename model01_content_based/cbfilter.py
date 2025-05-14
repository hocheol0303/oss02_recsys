from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_categories(categories):
    if isinstance(categories, list):
        return ' '.join(categories)
    return categories or ''

def get_recommendations(user, benefits, top_n=10):
    user_vec = preprocess_categories(user['categories'])

    benefit_texts = []
    benefit_info = []

    for benefit in benefits:
        benefit_texts.append(preprocess_categories(benefit['categories']))
        benefit_info.append({
            "id": benefit["id"],  # title은 생략
        })

    corpus = [user_vec] + benefit_texts
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]

    top_benefits = [benefit_info[i] for i in top_indices]
    return top_benefits
