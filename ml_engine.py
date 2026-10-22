import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

class ResumeJobEngine:
    def __init__(self, resumes, jobs):
        self.resumes = resumes
        self.jobs = jobs
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        all_text = list(resumes) + list(jobs)
        self.vectorizer.fit(all_text)
        self.resume_vectors = self.vectorizer.transform(resumes)
        self.job_vectors = self.vectorizer.transform(jobs)

    def choose_k(self, max_k=8):
        n = self.resume_vectors.shape[0]
        candidates = range(2, min(max_k, n-1)+1)
        best_k, best_score = 2, -1
        scores = {}
        dense = self.resume_vectors.toarray()
        for k in candidates:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(dense)
            if len(set(labels)) > 1:
                score = silhouette_score(dense, labels)
                scores[k] = score
                if score > best_score:
                    best_k, best_score = k, score
        return best_k, scores

    def cluster_resumes(self, k):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(self.resume_vectors.toarray())
        return model, labels

    def recommend(self, resume_text, top_n=5):
        vec = self.vectorizer.transform([resume_text])
        scores = cosine_similarity(vec, self.job_vectors).flatten()
        idx = np.argsort(scores)[::-1][:top_n]
        return [(self.jobs.iloc[i], float(scores[i])) for i in idx]

    def project_2d(self):
        if self.resume_vectors.shape[1] < 2:
            return np.zeros((self.resume_vectors.shape[0], 2))
        return TruncatedSVD(n_components=2, random_state=42).fit_transform(self.resume_vectors)
