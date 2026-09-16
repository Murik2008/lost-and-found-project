def describe_item(image_path):
    return "A test item description"

def embed(text):
    return [0.1, 0.2, 0.3, 0.4]

def cosine(vec1, vec2):
    return 0.95

def top_k(query_vec, candidate_vecs, k):
    return list(range(min(k, len(candidate_vecs))))
