"""
Run: python rag_retrieval_experiment.py.
"""
from collections import defaultdict, deque
from statistics import median
from time import perf_counter_ns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Each sentence asserts one relationship: dependent -> dependency.
DOCUMENTS = [
    ('d1', 'payment-service uses redis-x for session storage.', 'payment-service', 'redis-x'),
    ('d2', 'checkout-service calls payment-service to authorize purchases.', 'checkout-service', 'payment-service'),
    ('d3', 'frontend calls checkout-service to submit orders.', 'frontend', 'checkout-service'),
    ('d4', 'notification-service uses postgres-db to store notifications.', 'notification-service', 'postgres-db'),
    ('d5', 'reporting-service uses postgres-db for analytics.', 'reporting-service', 'postgres-db'),
    ('d6', 'billing-worker calls payment-service for invoice collection.', 'billing-worker', 'payment-service'),
    ('d7', 'email-worker uses kafka-broker to receive events.', 'email-worker', 'kafka-broker'),
    ('d8', 'audit-service calls reporting-service to fetch summaries.', 'audit-service', 'reporting-service'),
    ('d9', 'api-gateway calls frontend to route web requests.', 'api-gateway', 'frontend'),
    ('d10', 'search-service uses search-index for product lookup.', 'search-service', 'search-index'),
]

QUESTIONS = [
    ('redis-x', 'If redis-x fails, which services are potentially affected?'),
    ('postgres-db', 'If postgres-db fails, which services are potentially affected?'),
    ('kafka-broker', 'If kafka-broker fails, which services are potentially affected?'),
    ('payment-service', 'If payment-service fails, which services are potentially affected?'),
]

reverse = defaultdict(list)
for docid, text, dependent, dependency in DOCUMENTS:
    reverse[dependency].append(dependent)

def graph_impact(root):
    visited = {root}
    queue = deque([root])
    while queue:
        for dependent in reverse[queue.popleft()]:
            if dependent not in visited:
                visited.add(dependent)
                queue.append(dependent)
    return visited - {root}

# Vector search: rank source sentences by TF-IDF cosine similarity.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
corpus = vectorizer.fit_transform([d[1] for d in DOCUMENTS])

def vector_topk(query, k):
    scores = cosine_similarity(vectorizer.transform([query]), corpus)[0]
    return sorted(range(len(DOCUMENTS)), key=lambda i: (-scores[i], i))[:k]

# Deliberately simple, reproducible answer extraction: a retrieved sentence
# directly identifies its dependent, but we do not infer multi-hop impacts
def vector_answer(query, k):
    return {DOCUMENTS[i][2] for i in vector_topk(query, k)}

def metrics(pred, truth):
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 1.0
    return precision, recall, pred == truth

def time_us(fn, repeats=300):
    for _ in range(10): fn()
    times = []
    for _ in range(repeats):
        t0 = perf_counter_ns()
        fn()
        times.append((perf_counter_ns() - t0) / 1000)
    return median(times)

print('Dataset: %d source sentences; %d impact questions' % (len(DOCUMENTS), len(QUESTIONS)))
print('Vector: TF-IDF unigrams+bigrams, cosine, graph: curated reverse BFS')
print('NOTE: Vector answer uses retrieved direct dependents only; no LLM or multi-hop reasoning')
print()
print(f"{'Root':<17} {'Method':<12} {'P':>6} {'R':>6} {'Exact':>7} {'Median us':>11}  Predicted affected services")
print('-' * 112)
for root, query in QUESTIONS:
    truth = graph_impact(root)
    for label, fn in [
        ('Vector@1', lambda q=query: vector_answer(q, 1)),
        ('Vector@3', lambda q=query: vector_answer(q, 3)),
        ('Vector@10', lambda q=query: vector_answer(q, 10)),
        ('Graph', lambda r=root: graph_impact(r)),
    ]:
        result = fn()
        p, r, exact = metrics(result, truth)
        print(f'{root:<17} {label:<12} {p:6.2f} {r:6.2f} {str(exact):>7} {time_us(fn):11.1f}  {", ".join(sorted(result))}')
    print('Ground truth:', ', '.join(sorted(truth)))
    print()
print('Ranked docs for redis-x question:')
for i in vector_topk(QUESTIONS[0][1], 10):
    print(' ', DOCUMENTS[i][0], DOCUMENTS[i][1])
