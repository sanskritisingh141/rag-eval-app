from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def calculate_context_similarity(question, answer, retrieved_docs):
    contexts = " ".join([doc.page_content for doc in retrieved_docs])

    answer_embedding = model.encode([answer])
    context_embedding = model.encode([contexts])

    score = cosine_similarity(answer_embedding, context_embedding)[0][0]
    return float(score)


def calculate_question_answer_relevance(question, answer):
    question_embedding = model.encode([question])
    answer_embedding = model.encode([answer])

    score = cosine_similarity(question_embedding, answer_embedding)[0][0]
    return float(score)


def calculate_citation_coverage(answer, retrieved_docs):
    answer_words = set(answer.lower().split())

    context_words = set()
    for doc in retrieved_docs:
        context_words.update(doc.page_content.lower().split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


def calculate_retrieval_strength(retrieved_docs):
    if not retrieved_docs:
        return 0.0

    non_empty_docs = [
        doc for doc in retrieved_docs
        if doc.page_content.strip()
    ]

    return len(non_empty_docs) / len(retrieved_docs)


def evaluate_local(question, answer, retrieved_docs):
    context_similarity = calculate_context_similarity(
        question,
        answer,
        retrieved_docs
    )

    answer_relevance = calculate_question_answer_relevance(
        question,
        answer
    )

    citation_coverage = calculate_citation_coverage(
        answer,
        retrieved_docs
    )

    retrieval_strength = calculate_retrieval_strength(
        retrieved_docs
    )

    return {
        "context_similarity": round(context_similarity, 3),
        "answer_relevance": round(answer_relevance, 3),
        "citation_coverage": round(citation_coverage, 3),
        "retrieval_strength": round(retrieval_strength, 3),
    }