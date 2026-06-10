from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def retrieve_docs(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)

def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)


def load_vectorstore(path, embeddings):
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )