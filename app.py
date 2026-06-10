import streamlit as st
from dotenv import load_dotenv

from src.loaders import load_document
from src.chunking import chunk_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore, retrieve_docs
from src.rag_chain import answer_question
from src.evaluator import evaluate_rag
from src.local_evaluator import evaluate_local

load_dotenv()

st.set_page_config(
    page_title="Production RAG Evaluation App",
    layout="wide"
)

st.title("Document RAG with Evaluation Dashboard")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "history" not in st.session_state:
    st.session_state.history = []

tab_upload, tab_ask, tab_eval, tab_history = st.tabs(
    ["Upload", "Ask", "Evaluation", "History"]
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", 2, 8, 4)
    st.caption("Higher values give more context but slower answers.")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Loading and indexing document..."):
            docs = load_document(uploaded_file)
            chunks = chunk_documents(docs)
            embeddings = get_embeddings()
            vectorstore = create_vectorstore(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")

with tab_ask:
    question = st.text_input("Ask a question about the uploaded document")

    top_k = st.slider("Number of retrieved chunks", 2, 8, 4)

    if st.button("Get Answer"):
        if st.session_state.vectorstore is None:
            st.error("Please upload and process a document first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving and answering..."):
                    docs = retrieve_docs(
                        st.session_state.vectorstore,
                        question,
                        k=top_k
                    )

                    answer, citations = answer_question(question, docs)

                    st.session_state.last_question = question
                    st.session_state.last_answer = answer
                    st.session_state.last_docs = docs

                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "citations": citations
                    })

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source Citations")
                for i, citation in enumerate(citations, start=1):
                    with st.expander(
                    f"Source {i}: page {citation['page']}, chunk {citation['chunk_id']}"
                ):
                        st.write(citation["text"])

            except Exception as e:
                error_text = str(e)

                if(
                "429" in error_text
                or "quota" in error_text.lower()
                or "ResourceExhausted" in error_text
            ):
                    st.warning(
                    "Gemini quota was reached while generating the answer. "
                    "Please wait and try again with fewer retrieved chunks."
                )
                elif "api key" in error_text.lower() or "google_api_key" in error_text.lower():
                    st.error(
                    "Google API key is missing or invalid. Please check your `.env` file."
                )
                else:
                    st.error(f"Answer generation failed: {e}")

with tab_eval:
    st.subheader("Evaluation Dashboard")

    st.info(
        "Default evaluation uses local metrics and does not consume LLM quota. "
        "RAGAS evaluation is optional and may use Gemini API quota."
    )

    eval_mode = st.radio(
        "Choose evaluation mode",
        ["Local Evaluation", "RAGAS Evaluation"],
        horizontal=True
    )

    reference_answer = None

    if eval_mode == "RAGAS Evaluation":
        reference_answer = st.text_area(
            "Optional reference answer for stronger RAGAS evaluation"
        )

    if st.button("Run Evaluation"):
        if (
            "last_question" not in st.session_state
            or "last_answer" not in st.session_state
            or "last_docs" not in st.session_state
        ):
            st.warning("Please ask a question first before running evaluation.")

        else:
            if eval_mode == "Local Evaluation":
                scores = evaluate_local(
                    question=st.session_state.last_question,
                    answer=st.session_state.last_answer,
                    retrieved_docs=st.session_state.last_docs
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Context Similarity",
                    scores["context_similarity"]
                )

                col2.metric(
                    "Answer Relevance",
                    scores["answer_relevance"]
                )

                col3.metric(
                    "Citation Coverage",
                    scores["citation_coverage"]
                )

                col4.metric(
                    "Retrieval Strength",
                    scores["retrieval_strength"]
                )

                st.subheader("Metric Meaning")

                st.write(
                    "**Context Similarity:** Checks how semantically close the answer is to the retrieved chunks."
                )

                st.write(
                    "**Answer Relevance:** Checks how semantically close the answer is to the user question."
                )

                st.write(
                    "**Citation Coverage:** Checks how many answer words are supported by retrieved context words."
                )

                st.write(
                    "**Retrieval Strength:** Checks whether retrieved chunks contain usable text."
                )

                if "eval_history" not in st.session_state:
                    st.session_state.eval_history = []

                st.session_state.eval_history.append({
                    "mode": "Local Evaluation",
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "scores": scores
                })

            else:
                try:
                    scores = evaluate_rag(
                        question=st.session_state.last_question,
                        answer=st.session_state.last_answer,
                        retrieved_docs=st.session_state.last_docs,
                        reference_answer=reference_answer or None
                    )

                    st.dataframe(scores)

                    numeric_cols = scores.select_dtypes(include="number").columns

                    for col in numeric_cols:
                        value = scores[col][0]
                        st.metric(col, round(float(value), 3))

                    if "eval_history" not in st.session_state:
                        st.session_state.eval_history = []

                    st.session_state.eval_history.append({
                        "mode": "RAGAS Evaluation",
                        "question": st.session_state.last_question,
                        "answer": st.session_state.last_answer,
                        "scores": scores.to_dict()
                    })

                except Exception as e:
                    error_text = str(e)

                    if (
                        "429" in error_text
                        or "quota" in error_text.lower()
                        or "ResourceExhausted" in error_text
                    ):
                        st.warning(
                            "Gemini free-tier quota was reached. "
                            "Please wait and try again, or use Local Evaluation."
                        )
                    else:
                        st.error(f"RAGAS evaluation failed: {e}")

with tab_history:
    st.subheader("Question History")

    for item in st.session_state.history:
        with st.expander(item["question"]):
            st.write(item["answer"])

    st.subheader("Evaluation History")

    if "eval_history" not in st.session_state or not st.session_state.eval_history:
        st.info("No evaluations run yet.")
    else:
        for item in st.session_state.eval_history:
            with st.expander(f"{item['mode']} | {item['question']}"):
                st.write("Answer:")
                st.write(item["answer"])

                st.write("Scores:")
                st.json(item["scores"])
