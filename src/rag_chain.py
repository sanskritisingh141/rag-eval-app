from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are a careful document QA assistant.

Answer only using the provided context.
If the answer is not present in the context, say:
"I could not find this in the uploaded document."

Include concise reasoning.
Do not invent facts.

Context:
{context}
"""


def format_context(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")

        formatted.append(
            f"[Source {i}] file={source}, page={page}, chunk={chunk_id}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def answer_question(question, retrieved_docs):
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    context = format_context(retrieved_docs)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    citations = [
        {
            "source": doc.metadata.get("source", "uploaded document"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "text": doc.page_content[:500]
        }
        for doc in retrieved_docs
    ]

    return response.content, citations