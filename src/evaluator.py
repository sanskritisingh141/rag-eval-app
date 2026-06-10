from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.run_config import RunConfig

class RagasGeminiLLM(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        kwargs.pop("n", None)
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        kwargs.pop("n", None)
        return await super()._agenerate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

def evaluate_rag(question, answer, retrieved_docs, reference_answer=None):
    contexts = [doc.page_content for doc in retrieved_docs]

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [reference_answer or answer],
    }

    dataset = Dataset.from_dict(data)

    evaluator_llm = LangchainLLMWrapper(
    RagasGeminiLLM(
        model="gemini-2.5-flash-lite",
        convert_system_message_to_human=True,
    )
)

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=RunConfig(
        timeout=180,
        max_retries=2,
        max_wait=60,
    ),
    raise_exceptions=True,
)

    return result.to_pandas()