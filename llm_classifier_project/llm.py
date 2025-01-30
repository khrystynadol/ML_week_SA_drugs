from litellm import completion
from bert_retriever import BertRetriever
from rerankers import Reranker


class LLM:
    PROMPT = "You are a helpful assistant that can answer questions. " \
             "Your main goal is to name side effects by using exact drugs. " \
             "If you do not know the answer say that there is no side effects described in reviews. " \
             "Use the provided reviews to answer questions." \
             "It is very important to name all side effects described in reviews. "

    def __init__(self, docs: list[str], docs_headers_map: dict[str, str]):  # , params: dict[str, bool]):
        self.docs = docs
        self.docs_headers_map = docs_headers_map
        self.bert_retriever = BertRetriever(docs)
        # self.params = params

    def answer_question(self, query):
        context = self.bert_retriever.get_relevant_docs(query)

        reranker = Reranker('cross-encoder', model_type='cross-encoder')
        sorted_context = reranker.rank(query, context)
        context = [r.document.text for r in sorted_context.top_k(3)]

        headers = [self.docs_headers_map[c[:30]] for c in context]
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": self.PROMPT},
                {"role": "user",
                 "content": f"Context: {context} \n Query: {query}"}
            ],
        )
        return response.choices[0].message.content, context[:3], headers
