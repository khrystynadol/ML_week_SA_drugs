from sentence_transformers import SentenceTransformer, util
import numpy as np


class BertRetriever:
    def __init__(self, docs: list[str], model_name="paraphrase-MiniLM-L6-v2"):
        """
        Initialize the retriever with a list of documents and a SentenceTransformer model.

        Args:
            docs (list[str]): List of documents to retrieve from.
            model_name (str): Name of the SentenceTransformers model.
        """
        self.docs = docs
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.model.encode(docs, convert_to_tensor=True)

    def get_relevant_docs(self, query: str, n=3) -> list[str]:
        """
        Retrieve the top-n most relevant documents for a query.

        Args:
            query (str): The input query string.
            n (int): Number of top relevant documents to return.

        Returns:
            list[str]: List of top-n relevant documents.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        print(scores)
        top_n_indices = np.argsort(scores.cpu().numpy())[::-1][:n]
        print(top_n_indices)
        return [self.docs[i] for i in top_n_indices]
