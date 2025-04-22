"""
Examples of using Embeddings with Langchain.

This module demonstrates different ways to use Embedding Models in Langchain:
1. Basic text embeddings
2. Using embeddings for semantic search
3. Document embeddings
4. Creating a simple vector store with embeddings

Documentation:
- https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
- https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html
"""
import sys
import os

import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import get_embeddings


def basic_embedding_example() -> None:
    """
    Basic example of getting embeddings for text.
    """
    print("\n=== Basic Embedding Example ===")
    
    embedding_model = get_embeddings()
    
    # Get embeddings for a single text
    text = "Artificial intelligence is transforming the world."
    embedding = embedding_model.embed_query(text)
    
    # Displaying embedding info
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Get embeddings for multiple texts
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning is a subset of AI.",
        "Python is a popular programming language for AI."
    ]
    
    embeddings = embedding_model.embed_documents(texts)
    
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Each embedding dimension: {len(embeddings[0])}")


def embedding_similarity_example() -> None:
    """
    Example of using embeddings for semantic similarity.
    """
    print("\n=== Embedding Similarity Example ===")
    
    embedding_model = get_embeddings()
    
    texts = [
        "The cat sat on the mat.",
        "A feline was resting on a rug.",
        "The dog played in the yard.",
        "The weather is nice today.",
        "My cat likes sleeping on soft surfaces."
    ]
    
    embeddings = [embedding_model.embed_query(text) for text in texts]
    
    # Function to calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Compare similarity between first text and all others
    reference_text = texts[0]
    reference_embedding = embeddings[0]
    
    print(f"Reference text: '{reference_text}'")
    print("Similarities:")
    
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        similarity = cosine_similarity(reference_embedding, embedding)
        print(f"  - '{text}': {similarity:.4f}")


def document_embeddings_example() -> None:
    """
    Example of embedding documents and performing simple retrieval.
    """
    print("\n=== Document Embeddings Example ===")
    
    embedding_model = get_embeddings()
    
    # Create sample documents for vector store
    documents = [
        Document(page_content="Artificial intelligence (AI) is intelligence demonstrated by machines.",
                 metadata={"source": "wikipedia", "topic": "AI"}),
        Document(page_content="Machine learning is a subset of AI focused on data and algorithms.",
                 metadata={"source": "textbook", "topic": "ML"}),
        Document(page_content="Neural networks are computing systems inspired by biological neural networks.",
                 metadata={"source": "research paper", "topic": "Neural Networks"}),
        Document(page_content="Python is a high-level, general-purpose programming language.",
                 metadata={"source": "documentation", "topic": "Programming"}),
        Document(page_content="LangChain is a framework for developing applications powered by language models.",
                 metadata={"source": "github", "topic": "LLM"})
    ]
    
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    query = "Tell me about artificial intelligence"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"Query: '{query}'")
    print("Top 2 most relevant documents:")
    for i, doc in enumerate(results):
        print(f"Document {i+1}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    basic_embedding_example()
    embedding_similarity_example()
    document_embeddings_example()

'''
Output:

=== Basic Embedding Example ===
Text: Artificial intelligence is transforming the world.
Embedding dimension: 768
First 5 values: [-0.011484945192933083, -0.009393615648150444, -0.004084668587893248, 0.012290679849684238, -0.04011889919638634]
Number of embeddings: 3
Each embedding dimension: 768

=== Embedding Similarity Example ===
Reference text: 'The cat sat on the mat.'
Similarities:
  - 'The cat sat on the mat.': 1.0000
  - 'A feline was resting on a rug.': 0.7686
  - 'The dog played in the yard.': 0.5369
  - 'The weather is nice today.': 0.4228
  - 'My cat likes sleeping on soft surfaces.': 0.5907

=== Document Embeddings Example ===
Query: 'Tell me about artificial intelligence'
Top 2 most relevant documents:
Document 1:
  Content: Artificial intelligence (AI) is intelligence demonstrated by machines.
  Metadata: {'source': 'wikipedia', 'topic': 'AI'}

Document 2:
  Content: Machine learning is a subset of AI focused on data and algorithms.
  Metadata: {'source': 'textbook', 'topic': 'ML'}
'''