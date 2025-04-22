"""
Examples of Text Splitting and Chunking with Langchain.

This module demonstrates different ways to split text into chunks in Langchain:
1. Character-based text splitting
2. Recursive character text splitting
3. Token-based text splitting
4. Markdown header text splitting
5. Semantic chunking
6. Special handling for languages without word boundaries
7. Combining text splitters with different strategies

Documentation:
- https://python.langchain.com/docs/concepts/text_splitters/
"""
import sys
import os

from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import get_embeddings


def basic_character_splitter_example() -> None:
    """
    Example of basic character text splitter.
    This is the simplest text splitter, splitting on a specific character.
    """
    print("\n=== Basic Character Text Splitter Example ===")
    
    # Sample text
    text = """Langchain is a framework for developing applications powered by language models. 
    It provides a standard interface for chains, lots of integrations with other tools, 
    and end-to-end chains for common applications. Langchain makes it simple to create 
    applications using LLMs including chatbots, summarization tools, and more."""
    
    # Initialize splitter with newline character
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=0
    )
    
    # Split the text
    chunks = splitter.split_text(text)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"{chunk}")


def recursive_character_splitter_example() -> None:
    """
    Example of recursive character text splitter.
    This splitter tries multiple separators in order until chunks are small enough.
    """
    print("\n=== Recursive Character Text Splitter Example ===")
    
    # Sample text
    text = """Langchain is a framework for developing applications powered by language models. 
    
    It provides a standard interface for chains, lots of integrations with other tools, 
    and end-to-end chains for common applications.
    
    Langchain makes it simple to create applications using LLMs including:
    - Chatbots
    - Summarization tools
    - Question answering systems
    - And much more
    
    You can find the documentation at https://python.langchain.com/
    """
    
    # Initialize recursive splitter with default separators: ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"{chunk}")
    
    # Separators visual
    print("\nDemonstrating order of separators:")
    small_text = "First sentence. Second sentence. Third sentence with a, comma."
    small_splitter = RecursiveCharacterTextSplitter(
        separators=[". ", ", ", " "],
        chunk_size=20,
        chunk_overlap=0
    )
    small_chunks = small_splitter.split_text(small_text)
    print(f"Number of small chunks: {len(small_chunks)}")
    for i, chunk in enumerate(small_chunks):
        print(f"Small chunk {i+1}: {chunk}")


def token_splitter_example() -> None:
    """
    Example of token-based text splitter.
    This splits by token count rather than character count.
    """
    print("\n=== Token Text Splitter Example ===")
    
    text = """Language models like GPT-4 process text using tokens rather than characters.
    A token is commonly a word or part of a word, but can also be a character, or even 
    a byte of UTF-8 encoded text. For English text, 1 token is approximately 4 characters 
    or 0.75 words. Chunking text by tokens ensures that each chunk fits within the model's
    context window."""
    
    splitter = TokenTextSplitter(
        chunk_size=20,
        chunk_overlap=2
    )
    
    chunks = splitter.split_text(text)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"{chunk}")


def markdown_header_splitter_example() -> None:
    """
    Example of markdown header text splitter.
    This splits markdown text by specified headers and preserves header context in metadata.
    """
    print("\n=== Markdown Header Text Splitter Example ===")
    
    markdown_text = """# Langchain Framework
    
    Langchain is a framework for LLM applications.
    
    ## Components
    
    ### Chains
    Chains allow combining multiple components together to solve a task.
    
    ### Agents
    Agents use LLMs to determine which actions to take.
    
    ## Use Cases
    
    ### Chatbots
    Build chatbots on your data.
    
    ### RAG
    Build retrieval augmented generation systems.
    """
    
    # Define markdown headers for splitting
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    chunks = splitter.split_text(markdown_text)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")


def language_aware_splitter_example() -> None:
    """
    Example of language-aware text splitting.
    RecursiveCharacterTextSplitter can be configured for specific programming languages.
    """
    print("\n=== Language-Aware Text Splitter Example ===")
    
    # Sample Python code
    python_code = """
    def hello_world():
        \"\"\"This is a simple function that prints a greeting.\"\"\"
        print("Hello, world!")

        for i in range(3):
            print(f"Count: {i}")

    class MyClass:
        def __init__(self, name):
            self.name = name

        def greet(self):
            return f"Hello, {self.name}"

    if __name__ == "__main__":
        hello_world()
        obj = MyClass("Langchain")
        print(obj.greet())
    """
    
    # Initialize language-specific splitter for Python
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, #https://python.langchain.com/v0.2/docs/how_to/code_splitter/
        chunk_size=100,
        chunk_overlap=0
    )
    
    chunks = python_splitter.split_text(python_code)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"{chunk}")


def combined_splitting_strategy_example() -> None:
    """
    Example of combining different splitting strategies.
    First split by markdown headers, then split large chunks by characters.
    """
    print("\n=== Combined Splitting Strategy Example ===")
    
    # Sample markdown text with long sections
    markdown_text = """# Machine Learning
    
    Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
    
    ## Supervised Learning
    
    Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal).
    
    ## Unsupervised Learning
    
    Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis to find hidden patterns or groupings in data.
    """
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    
    # First, split by markdown headers
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_chunks = md_splitter.split_text(markdown_text)
    
    print(f"Number of markdown chunks: {len(md_chunks)}")
    for i, chunk in enumerate(md_chunks):
        print(f"\nMarkdown Chunk {i+1}:")
        print(f"Content: {chunk.page_content[:50]}...")
        print(f"Metadata: {chunk.metadata}")
    
    # Second, further split long chunks by characters
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    
    final_chunks = []
    for md_chunk in md_chunks:
        smaller_chunks = char_splitter.split_text(md_chunk.page_content)
        for small_chunk in smaller_chunks:
            # Copy metadata from the markdown chunk
            final_chunks.append({
                "content": small_chunk,
                "metadata": md_chunk.metadata
            })
    
    print(f"\nNumber of final chunks after character splitting: {len(final_chunks)}")
    for i, chunk in enumerate(final_chunks):
        print(f"\nFinal Chunk {i+1}:")
        print(f"Content: {chunk['content']}")
        print(f"Metadata: {chunk['metadata']}")


def semantic_chunker_example() -> None:
    """
    Example of semantic chunking.
    Splits text based on semantic similarity rather than just characters or tokens.
    Requires langchain_experimental package and an embedding model.
    """    
    print("\n=== Semantic Chunker Example ===")
    
    # Sample text with different topics
    text = """Artificial intelligence is transforming industries around the world. 
    Machine learning models can now recognize patterns in data that humans cannot detect.
    
    Basketball is a team sport where players try to score points by getting a ball through a hoop.
    The NBA is the premier professional basketball league in the United States.
    
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This process provides the oxygen that we breathe and the food that we eat."""
    
    embeddings = get_embeddings()
    semantic_splitter = SemanticChunker(embeddings)
    
    chunks = semantic_splitter.split_text(text)
    
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"{chunk}")


if __name__ == "__main__":
    basic_character_splitter_example()
    recursive_character_splitter_example()
    token_splitter_example()
    markdown_header_splitter_example()
    language_aware_splitter_example()
    combined_splitting_strategy_example()
    semantic_chunker_example()

'''
Output:

=== Basic Character Text Splitter Example ===
Number of chunks: 4

Chunk 1:
Langchain is a framework for developing applications powered by language models.

Chunk 2:
It provides a standard interface for chains, lots of integrations with other tools,

Chunk 3:
and end-to-end chains for common applications. Langchain makes it simple to create

Chunk 4:
applications using LLMs including chatbots, summarization tools, and more.

=== Recursive Character Text Splitter Example ===
Number of chunks: 6

Chunk 1:
Langchain is a framework for developing applications powered by language models.

Chunk 2:
It provides a standard interface for chains, lots of integrations with other tools,

Chunk 3:
and end-to-end chains for common applications.

Chunk 4:
Langchain makes it simple to create applications using LLMs including:
    - Chatbots

Chunk 5:
- Chatbots
    - Summarization tools
    - Question answering systems
    - And much more

Chunk 6:
You can find the documentation at https://python.langchain.com/

Demonstrating order of separators:
Number of small chunks: 5
Small chunk 1: First sentence
Small chunk 2: . Second sentence
Small chunk 3: . Third sentence
Small chunk 4: with a
Small chunk 5: , comma.

=== Token Text Splitter Example ===
Number of chunks: 6

Chunk 1:
Language models like GPT-4 process text using tokens rather than characters.
    A

Chunk 2:
  A token is commonly a word or part of a word, but can also be a character,

Chunk 3:
 character, or even
    a byte of UTF-8 encoded text. For English

Chunk 4:
 For English text, 1 token is approximately 4 characters
    or 0.75 words

Chunk 5:
75 words. Chunking text by tokens ensures that each chunk fits within the model's


Chunk 6:

    context window.

=== Markdown Header Text Splitter Example ===
Number of chunks: 5

Chunk 1:
Content: Langchain is a framework for LLM applications.
Metadata: {'Header 1': 'Langchain Framework'}

Chunk 2:
Content: Chains allow combining multiple components together to solve a task.
Metadata: {'Header 1': 'Langchain Framework', 'Header 2': 'Components', 'Header 3': 'Chains'}        

Chunk 3:
Content: Agents use LLMs to determine which actions to take.
Metadata: {'Header 1': 'Langchain Framework', 'Header 2': 'Components', 'Header 3': 'Agents'}        

Chunk 4:
Content: Build chatbots on your data.
Metadata: {'Header 1': 'Langchain Framework', 'Header 2': 'Use Cases', 'Header 3': 'Chatbots'}       

Chunk 5:
Content: Build retrieval augmented generation systems.
Metadata: {'Header 1': 'Langchain Framework', 'Header 2': 'Use Cases', 'Header 3': 'RAG'}

=== Language-Aware Text Splitter Example ===
Number of chunks: 7

Chunk 1:
def hello_world():
        """This is a simple function that prints a greeting."""

Chunk 2:
print("Hello, world!")

Chunk 3:
for i in range(3):
            print(f"Count: {i}")

Chunk 4:
class MyClass:
        def __init__(self, name):
            self.name = name

Chunk 5:
def greet(self):
            return f"Hello, {self.name}"

Chunk 6:
if __name__ == "__main__":
        hello_world()
        obj = MyClass("Langchain")

Chunk 7:
print(obj.greet())

=== Combined Splitting Strategy Example ===
Number of markdown chunks: 3

Markdown Chunk 1:
Content: Machine learning is a field of study that gives co...
Metadata: {'Header 1': 'Machine Learning'}

Markdown Chunk 2:
Content: Supervised learning is the machine learning task o...
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Markdown Chunk 3:
Content: Unsupervised learning is a type of machine learnin...
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}

Number of final chunks after character splitting: 11

Final Chunk 1:
Content: Machine learning is a field of study that gives computers the ability to learn without being
Metadata: {'Header 1': 'Machine Learning'}

Final Chunk 2:
Content: learn without being explicitly programmed.
Metadata: {'Header 1': 'Machine Learning'}

Final Chunk 3:
Content: Supervised learning is the machine learning task of learning a function that maps an input to an
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Final Chunk 4:
Content: maps an input to an output based on example input-output pairs. It infers a function from labeled
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Final Chunk 5:
Content: from labeled training data consisting of a set of training examples. In supervised learning, each
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Final Chunk 6:
Content: learning, each example is a pair consisting of an input object (typically a vector) and a desired
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Final Chunk 7:
Content: and a desired output value (also called the supervisory signal).
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}

Final Chunk 8:
Content: Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}

Final Chunk 9:
Content: from datasets consisting of input data without labeled responses. The most common unsupervised
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}

Final Chunk 10:
Content: common unsupervised learning method is cluster analysis, which is used for exploratory data 
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}

Final Chunk 11:
Content: exploratory data analysis to find hidden patterns or groupings in data.
Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}

=== Semantic Chunker Example ===
Number of chunks: 2

Chunk 1:
Artificial intelligence is transforming industries around the world. Machine learning models can now recognize patterns in data that humans cannot detect. Basketball is a team sport where players try to score points by getting a ball through a hoop. The NBA is the premier professional basketball league in the United States. Photosynthesis is the process by which plants convert light energy into chemical energy.

Chunk 2:
This process provides the oxygen that we breathe and the food that we eat.
'''