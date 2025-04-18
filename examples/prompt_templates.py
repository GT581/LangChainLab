"""
Examples of using Prompt Templates with Langchain.

This module demonstrates the most useful prompt template patterns in LangChain:
1. Basic prompt templates - Simple variable substitution
2. Chat prompt templates - Creating structured chat conversations
3. Few-shot prompt templates - Learning from examples
4. LLM integration - Using templates with language models

Documentation:
- https://python.langchain.com/docs/concepts/prompt_templates/
- https://python.langchain.com/api_reference/core/prompts.html
"""
from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    FewShotPromptTemplate
)
from langchain.schema import HumanMessage, SystemMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import get_llm


def basic_prompt_template_example() -> None:
    """
    Example of using basic prompt templates with variable substitution.
    """
    print("\n=== Basic Prompt Template Example ===")
    
    template = "You are an expert on {topic}. Please explain {concept} in simple terms."
    prompt_template = PromptTemplate.from_template(template)
    
    # Format the template with topic and concept values
    formatted_prompt = prompt_template.format(
        topic="artificial intelligence",
        concept="neural networks"
    )
    
    print(f"Template: {template}")
    print(f"Formatted prompt: {formatted_prompt}")
    print(f"Input variables required: {prompt_template.input_variables}")


def chat_prompt_template_example() -> None:
    """
    Example of creating structured chat conversations with templates.
    """
    print("\n=== Chat Prompt Template Example ===")
    
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant that specializes in {subject}."),
        HumanMessage(content="Can you tell me about {topic}?"),
    ])
    
    # Format the template with subject and topic values
    messages = template.format_messages(
        subject="history",
        topic="The Industrial Revolution"
    )
    
    print("Chat messages after formatting:")
    for message in messages:
        print(f"  Role: {message.type}")
        print(f"  Content: {message.content}")


def few_shot_prompt_template_example() -> None:
    """
    Example of few-shot learning templates with examples.
    """
    print("\n=== Few-Shot Prompt Template Example ===")
    
    # Define the examples for few-shot learning
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"}
    ]
    
    # Define the example formatter template
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of each input:",
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
        example_separator="\n\n"
    )
    
    # Format with the input
    few_shot_formatted = few_shot_prompt.format(input="big")
    print(few_shot_formatted)


def llm_with_templates_example() -> None:
    """
    Example of using prompt templates with an LLM.
    """
    print("\n=== Using Prompt Templates with LLM ===")
    
    llm = get_llm()
    
    template = PromptTemplate.from_template(
        "Write a short {tone} joke about {topic}."
    )
    
    prompt = template.format(tone="funny", topic="programming")
    print(f"Sending prompt: {prompt}")
    
    response = llm.invoke(prompt)
    print(f"LLM response: {response.content}")
    

if __name__ == "__main__":
    basic_prompt_template_example()
    chat_prompt_template_example()
    few_shot_prompt_template_example()
    llm_with_templates_example()

'''
Output:

=== Basic Prompt Template Example ===
Template: You are an expert on {topic}. Please explain {concept} in simple terms.
Formatted prompt: You are an expert on artificial intelligence. Please explain neural networks in simple terms.
Input variables required: ['concept', 'topic']

=== Chat Prompt Template Example ===
Chat messages after formatting:
  Role: system
  Content: You are a helpful AI assistant that specializes in {subject}.
  Role: human
  Content: Can you tell me about {topic}?

=== Few-Shot Prompt Template Example ===
Give the antonym of each input:

Input: happy
Output: sad

Input: tall
Output: short

Input: energetic
Output: lethargic

Input: big
Output: small

=== Using Prompt Templates with LLM ===
Sending prompt: Write a short funny joke about programming.
LLM response: Why do programmers prefer dark mode?  Because light attracts bugs!
'''