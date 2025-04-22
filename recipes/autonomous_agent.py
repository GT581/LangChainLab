"""
Autonomous Agent Recipe

This file demonstrates how to build an autonomous agent with custom tools
using Langchain. The agent can reason over provided tasks, use available tools as needed, and
create multi-step plans to solve problems or questions.

Key components:
1. Custom tool definitions
2. Agent with tool use capabilities
3. Memory for maintaining context
4. Planning capabilities

Documentation:
- https://python.langchain.com/v0.1/docs/modules/agents/
- https://python.langchain.com/api_reference/langchain/agents.html
"""
from datetime import datetime
import os
import random
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import get_llm

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory


class AutonomousAgent:
    """
    Autonomous agent that can use tools and reasoning to solve tasks.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the autonomous agent.
        
        Args:
            verbose: Whether to print detailed execution information
        """
        self.llm = get_llm()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[Any]:
        """
        Create the tools that the agent can use.
        
        Returns:
            List of tools
        """
        # Weather tool
        @tool
        def get_current_weather(location: str) -> str:
            """Get the current weather in a given location."""

            # Mock weather data
            weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
            temperatures = {"sunny": "75°F", "cloudy": "65°F", "rainy": "60°F", "snowy": "30°F", "windy": "55°F"}
            
            condition = random.choice(weather_conditions)
            temp = temperatures[condition]
            
            return f"Current weather in {location} is {condition} with a temperature of {temp}."
        
        # Time tool
        @tool
        def get_current_time() -> str:
            """Get the current date and time."""

            now = datetime.now()

            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Calculator tool
        @tool
        def calculate(operation: str) -> str:
            """Perform a mathematical calculation. Input should be a mathematical expression like '2 + 2'."""
            try:
                result = eval(operation)
                return f"Result: {result}"
            except Exception as e:
                return f"Error performing calculation: {str(e)}"
        
        # Internet Search tool
        search_tool = DuckDuckGoSearchResults(num_results=3)
        
        # Knowledge base tool
        @tool
        def query_knowledge_base(query: str) -> str:
            """Query the internal knowledge base for specific information."""
            
            # Mock knowledge base implementation
            knowledge_base = {
                "langchain": "LangChain is a framework for developing applications powered by language models.",
                "rag": "Retrieval Augmented Generation (RAG) combines retrieval of external data with text generation.",
                "llm": "Large Language Models (LLMs) are AI systems trained on vast text datasets to understand and generate human language.",
                "agent": "AI agents are systems that can perceive their environment, make decisions, and act to achieve goals.",
                "prompt engineering": "The practice of designing and optimizing prompts to effectively communicate with language models."
            }
            
            # Simple keyword matching
            results = []
            for key, value in knowledge_base.items():
                if query.lower() in key.lower() or key.lower() in query.lower():
                    results.append(f"{key}: {value}")
            
            if results:
                return "\n".join(results)
            else:
                return "No relevant information found in the knowledge base."
        
        return [
            get_current_weather,
            get_current_time,
            calculate,
            search_tool,
            query_knowledge_base
        ]
    
    def _create_agent(self) -> Any:
        """
        Create the agent with the custom tools.
        
        Returns:
            Agent
        """
        # Create the agent prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent AI assistant with access to various tools.
            Your goal is to help users by answering questions, performing tasks, and providing information.
            
            When a user asks a question, you should:
            1. Analyze what information is needed to answer
            2. Determine which tool is most appropriate to get that information
            3. Call the tool with the correct input
            4. Use the tool's output to provide a complete answer to the user
            
            You MUST use tools when they can provide relevant information.
            Don't just say what tool you would use - actually use it.
            
            After using tools, explain the results clearly and concisely.
            If you can answer directly without tools, do so.
            Always maintain a helpful, informative, and conversational tone.
            
            chat_history: {chat_history}
            """),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        # Create the agent (works with gemini llm instance, even though named openai)
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent to process a query.
        
        Args:
            query: The user query
            
        Returns:
            Dict with the result
        """
        return self.agent_executor.invoke({"input": query})


def demo():
    """
    Demonstrate the autonomous agent.
    """
    agent = AutonomousAgent(verbose=True)
    
    # Example queries for agent to execute
    queries = [
        "What's the weather like in New York?",
        "What is LangChain?",
        "Calculate 1234 * 5678",
        "What time is it now?",
        "Find information about the latest developments in AI",
        "What is RAG in the context of language models?"
    ]
    
    # Run each query
    for i, query in enumerate(queries, 1):
        print(f"\n=== Query {i}: {query} ===")
        try:
            result = agent.run(query)
            print(f"Agent response: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    demo() 

'''
Output:

=== Query 1: What's the weather like in New York? ===


> Entering new AgentExecutor chain...

Invoking: `get_current_weather` with `{'location': 'New York'}`


Current weather in New York is cloudy with a temperature of 65°F.I used the `get_current_weather` tool to get the current weather in New York.  It's currently cloudy with a temperature of 65°F.

> Finished chain.
Agent response: I used the `get_current_weather` tool to get the current weather in New York.  It's currently cloudy with a temperature of 65°F.

=== Query 2: What is LangChain? ===


> Entering new AgentExecutor chain...

Invoking: `query_knowledge_base` with `{'query': 'What is LangChain?'}`
responded: I can use the `query_knowledge_base` tool to answer this question.  I'll query it with "What is LangChain?".

langchain: LangChain is a framework for developing applications powered by language models.LangChain is a framework for developing applications powered by language models.

> Finished chain.
Agent response: LangChain is a framework for developing applications powered by language models.     

=== Query 3: Calculate 1234 * 5678 ===


> Entering new AgentExecutor chain...

Invoking: `calculate` with `{'operation': '1234 * 5678'}`


Result: 70066521234 multiplied by 5678 is 7,006,652.

> Finished chain.
Agent response: 1234 multiplied by 5678 is 7,006,652.

=== Query 4: What time is it now? ===


> Entering new AgentExecutor chain...

Invoking: `get_current_time` with `{}`


Current date and time: 2025-04-20 00:18:26The current date and time is 2025-04-20 00:18:26.

> Finished chain.
Agent response: The current date and time is 2025-04-20 00:18:26.

=== Query 5: Find information about the latest developments in AI ===


> Entering new AgentExecutor chain...

Invoking: `duckduckgo_results_json` with `{'query': 'latest developments in AI'}`
responded: I will use the `duckduckgo_results_json` tool to find information about the latest developments in AI.

snippet: Artificial Intelligence News. Everything on AI including futuristic robots with artificial intelligence, computer models of human intelligence and more., title: Artificial Intelligence News -- ScienceDaily, link: https://www.sciencedaily.com/news/computers_math/artificial_intelligence/, snippet: Discover the 10 major AI trends set to reshape 2025: from augmented working and real-time decision-making to advanced AI legislation and sustainable AI initiatives., title: The 10 Biggest AI Trends Of 2025 Everyone Must Be Ready For Today - Forbes, link: https://www.forbes.com/sites/bernardmarr/2024/09/24/the-10-biggest-ai-trends-of-2025-everyone-must-be-ready-for-today/, snippet: Making AI-generated code more accurate in any language. A new technique automatically guides an LLM toward outputs that adhere to the rules of whatever programming language or other format is being used. April 18, 2025. Read full story →, title: Artificial intelligence | MIT News | Massachusetts Institute of Technology, link: https://news.mit.edu/topic/artificial-intelligence2Recent developments in AI include advancements in AI-generated code accuracy,  new AI trends predicted for 2025 (such as augmented working and real-time decision-making), and ongoing news and research in the field, as reported by sources like ScienceDaily and MIT News.  For more detailed information, please consult the links provided in the tool's output.

> Finished chain.
Agent response: Recent developments in AI include advancements in AI-generated code accuracy,  new AI trends predicted for 2025 (such as augmented working and real-time decision-making), and ongoing news and research in the field, as reported by sources like ScienceDaily and MIT News.  For more detailed information, please consult the links provided in the tool's output.

=== Query 6: What is RAG in the context of language models? ===


> Entering new AgentExecutor chain...

Invoking: `query_knowledge_base` with `{'query': 'What is RAG in the context of language models?'}`  
responded: I can use the `query_knowledge_base` tool to answer your question about RAG in the context of language models.  One moment.

rag: Retrieval Augmented Generation (RAG) combines retrieval of external data with text generation.In the context of language models, RAG stands for Retrieval Augmented Generation.  It combines the retrieval of external data with text generation.

> Finished chain.
Agent response: In the context of language models, RAG stands for Retrieval Augmented Generation.  It combines the retrieval of external data with text generation.
'''