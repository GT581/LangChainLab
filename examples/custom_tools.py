"""
Creating Custom Tools for Agents in Langchain

This module demonstrates how to create and use custom tools with Langchain agents.
It covers different approaches to tool creation and using tools with external APIs.

Key concepts:
1. Basic tool creation with decorators
2. Integration with external APIs

Documentation:
- https://python.langchain.com/docs/concepts/tools/
- https://api.python.langchain.com/en/latest/tools/langchain.tools.base.BaseTool.html
"""
from datetime import datetime
import os
import sys

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import BaseTool, tool
from langchain.utilities import WikipediaAPIWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import get_llm


# Basic Tool Creation

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate.
        
    Returns:
        The result of the evaluation.
    """
    try:
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "Expression contains invalid characters. Only numbers and basic operators are allowed."
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        The current date and time as a string.
    """
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def word_count(text: str) -> str:
    """
    Count the number of words in a text.
    
    Args:
        text: The text to count words in.
        
    Returns:
        The number of words in the text.
    """
    words = text.split()
    return f"Word count: {len(words)}"


def basic_tools_example():
    """
    Demonstrate using basic tools with an agent.
    """
    print("\n=== Basic Tools Example ===\n")
    
    tools = [calculator, current_time, word_count]
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools. Use them when needed to provide accurate information."),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    
    # Create the agent and executor
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Run the agent with different inputs
    inputs = [
        "What is 25 * 4 + 30 / 2?",
        "What time is it right now?",
        "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?"
    ]
    
    for i, input_text in enumerate(inputs):
        print(f"\nInput {i+1}: {input_text}")
        result = agent_executor.invoke({"input": input_text})
        print(f"Output: {result['output']}")


# Integration with External APIs

def create_wikipedia_tool() -> BaseTool:
    """
    Create a tool for searching Wikipedia.
    
    Returns:
        A tool for searching Wikipedia.
    """
    try:
        wikipedia = WikipediaAPIWrapper()
        
        @tool
        def search_wikipedia(query: str) -> str:
            """
            Search Wikipedia for information on a topic.
            
            Args:
                query: The search query.
                
            Returns:
                Summary information from Wikipedia.
            """
            try:
                result = wikipedia.run(query)
                if not result or len(result.strip()) == 0:
                    return f"No Wikipedia information found for: {query}"
                return result
            except Exception as e:
                return f"Error searching Wikipedia: {str(e)}"
        
        return search_wikipedia
    
    except Exception as e:
        return f"Wikipedia search is not available. Error: {str(e)}"


def create_duckduckgo_search_tool() -> BaseTool:
    """
    Create a tool for DuckDuckGo Search.
    
    Returns:
        A tool for DuckDuckGo Search.
    """
    try:
        search_tool = DuckDuckGoSearchResults(num_results=5)
        return search_tool
    
    except Exception as e:
        return f"DuckDuckGo search is not available. Error: {str(e)}"


def api_tools_example():
    """
    Demonstrate tools that integrate with external APIs.
    """
    print("\n=== External API Integration Example ===\n")
    
    # Create API tools
    wikipedia_tool = create_wikipedia_tool()
    duckduckgo_tool = create_duckduckgo_search_tool() 
    
    tools = [wikipedia_tool, duckduckgo_tool]
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools including a calculator, Wikipedia search, and DuckDuckGo search. Use them to provide accurate and up-to-date information."),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    inputs = [
        "Who won the serie a last season?",
        "What is the capital of Italy?",
        "Tell me about the wikipedia page for Cats"
    ]
    
    for i, input_text in enumerate(inputs):
        print(f"\nInput {i+1}: {input_text}")
        result = agent_executor.invoke({"input": input_text})
        print(f"Output: {result['output']}")


if __name__ == "__main__":
    basic_tools_example()
    api_tools_example()


'''
Output:

=== Basic Tools Example ===


Input 1: What is 25 * 4 + 30 / 2?


> Entering new AgentExecutor chain...

Invoking: `calculator` with `{'expression': '25 * 4 + 30 / 2'}`


Result: 115.0The answer is 115.0

> Finished chain.
Output: The answer is 115.0

Input 2: What time is it right now?


> Entering new AgentExecutor chain...

Invoking: `current_time` with `{}`


Current time: 2025-04-20 19:49:01Current time: 2025-04-20 19:49:01

> Finished chain.
Output: Current time: 2025-04-20 19:49:01

Input 3: How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?


> Entering new AgentExecutor chain...

Invoking: `word_count` with `{'text': 'The quick brown fox jumps over the lazy dog'}`


Word count: 9There are 9 words.

> Finished chain.
Output: There are 9 words.

=== External API Integration Example ===


Input 1: Who won the serie a last season?


> Entering new AgentExecutor chain...

Invoking: `duckduckgo_results_json` with `{'query': 'Who won the serie a last season?'}`


snippet: The 2021-22 Serie A (known as the Serie A TIM for sponsorship reasons) was the 120th season of top-tier Italian football, the 90th in a round-robin tournament, and the 12th since its organization under an own league committee, the Lega Serie A. [2] Inter Milan were the defending champions.. On 22 May 2022, following victory in their final match against Sassuolo, Milan were crowned champions ..., title: 2021-22 Serie A - Wikipedia, link: https://en.wikipedia.org/wiki/2021–22_Serie_A, snippet: Inter have won the 2023-24 Serie A title after an incredible season. The team coached by Simone Inzaghi has secured its 20th Serie A title, the second-most of any in Italy, behind only Juventus., title: Inter become Serie A champions with derby win over Milan, earn 'second ..., link: https://www.cbssports.com/soccer/news/inter-become-serie-a-champions-with-derby-win-over-milan-earn-second-star-for-historic-20th-scudetto-win/, snippet: The Serie A (Italian pronunciation: [ˈsɛːrje ˈa]), [1] officially known as Serie A Enilive [2] in Italy and Serie A Made in Italy abroad for sponsorship reasons, is a professional association football league in Italy and the highest level of the Italian football league system.The winners are awarded the Coppa Campioni d'Italia trophy and the scudetto, a decoration that they wear on the ..., title: Serie A - Wikipedia, link: https://en.wikipedia.org/wiki/Serie_A, snippet: Inter Milan won Serie A on Monday after beating AC Milan 2-1 and creating an unassailable lead at the top of the league with their sixth straight derby victory., title: Inter Milan win Serie A title in derby thriller with AC Milan - France 24, link: https://www.france24.com/en/live-news/20240422-inter-milan-win-serie-a-title-in-derby-thriller-with-ac-milan, snippet: Inter have won the Serie A title after their 2-1 victory over rivals AC Milan in the Derby della Madonnina. Inter took the lead in the 18th minute after defender Benjamin Pavard flicked on a ..., title: Inter win Serie A title after 2-1 victory over rivals AC Milan, link: https://www.nytimes.com/athletic/5436791/2024/04/22/inter-win-serie-a-title/AC Milan won the Serie A in the 2021-22 season.  Inter Milan won the 2023-24 season.

> Finished chain.
Output: AC Milan won the Serie A in the 2021-22 season.  Inter Milan won the 2023-24 season.

Input 2: What is the capital of Italy?


> Entering new AgentExecutor chain...

Invoking: `search_wikipedia` with `{'query': 'Italy'}`


Page: Italy
Summary: Italy, officially the Italian Republic, is a country in Southern and Western Europe. It consists of a peninsula that extends into the Mediterranean Sea, with the Alps on its northern land border, as well as nearly 800 islands, notably Sicily and Sardinia. Italy shares land borders with France to the west; Switzerland and Austria to the north; Slovenia to the east; and the two enclaves of Vatican City and San Marino. It is the tenth-largest country in Europe by area, covering 301,340 km2 (116,350 sq mi), and the third-most populous member state of the European Union, with nearly 60 million inhabitants. Italy's capital and largest city is Rome; other major urban areas include Milan, Naples, Turin, Palermo, Bologna, Florence, Genoa, and Venice.
The history of Italy goes back to numerous Italic peoples—notably including the ancient Romans, who conquered the Mediterranean world during the Roman Republic and ruled it for centuries during the Roman Empire. With the spread of Christianity, Rome became the seat of the Catholic Church and the Papacy. Barbarian invasions and other factors led to the decline and fall of the Western Roman Empire between late antiquity and the Early Middle Ages. By the 11th century, Italian city-states and maritime republics expanded, bringing renewed prosperity through commerce and laying the groundwork for modern capitalism. The Italian Renaissance flourished during the 15th and 16th centuries and spread to the rest of Europe. Italian explorers discovered new routes to the Far East and the New World, contributing significantly to the Age of Discovery.
After centuries of political and territorial divisions, Italy was almost entirely unified in 1861, following wars of independence and the Expedition of the Thousand, establishing the Kingdom of Italy. From the late 19th to the early 20th century, Italy rapidly industrialised—mainly in the north—and acquired a colonial empire, while the south remained largely impoverished, fueling a large immigrant diaspora to the Americas. From 1915 to 1918, Italy took part in World War I with the Entente against the Central Powers. In 1922, the Italian fascist dictatorship was established. During World War II, Italy was first part of the Axis until its surrender to the Allied powers (1940–1943), then a co-belligerent of the Allies during the Italian resistance and the liberation of Italy (1943–1945). Following the war, the monarchy was replaced by a republic and the country enjoyed a strong recovery.
A developed country with an advanced economy, Italy has the ninth-largest nominal GDP in the world, the second-largest manufacturing sector in Europe, and plays a significant role in regional and—to a lesser extent—global economic, military, cultural, and political affairs. Italy is a founding and leading member of the European Union, and  is part of numerous other  international organizations and forums. As a cultural superpower, Italy has long been a renowned global centre of art, music, literature, cuisine, fashion, science and technology, and the source of multiple inventions and discoveries. It has the highest number of World Heritage Sites (60) and is the fourth-most visited country in the world.



Page: Italians
Summary: Italians (Italian: italiani, pronounced [itaˈljaːni]) are an ethnic group native to the Italian geographical region. Italians share a common culture, history, ancestry and language. Their predecessors differ regionally, but generally include native populations such as the Etruscans, Rhaetians, Ligurians, Adriatic Veneti, and Italic peoples, including Latins, from which Romans emerged and helped create and evolve the modern Italian identity. Foreign influences include the ancient Greeks in Magna Graecia, and the Phoenicians, who had a presence in Sicily and Sardinia, the Celts, who settled in parts of the north, the Germanics and the Slavs. Legally, Italian nationals are citizens of Italy, regardless of ancestry or nation of residencRome is the capital of Italy.

> Finished chain.
Output: Rome is the capital of Italy.

Input 3: Tell me about the wikipedia page for Cats


> Entering new AgentExecutor chain...

Invoking: `search_wikipedia` with `{'query': 'Cats'}`


Page: Cat
Summary: The cat (Felis catus), also referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal. It is the only domesticated species of the family Felidae. Advances in archaeology and genetics have shown that the domestication of the cat occurred in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges freely as a feral cat avoiding human contact. It is valued by humans for companionship and its ability to kill vermin. Its retractable claws are adapted to killing small prey species such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth, and its night vision and sense of smell are well developed. It is a social species, but a solitary hunter and a crepuscular predator.
Cat intelligence is evident in their ability to adapt, learn through observation, and solve problems, with research showing they possess strong memories, exhibit neuroplasticity, and display cognitive skills comparable to a young child. Cat communication includes meowing, purring, trilling, hissing, growling, grunting, and body language. It can hear sounds too faint or too high in frequency for human ears, such as those made by small mammals. It secretes and perceives pheromones.
Female domestic cats can have kittens from spring to late autumn in temperate zones and throughout the year in equatorial regions, with litter sizes often ranging from two to five kittens. Domestic cats are bred and shown at cat fancy events as registered pedigreed cats. Population control includes spaying and neutering, but pet abandonment has exploded the global feral cat population, which has driven the extinction of bird, mammal, and reptile species.
Domestic cats are found across the globe, though their popularity as pets varies by region. Out of the estimated 600 million cats worldwide, 400 million reside in Asia, including 58 million pet cats in China. The United States leads in cat ownership with 73.8 million cats despite having a significantly smaller human population. In the United Kingdom, approximately 10.9 million domestic cats are kept as pets.

Page: Cats (musical)
Summary: Cats is a sung-through musical with music by Andrew Lloyd Webber. It is based on the 1939 poetry collection Old Possum's Book of Practical Cats by T. S. Eliot. The musical tells the story of a tribe of cats called the Jellicles and the night they make the "Jellicle choice" by deciding which cat will ascend to the Heaviside Layer and come back to a new life. As of 2024, Cats remains the fifth-longest-running Broadway show and the eighth-longest-running West End show.
Lloyd Webber began setting Eliot's poems to music in 1977, and the compositions were first presented as a song cycle in 1980. Producer Cameron Mackintosh then recruited director Trevor Nunn and choreographer Gillian Lynne to turn the songs into a complete musical. Cats opened to positive reviews at the New London Theatre in the West End in 1981 and then to mixed reviews at the Winter Garden Theatre on Broadway in 1982. It won numerous awards including Best Musical at both the Laurence Olivier and Tony Awards. Despite its unusual premise that deterred investors initially, the musical turned out to be an unprecedented commercial success, with a worldwide gross of US$3.5 billion by 2012.
The London production ran for 21 years and 8,949 performances, while the Broadway production ran for 18 years and 7,485 performances, making Cats the longest-running musical in both theatre districts for a number of years. Cats has since been revived in the West End twice and on Broadway once. It has also been translated into multiple languages and performed around the world many times. Long-running foreign productions include a 15-year run at the Operettenhaus in Hamburg that played over 6,100 performances, as well as an ongoing run in a purpose-built theatre in Japan that has played over 10,000 performances since it opened in 1983.
Cats started the megamBased on my search of Wikipedia, there are two main pages that come up when searching for "Cats": one about the animal and one about the musical.  The animal page details the domestic cat's biology, behavior, history, and global distribution.  The musical page discusses Andrew Lloyd Webber's adaptation of T.S. Eliot's poems into a stage production, its history, and its lasting impact on the theater world.

> Finished chain.
Output: Based on my search of Wikipedia, there are two main pages that come up when searching for "Cats": one about the animal and one about the musical.  The animal page details the domestic cat's biology, behavior, history, and global distribution.  The musical page discusses Andrew Lloyd Webber's adaptation of T.S. Eliot's poems into a stage production, its history, and its lasting impact on the theater world.
'''