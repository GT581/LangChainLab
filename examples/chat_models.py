"""
Examples of using Chat Models with Langchain.

This module demonstrates different ways to use Chat Models in Langchain:
1. Basic chat with system and human messages
2. Using multiple messages in a conversation
3. Using streaming for real-time responses

Documentation:
- https://python.langchain.com/docs/concepts/chat_models/
- https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html
"""
import os
import sys
import time

from langchain.schema import HumanMessage, SystemMessage, AIMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_utils import get_llm


def basic_chat_example() -> None:
    """
    Basic example of using a chat model with Langchain system and human messages.
    """
    print("\n=== Basic Chat Example ===")
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in explaining complex topics in a simple way."),
        HumanMessage(content="Explain Large Language Models to me in 5 sentences.")
    ]
    
    # Get response with created System / Human messages
    response = llm.invoke(messages)
    
    print(f"Response: {response.content}")


def multi_turn_conversation_example() -> None:
    """
    Example of a multi-turn conversation with a chat model.
    """
    print("\n=== Multi-turn Conversation Example ===")
    
    llm = get_llm()
    
    # Create the initial messages for the conversation
    messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in world geography."),
        HumanMessage(content="What are the boroughs of London, England??"),
    ]
    
    response = llm.invoke(messages)
    print(f"AI: {response.content}")
    
    # Add AI response, and a follow-up question to the conversation
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="What soccer teams in the premier league are from each borough?"))
    
    response = llm.invoke(messages)
    print(f"AI: {response.content}")
    
    # Add second response, and another follow-up
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="What are the biggest rivalries in these areas?"))
    
    response = llm.invoke(messages)
    print(f"AI: {response.content}")


def streaming_example() -> None:
    """
    Example of streaming responses from a chat model.
    """
    print("\n=== Streaming Example ===")
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in creating stories."),
        HumanMessage(content="Tell me a short story about someone learning to golf.")
    ]
    
    # Stream the response
    print("Streaming response:")
    response_stream = llm.stream(messages)
    
    for chunk in response_stream:
        print(chunk.content, end="", flush=True)
        time.sleep(.5)  # Simulate delay with sleep for better visualization
    
    print("\n")


if __name__ == "__main__":
    basic_chat_example()
    multi_turn_conversation_example()
    streaming_example()


'''
Output:

=== Basic Chat Example ===
Response: Large language models (LLMs) are computer programs trained on massive amounts of text data to understand and generate human-like text.  They learn patterns and relationships in the data, allowing them to predict the next word in a sentence or even generate entire paragraphs.  Think of them as incredibly sophisticated autocomplete systems, but on a much larger scale.  This ability enables them to translate languages, answer questions, and even write creative content.  However, they don't truly "understand" the meaning in the way humans do; they're mimicking patterns they've learned.

=== Multi-turn Conversation Example ===
AI: London is divided into 32 boroughs and the City of London.  The City of London is a separate administrative area, historically and currently distinct from the boroughs.  Therefore, there aren't just "boroughs of London," but also the City of London.  Listing all 32 boroughs would be quite lengthy, but here are some examples:

* **Inner London Boroughs:**  These are generally closer to the centre. Examples include Westminster, Kensington and Chelsea, Camden, Islington, etc.

* **Outer London Boroughs:** These are further from the centre. Examples include Barnet, Bromley, Croydon, Harrow, etc.

To get a complete list, you can easily search online for "London boroughs list" and find a comprehensive list with maps.
AI: It's not a simple one-to-one mapping of Premier League teams to London boroughs.  Many Premier League teams' stadiums are located in one borough, but their fanbase and operational reach extend far beyond those boundaries.  Also, team locations can change over time.

To give you a more accurate answer, I need to specify that I'm referring to the borough where their *stadium* is located, as of the 2023-2024 season.  Even this is subject to change if a team moves stadiums.

Here's what I can offer, keeping in mind this limitation:

* **Brentford:** Brentford (Brentford FC's stadium is in Brentford, Hounslow borough)
* **Chelsea:** Fulham (Chelsea FC's stadium is in Fulham, Hammersmith and Fulham borough)
* **Crystal Palace:** Croydon (Crystal Palace's stadium is in Selhurst, Croydon borough)
* **Arsenal:** Islington (Arsenal's stadium, the Emirates Stadium, is in Islington)
* **Tottenham Hotspur:** Haringey (Tottenham Hotspur Stadium is in Haringey)
* **West Ham United:** Newham (West Ham's stadium, the London Stadium, is in Newham)


It's important to note that this is not exhaustive of *all* Premier League teams based in London, as the league's composition changes yearly.  Always check the current Premier League standings and team information for the most up-to-date details.
AI: The biggest rivalries involving the London Premier League teams you mentioned are largely defined by proximity and historical context:

* **Arsenal vs. Tottenham Hotspur (North London Derby):** This is arguably the fiercest and most intense rivalry in London, and possibly even in all of English football.  The close proximity of the two clubs in North London fuels the passion, and the history of the rivalry is long and storied.

* **Chelsea vs. Tottenham Hotspur:** While not as historically significant as the North London Derby, this rivalry has intensified in recent years due to the teams' competitive positions and proximity. 

* **Chelsea vs. Arsenal:** Another significant rivalry, fueled by the teams' consistent presence at the top of the Premier League and their geographical proximity.

* **West Ham United vs. Millwall (East London Derby):**  While Millwall isn't currently in the Premier League, this is a significant and often fiercely contested rivalry rooted in the working-class history of East London.  The rivalry extends beyond football and into the broader social fabric of the communities.

* **Other Rivalries:**  Rivalries also exist between other London clubs, though perhaps less intense or consistently featured in the Premier League at the same time.  For example, Fulham and Chelsea have a local rivalry, as do other teams depending on their league position and historical context.      


It's important to note that the intensity of these rivalries can fluctuate depending on the teams' current performance and league standings.  However, the ones listed above are consistently considered the most significant and historically important.

=== Streaming Example ===
Streaming response:
Agnes clutched the driver like a lifeline, her knuckles white against the polished wood.  The pristine green stretched before her, a vast, mocking expanse of perfectly manicured grass.  At 62, Agnes had decided golf was her next adventure, a rebellion against the quiet predictability of retirement.  Her grandson, Leo, a lanky teenager with more patience than skill, was her reluctant instructor.

Their first lesson was a disaster.  Agnes's swing resembled a windmill caught in a hurricane, sending clumps of turf flying in unpredictable directions.  The balls, when they actually made contact with the club, veered wildly off course, disappearing into the rough or, on one memorable occasion, into a nearby sand trap with a satisfying *plunk*.

Leo, initially exasperated, gradually softened.  He saw the fierce determination in Agnes's eyes, the stubborn refusal to give up.  He started breaking down her swing into smaller, manageable parts, focusing on her stance, her grip, the follow-through.  He even made her practice with a plastic club in the backyard, much to the amusement of their neighbour's poodle.

Weeks turned into months.  Agnes's progress was slow, painstaking.  There were days of frustration, of muttered curses under her breath, of balls sailing into the water hazard with depressing regularity.  But there were also moments of triumph – a perfectly struck ball soaring high and true, a putt sinking into the hole with a satisfying click.

One crisp autumn afternoon, Agnes stood on the 18th tee, her heart pounding a rhythm of nervous excitement.  The sun cast long shadows across the course.  She took a deep breath, remembering Leo's instructions, feeling the familiar weight of the club in her hands.  She swung.

The ball flew.  It arced gracefully through the air, a tiny white speck against the vast blue sky, landing softly on the green, just a few feet from the hole.  Agnes smiled, a slow, contented smile that reached her eyes.  It wasn't a hole-in-one, but it was perfect.  It was hers.  And in that moment, the score didn't matter.  She had conquered the green, one frustrating, exhilarating swing at a time.
'''