import requests
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from dotenv import load_dotenv
import json
import warnings
from datetime import datetime  # Add this import if not already present
warnings.filterwarnings('ignore')

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
CX_KEY = os.getenv("CX_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# 1) Define a custom tool for Tavily
@tool("tavily_search", return_direct=False)
def tavily_search(query: str) -> str:
    """
    Tavily Search tool:
    This tool performs a search using Tavily's API and returns relevant results.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "include_answer": True,
        "search_depth": "advanced"
    }
    headers = {
        "content-type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the JSON response
        result = response.json()
        
        # Extract the answer or results
        if 'answer' in result:
            return result['answer']
        elif 'results' in result:
            # Format the first few search results
            formatted_results = []
            for item in result['results'][:3]:  # Limit to top 3 results
                formatted_results.append(f"Title: {item.get('title', 'N/A')}\nContent: {item.get('content', 'N/A')}\n")
            return "\n".join(formatted_results)
        else:
            return "No results found"
            
    except requests.RequestException as e:
        return f"Error with Tavily API: {str(e)}"
    except json.JSONDecodeError:
        return f"Error parsing Tavily response: {response.text}"

@tool("google_search", return_direct=False)
def google_search(query: str) -> str:
    """
    Google Search tool:
    This tool performs a search using Google's Custom Search API and returns relevant results.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Add current date to the query to force recent results
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_year = datetime.now().strftime('%Y')
    query = f"{query} {current_year}"
    
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": CX_KEY,
        "q": query,
        # "dateRestrict": "m1",  # Restrict to last month
        "sort": "date"  # Sort by date
    }
    
    try:
        print(f"Sending request to Google with query: {query}")
        response = requests.get(url, params=params)
        
        if response.status_code == 400:
            print(f"Response content: {response.text}")
            return f"Error: Bad request. Response: {response.text}"
        
        response.raise_for_status()
        result = response.json()
        
        if 'items' in result:
            formatted_results = []
            for item in result['items'][:3]:
                title = item.get('title', 'N/A')
                snippet = item.get('snippet', 'N/A')
                formatted_results.append(f"Title: {title}\nContent: {snippet}\n")
            return "\n".join(formatted_results)
        else:
            return "No results found"
            
    except requests.RequestException as e:
        print(f"Full error details: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        return f"Error with Google API: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing Google response: {str(e)}"

@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """
    News Search tool:
    This tool performs a search using NewsAPI to get the latest news articles.
    """
    url = "https://newsapi.org/v2/everything"
    
    # Format current date for the API
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_year = datetime.now().strftime('%Y')
    
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query + " " + current_year,
        "from": current_date,
        "sortBy": "publishedAt",  # Sort by publication date
        "language": "en"  # English articles only
    }
    
    try:
        print(f"Sending request to NewsAPI with query: {query}")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Response content: {response.text}")
            return f"Error: NewsAPI request failed. Status code: {response.status_code}"
        
        result = response.json()
        
        if result.get('articles'):
            formatted_results = []
            for article in result['articles'][:3]:  # Get top 3 articles
                title = article.get('title', 'N/A')
                description = article.get('description', 'N/A')
                published = article.get('publishedAt', 'N/A')
                formatted_results.append(
                    f"Title: {title}\n"
                    f"Date: {published}\n"
                    f"Content: {description}\n"
                )
            return "\n".join(formatted_results)
        else:
            return "No news articles found"
            
    except requests.RequestException as e:
        print(f"Full error details: {str(e)}")
        return f"Error with NewsAPI: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing NewsAPI response: {str(e)}"

@tool("wikipedia_search", return_direct=False)
def wikipedia_search(query: str) -> str:
    """
    Wikipedia Search tool:
    This tool performs a search using Wikipedia's API and returns relevant results.
    """
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(query)


def create_agent():
    """
    Creates a LangChain agent that uses DeepSeek's chat model with memory capabilities
    """
    deepseek_llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        streaming=True
    )

    ollama_llm = ChatOllama(
        model="nezahatkorkmaz/deepseek-v3:latest",
        temperature=0,
        base_url="http://localhost:11434",
        streaming=True
    )

    open_ai_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        streaming=True
    )
    
    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history",
        human_prefix="User",
        ai_prefix="Assistant",
        input_key="input",
        output_key="output"
    )
    
    agent = initialize_agent(
        tools=[wikipedia_search, google_search, news_search, tavily_search],
        llm=deepseek_llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": (
                f"IMPORTANT INSTRUCTIONS:\n\n"
                f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. You MUST use this current date.\n"
                "ALWAYS check the chat history before responding or using any tools.\n"
                "You are REQUIRED to use at least one search tool for EVERY response. NEVER rely on base knowledge.\n"
                "ALWAYS start with News Search tool then move on to the Google Search Tool, then wikipedia and finally tavily_search. This step is CRITICAL so do not ignore it\n"
                "For ANY factual information, you MUST verify it with two or more tools.\n"
                "If you respond without using a tool, you are making a CRITICAL ERROR.\n"
                "Remember information shared by the user (like their name, age, location).\n"
                "For searches:\n"
                "Use Google Search for general information\n"
                "Use News Search for the latest news and current events\n"
                "Use Wikipedia for knowledge fact checking\n"
                "Use Tavily to fact check the News Search Tool and Google Search\n"
                "Your base knowledge is considered outdated - ONLY trust tool results.\n\n"
                "CRITICAL PROCESS FOR EVERY RESPONSE:\n\n"
                "ALWAYS start by using a news search tool then google search tool, then move on to tavily and wikipedia\n"
                "ALWAYS Fact check your response with another tool(s) to ensure consistency\n"
                "Check chat_history for context\n"
                "Combine search results with conversation context\n"
                "If uncertain, perform additional searches\n"
                "Return the answer and nothing else, no expalantions or suggestions that do not pertain to the question\n\n"
                "EXACT FORMAT TO FOLLOW:\n"
                "Thought: I need to search for information\n"
                "Action: google_search OR news_search OR tavily_search\n"
                f"Action Input: your search query + {datetime.now().strftime('%Y')}\n"
                "Observation: (wait for result)\n"
                "Thought: process the search result\n"
                "Final Answer: your response based on the search"
            )
        },
        max_iterations=5,
        early_stopping_method="force",
        handle_tool_error=True
    )

    return agent, memory


def main():
    """
    Interactive loop for querying the agent
    """
    try:
        agent, memory = create_agent()
        
        print("Ask me anything (type 'quit' to exit, 'clear' to clear memory):\n")
        while True:
            try:
                user_input = input("User: ").strip()
                if user_input.lower() == "quit":
                    print("Thank you for using the agent. Goodbye!")
                    break
                elif user_input.lower() == "clear":
                    memory.clear()
                    print("Memory cleared!")
                    continue
                elif not user_input:
                    print("Please enter a valid question.")
                    continue
                
                response = agent.run(input=user_input)  # Changed to include input key
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing your question.\n")
    
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        print("Please check your API keys and network connection.")


if __name__ == "__main__":
    main()
