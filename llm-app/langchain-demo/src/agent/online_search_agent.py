from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from src.llm.zhipu_llm import llm

load_dotenv()


def tavily_search_results():
    search = TavilySearchResults(max_results=2)
    search_results = search.invoke("what is the weather in SF")
    print(search_results)
    # If we want, we can create other tools.
    # Once we have all the tools we want, we can put them in a list that we will reference later.


from langgraph.prebuilt import create_react_agent

search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(llm, tools)

# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
if __name__ == '__main__':
    print(response["messages"])
