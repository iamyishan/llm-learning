# The function name, type hints, and docstring are all part of the tool
# schema that's passed to the model. Defining good, descriptive schemas
# is an extension of prompt engineering and is an important part of
# getting models to perform well.
from dotenv import load_dotenv

load_dotenv()

from src.llm.zhipu_llm import llm

from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


def test1() -> str:
    tools = [add, multiply]
    llm_with_tools = llm.bind_tools(tools)

    query = "What is 3 * 12?"

    res = llm_with_tools.invoke(query)
    print(res)
    print(res.tool_calls)



def test2():
    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import HumanMessage

    query = "What is 3 * 12? Also, what is 11 + 49?"

    messages = [HumanMessage(query)]

    ai_msg = llm_with_tools.invoke(messages)

    print(ai_msg.tool_calls)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    print(messages)
    res=llm_with_tools.invoke(messages)
    print(res)



if __name__ == '__main__':
    # test1()
    test2()