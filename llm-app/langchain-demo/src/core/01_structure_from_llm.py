from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm.zhipu_llm import llm


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke)


def test1():
    """模型返回一个 Pydantic 对象, Pydantic 的主要优势在于模型生成的输出将被验证"""
    res = structured_llm.invoke("Tell me a joke about cats")
    print(type(res))
    print(res)


def test2():
    """TypedDict 或 JSON Schema:
    不想使用 Pydantic，明确不想验证参数，或者希望能够流式传输模型输出"""
    # Pydantic
    from typing_extensions import TypedDict, Annotated
    # TypedDict
    class Joke(TypedDict):
        """Joke to tell user."""

        setup: Annotated[str, ..., "The setup of the joke"]

        # Alternatively, we could have specified setup as:

        # setup: str                    # no default, no description
        # setup: Annotated[str, ...]    # no default, no description
        # setup: Annotated[str, "foo"]  # default, no description

        punchline: Annotated[str, ..., "The punchline of the joke"]
        rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

    structured_llm = llm.with_structured_output(Joke)

    # res = structured_llm.invoke("Tell me a joke about cats")
    # print(type(res))
    # print(res)

    # 流式输出
    for chunk in structured_llm.stream("Tell me a joke about cats"):
        print(chunk)


def test3():
    """在多个模式之间进行选择"""
    from typing import Union

    # Pydantic
    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )

    class ConversationalResponse(BaseModel):
        """Respond in a conversational manner. Be kind and helpful."""

        response: str = Field(description="A conversational response to the user's query")

    class FinalResponse(BaseModel):
        final_output: Union[Joke, ConversationalResponse]

    structured_llm = llm.with_structured_output(FinalResponse)

    res = structured_llm.invoke("Tell me a joke about cats")
    print(type(res))
    print(res)


def test4():
    """少样本提示"""
    from langchain_core.prompts import ChatPromptTemplate

    system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
    Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

    Here are some examples of jokes:

    example_user: Tell me a joke about planes
    example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

    example_user: Tell me another joke about planes
    example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

    example_user: Now about caterpillars
    example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

    few_shot_structured_llm = prompt | structured_llm
    res = few_shot_structured_llm.invoke("what's something funny about woodpeckers")
    print(res)


def test5():
    """工具的方式传入样例"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    examples = [
        HumanMessage("Tell me a joke about planes", name="example_user"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "Why don't planes ever get tired?",
                        "punchline": "Because they have rest wings!",
                        "rating": 2,
                    },
                    "id": "1",
                }
            ],
        ),
        # Most tool-calling models expect a ToolMessage(s) to follow an AIMessage with tool calls.
        ToolMessage("", tool_call_id="1"),
        # Some models also expect an AIMessage to follow any ToolMessages,
        # so you may need to add an AIMessage here.
        HumanMessage("Tell me another joke about planes", name="example_user"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "Cargo",
                        "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
                        "rating": 10,
                    },
                    "id": "2",
                }
            ],
        ),
        ToolMessage("", tool_call_id="2"),
        HumanMessage("Now about caterpillars", name="example_user"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "Caterpillar",
                        "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
                        "rating": 5,
                    },
                    "id": "3",
                }
            ],
        ),
        ToolMessage("", tool_call_id="3"),
    ]
    system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
    Return a joke which has the setup (the response to "Who's there?") \
    and the final punchline (the response to "<setup> who?")."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("placeholder", "{examples}"), ("human", "{input}")]
    )
    few_shot_structured_llm = prompt | structured_llm
    print(few_shot_structured_llm.invoke({"input": "crocodiles", "examples": examples}))

def test6():
    from typing import List

    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        """Information about a person."""

        name: str = Field(..., description="The name of the person")
        height_in_meters: float = Field(
            ..., description="The height of the person expressed in meters."
        )

    class People(BaseModel):
        """Identifying information about all people in a text."""

        people: List[Person]

    # Set up a parser
    parser = PydanticOutputParser(pydantic_object=People)

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    query = "Anna is 23 years old and she is 6 feet tall"
    print(prompt.invoke(query).to_string())
    chain = prompt | llm | parser
    res=chain.invoke({"query": query})
    print(res)
if __name__ == '__main__':
    test6()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
