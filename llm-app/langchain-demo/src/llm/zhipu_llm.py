import os

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

os.environ["ZHIPUAI_API_KEY"] = "5d9cb7c8e7fd91d037521bf22bf1f986.vb8QSdN7Hi0V0lBt"

llm = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.5,
)

if __name__ == '__main__':
    messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="Your role is a poet."),
        HumanMessage(content="Write a short poem about AI in four lines."),
    ]
    response = llm.invoke(messages)
    print(response.content)  # Displays the AI-generated poem
