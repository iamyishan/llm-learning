from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

llm = Ollama(base_url="http://localhost:8000", model="qwen2:7b", verbose=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深的软件开发工程师"),
    ("user", "{input}")
])
chain = prompt | llm | output_parser
response = chain.invoke({"input": "请帮我写一个python代码，实现一个简单的http服务器，能够接收post请求，并将请求体中的json数据解析出来，然后返回一个json响应"})

if __name__ == '__main__':
    print(response)
