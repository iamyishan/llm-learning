from langchain_community.llms import Ollama
# from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = Ollama(base_url="http://localhost:8000", model="qwen2:7b", verbose=True)

prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:
<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)


def test1():
    """自定义文档内容"""

    from langchain_core.documents import Document
    docs = [Document(page_content="Langsmith可以帮助你可视化测试结果")]
    response = document_chain.invoke({
        "input": "langsmith如何辅助测试",
        "context": docs
    })
    print(response)


def test2():
    """ 从网页加载文档内容"""
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://bbs.csdn.net/topics/618378840")
    docs = loader.load()
    response = document_chain.invoke({
        "input": "langsmith如何辅助测试",
        "context": docs
    })
    print(response)


def test3():
    """ 从网页加载文档内容"""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("LangSmith.pdf")
    docs = loader.load()

    response = document_chain.invoke({
        "input": "langsmith如何辅助测试",
        "context": docs
    })
    print(response)


if __name__ == '__main__':
    # test1()
    # test2()
    test3()