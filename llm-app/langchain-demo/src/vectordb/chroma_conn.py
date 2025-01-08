import chromadb

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ai.common import model_emb


class ChromaDB:
    def __init__(self,
                 chroma_server_type="local",  # 服务器类型：http是http方式连接方式，local是本地读取文件方式
                 host="localhost", port=8000,  # 服务器地址，http方式必须指定
                 collection_name="ikanjian",  # 数据库的集合名称
                 persist_path="chroma_db",  # 数据库的路径：如果是本地连接，需要指定数据库的路径
                 embed=None  # 数据库的向量化函数
                 ):

        self.host = host
        self.port = port
        self.path = persist_path
        self.embed = embed
        self.store = None

        # 如果是http协议方式连接数据库
        if chroma_server_type == "http":
            client = chromadb.HttpClient(host=host, port=port)

            self.store = Chroma(collection_name=collection_name,
                                embedding_function=embed,
                                client=client)

        if chroma_server_type == "local":
            self.store = Chroma(collection_name=collection_name,
                                embedding_function=embed,
                                persist_directory=persist_path)

        if self.store is None:
            raise ValueError("Chroma store init failed!")

    def add(self, docs):
        """
        将文档添加到数据库
        """
        self.store.add_documents(documents=docs)

    def get_store(self):
        """
        获得向量数据库的对象实例
        """
        return self.store


if __name__ == '__main__':
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"source": "fish-pets-doc"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"source": "bird-pets-doc"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
    vectordb = ChromaDB(embed=model_emb)
    vectordb.add(documents)
    retriever = vectordb.get_store().as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"k": 4, "score_threshold": 0.1})
    print(retriever)
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    runnable = RunnableParallel(
        passed=RunnablePassthrough(),
        modified=lambda x: x["num"] + 1,
    )

    print(runnable.invoke({"num": 1}))
