import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 加载embedding模型，用于将chunk向量化，word2vec
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')


def save_local():
    """保存向量数据库"""
    # 解析PDF，切成chunk片段,# 使用OCR解析pdf中图片里面的文字
    pdf_loader = PyPDFLoader('LLM.pdf', extract_images=True)
    # chunk_size太长会导致检索不准，chunk_overlap是重叠的字符数，用于解决切分问题
    chunks = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))
    for chunk in chunks:
        print(chunk)
    # 将chunk插入到faiss本地向量数据库
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local('LLM.faiss')
    print('faiss saved!')


def load_local():
    """检索向量数据库"""
    vector_db = FAISS.load_local('LLM.faiss', embeddings, allow_dangerous_deserialization=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 执行相似性搜索
    query = "transformer原理"
    results = vector_db.similarity_search(query)
    # 打印结果
    for doc in results:
        print(doc)
        print(doc.page_content)



if __name__ == '__main__':
    load_local()
    # save_local()
