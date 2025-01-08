from src.llm.zhipu_llm import llm

chunks = []
for chunk in llm.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content)
