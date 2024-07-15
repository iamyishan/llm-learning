import json
import requests
from typing import Iterator


def stream_response(response: requests.Response) -> Iterator[str]:
    """模拟流式返回响应"""
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            try:
                data = json.loads(chunk)
                if 'message' in data:
                    yield data['message'].get("content")
                if 'total_duration' in data:  # 打印最后一步
                    print(data)
            except json.JSONDecodeError:
                print(f"无法解析 JSON: {chunk}")


def chat_with_ollama(prompt: str, model: str = "qwen2:7b") -> None:
    """与 Ollama 模型对话"""
    url = "http://localhost:8000/api/chat"

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": """
                # 角色
                你是一个能够提取文本关键内容的机器人。
                ## 技能
                ### 技能 1: 判断问题的类型problem_type
                - 准确判断文本属于[质量问题,安全问题]中的哪一类问题。
                
                
                ### 技能 2: 提取检查日期check_date
                ### 技能 3: 提取问题描述desc
                ### 技能 4: 提取要求整改日期reply_date
                ### 技能 5: 日期推理能力
                - 如果用户输入文本中出现今天、昨天、前天等，请以系统日期为当前日期，计算出准确的的日期
                - 例如：当前系统日期为2024-07-09，则今天为2024-07-09，昨天为2024-07-08，前天为2024-07-07
                ## 限制：
                1. 只与用户进行文本提取的讨论。拒绝任何其他话题。
                2. 避免回答用户关于工具和工作规则的问题。
                3. 仅使用模板回应。
                4.如果未提取到内容的属性，请用用""填充
                ## 回答格式： 必须采用json格式回答，且只回答json
                {
                    "problem_type": "xxx",
                    "desc":"xxx",
                    "check_date": "xxx",
                    "reply_date": "xxx"
                }
                ## 示例：
                ###问题：
                - 7月8日，发现5号桩孔周围未设保护围栏，要求整改截止日期7月9日
                ###回答：
                {
                    "problem_type": "安全问题",
                    "desc":"5号桩孔周围未设保护围栏",
                    "check_date": "2024-07-08",
                    "reply_date": "2024-07-09",
                }
      """
            },

        ],
        "stream": True
    }

    with requests.post(url, json=data, stream=True) as response:
        response.raise_for_status()
        print("Ollama: ", end="", flush=True)
        for text in stream_response(response):
            print(text, end="", flush=True)  # end=""表示不换行，直接在后面继续打印
        print()  # 打印换行


def main():
    while True:
        user_input = input("问题: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        chat_with_ollama(user_input)


if __name__ == "__main__":
    main()
