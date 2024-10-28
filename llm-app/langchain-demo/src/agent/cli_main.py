# agent入门
"""
todo:
    1.环境变量的设置
    2.工具的引入
    3.promote模板
    4.模型的初始化
"""
import time


def parse_thoughts(response):
    """
    response:
    {
        "action"{
            "name":"action name",
            "args":{
                "arg name":"arg value"
                }

             }
        "thoughts":{
            "text":"thought",
            "plan":"plan",
            "criticism":"criticism",
            "speak":"speak",
            "reasoning":""
            }
    }
    """
    try:
        thoughts = response.get("thoughts")
        observation = response.get("speak")
        plan = thoughts.get("plan")
        reasoning = thoughts.get("reasoning")
        criticism = thoughts.get("criticism")
        prompt = f"paln:{plan}\nreasoning:{reasoning}\ncriticism:{criticism}\nobservation:{observation}"
        return prompt
    except Exception as err:
        print("解析thoughts异常：", err)
        return "".format(err)


def agent_execute(query, max_request_time=10):
    cur_request_time = 0
    chat_history = []
    agent_scratch = ""
    while cur_request_time < max_request_time:
        cur_request_time += 1
        """
        如果返回结果达到预期，则直接返回
        """
        """
        prompt包含的功能:
            1.任务描述
            2.工具描述
            3.用户的输入user_msg
            4.assistant_msg
            5. 限制
            6.给出更好实践的描述
        """
        prompt = gen_prompt(query, chat_history, agent_scratch)
        print("**********************{}.开始调用大模型............".format(cur_request_time), flush=True)
        start_time = time.time()
        response = call_llm()
        end_time = time.time()
        print("**********************{}.调用大模型结束，耗时：{}".format(cur_request_time, end_time - start_time), flush=True)
        if not response or not isinstance(response, dict):
            print("调用大模型错误，即将重试...", response)
            continue
        """
        response:
        {
            "action"{
                "name":"action name",
                "args":{
                    "arg name":"arg value"
                    }
            
                 }
            "thoughts":{
                "text":"thought",
                "plan":"plan",
                "criticism":"criticism",
                "speak":"speak",
                "reasoning":""
                }
        }
        
        """
        action_info = response.get("action")
        action_name = action_info.get("name")
        action_args = action_info.get("args")
        print("当前action name: ", action_name, action_args)
        if action_name == "finish":
            finish_answer = action_args.get("answer")
            print("finish answer: ", finish_answer)
            break
        try:
            """
            action_name到函数的映射：map->{action_name:func}
            """
            tools_map = {}
            func = tools_map.get(action_name)
            observation = func(**action_args)
        except Exception as err:
            print("调用工具异常：", err)
        agent_scratch = agent_scratch + "\n" + observation
        user_msg = "决定使用哪个工具"
        assistant_msg = parse_thoughts(response)

        chat_history.append({"role": "user", "content": user_msg})

    print(query)


def main():
    # 需求：支持用户的多次输入
    max_request_time = 10
    while True:
        query = input("请输入你的问题：")
        if query == "exit":
            return
        agent_execute(query, max_request_time=max_request_time)


if __name__ == '__main__':
    main()
