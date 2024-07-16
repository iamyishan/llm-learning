from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(base_url="http://localhost:8000", model="qwen2:7b", temperature=0.9)

# LangChain相关模块的导入
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate


def test1_llm_chain():
    """
    LLMChain是一个整合语言模型和提示模板的最简单链。

    :return:
    """
    # 根据prompt模板生成prompt实例
    prompt = ChatPromptTemplate.from_template(
        "请给生产: {product} 的工厂起一个恰当的厂名，并给出一句广告语。"
    )
    # 组合大模型实例和prompt实例，生成LLMChain实例，将结构固定，方便复用
    chain = LLMChain(
        # 大模型实例
        llm=llm,
        # prompt实例
        prompt=prompt,
        # 开启详细模式，会将大模型调用细节输出到控制台
        verbose=True
    )
    # 通过run方法，传入模版中需要的参数，调用大模型获取结果
    product = "IPhone2014"
    res = chain.run(product)
    print(res)


def test2_simple_sequential_chain():
    """
    SimpleSequentialChain:串联式调用语言模型链的一种，简单的串联每个步骤（Chain 实例），每个步骤都有单一的输入/输出，
    并且一个步骤的输入是下一个步骤的输出

    :return:
    """

    '''
    ###################
    ### 第一个Chain ###
    ###################
    '''
    # 第一个LLM请求的prompt模板
    first_prompt = ChatPromptTemplate.from_template(
        "请给生产 {product} 的工厂起一个恰当的厂名"
    )
    # 第一个Chain，接收外部输入，根据模版请求大模型获取输出，作为第二个Chain的输入
    chain_one = LLMChain(llm=llm, prompt=first_prompt, verbose=True)

    '''
    ###################
    ### 第二个Chain ###
    ###################
    '''
    # 第二个大模型请求的prompt模版
    second_prompt = ChatPromptTemplate.from_template(
        "为厂名写一段不少于20字的广告语: {company_name}"
    )
    # 第二个Chain，接收第一个Chain的输出，根据模版请求大模型获取输出
    chain_two = LLMChain(llm=llm, prompt=second_prompt, verbose=True)

    '''
    ##################################
    ### 组建SimpleSequentialChain  ###
    ##################################
    '''
    # 将请求拆分成两个Chain，可以针对每段请求细化相应的prompt内容，得到更准确更合理的结果，并且也可以复用其中的每个Chain实例
    # 使用SimpleSequentialChain将两个Chain串联起来，其中每个Chain都只支持一个输入和一个输出，根据chains列表中的顺序，将前一个Chain的输出作为下一个Chain的输入
    overall_simple_chain = SimpleSequentialChain(
        chains=[chain_one, chain_two],
        verbose=True
    )

    # 第一个Chain需要的输入
    product = "IPhone2014"
    # 通过run方法，传入参数，逐个运行整个Chain后，获取最终的结果
    res = overall_simple_chain.run(product)
    print(res)


def test3_sequential_chain():
    """"
    序列中的每个 Chain 实例都支持多个输入和输出，最终 SequentialChain 运行时根据 Chains 参数和每个 Chain 示例中设定的参数，
    分析每个实例所需的参数并按需传递

     n个任务都会执行！
    【任务1】和【任务3】的输入只需要product_name，各输出一个变量
    【任务2】的输入是【任务1】的输出
    【任务4】的输入是【任务2】和【任务3】的输出
    """

    '''
    ######################################
    ### Chain1 给中文产品名称翻译成英文  ###
    ######################################
    '''
    # Chain1 语言转换，产生英文产品名
    prompt1 = ChatPromptTemplate.from_template(
        "将以下文本翻译成英文: {product_name}"
    )
    chain1 = LLMChain(
        # 使用的大模型实例
        llm=llm,
        # prompt模板
        prompt=prompt1,
        # 输出数据变量名
        output_key="english_product_name",
    )

    '''
    ##################################################
    ### Chain2 根据英文产品名，生成一段英文介绍文本   ###
    ##################################################
    '''
    # Chain2 根据英文产品名，生成一段英文介绍文本
    prompt2 = ChatPromptTemplate.from_template(
        "Based on the following product, give an introduction text about 100 words: {english_product_name}"
    )
    chain2 = LLMChain(
        llm=llm,
        prompt=prompt2,
        output_key="english_introduce"
    )

    '''
    ###########################################
    ### Chain3 产品名的语言判定(中文or英文)   ###
    ###########################################
    '''
    # Chain3 找到产品名所属的语言
    prompt3 = ChatPromptTemplate.from_template(
        "下列文本使用的语言是什么?: {product_name}"
    )
    chain3 = LLMChain(
        llm=llm,
        prompt=prompt3,
        output_key="language"
    )

    '''
    #########################
    ### Chain4 生成概述   ###
    #########################
    '''
    # Chain4 根据Chain2生成的英文介绍，使用产品名称原本的语言生成一段概述
    prompt4 = ChatPromptTemplate.from_template(
        "使用语言类型为: {language} ，为下列文本写一段不多于50字的概述: {english_introduce}"
    )
    chain4 = LLMChain(
        llm=llm,
        prompt=prompt4,
        output_key="summary"
    )

    '''
    ############################
    ### 组建SequentialChain  ###
    ############################
    '''
    # 标准版的序列Chain,SequentialChain,其中每个chain都支持多个输入和输出，
    # 根据chains中每个独立chain对象，和chains中的顺序，决定参数的传递，获取最终的输出结果
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3, chain4],
        input_variables=["product_name"],
        output_variables=["english_product_name", "english_introduce", "language", "summary"],
        verbose=True
    )
    product_name = "重庆小面"
    res = overall_chain(product_name)
    print(res)


from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate


def test4_router_Chain():
    """
    LLMRouterChain 是根据提示词的不同而选择不同的Chain进行执行，实现分支判断的作用

    :return:
    """
    ## 【Step1】初始化语言模型
    # from langchain.llms import OpenAI
    # llm = OpenAI()
    # llm = AzureChatOpenAI(deployment_name="GPT-4", temperature=0)
    ## 【Step2】构建提示信息（json格式），包括：key、description 和 template
    # 【Step2.1】构建两个场景的模板
    flower_care_template = """
    你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
    下面是需要你来回答的问题:
    {input}
    """
    flower_deco_template = """
    你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
    下面是需要你来回答的问题:
    {input}
    """
    # 【Step2.2】构建提示信息
    prompt_infos = [
        {
            "key": "flower_care",
            "description": "适合回答关于鲜花护理的问题",
            "template": flower_care_template,
        },
        {
            "key": "flower_decoration",
            "description": "适合回答关于鲜花装饰的问题",
            "template": flower_deco_template,
        }
    ]
    ## 【Step3】构建目标链chain_map（json格式），以提示信息prompt_infos中的key为key，以Chain为value
    chain_map = {}
    for info in prompt_infos:
        prompt = PromptTemplate(
            template=info['template'],
            input_variables=["input"]
        )
        print("目标提示:\n", prompt)

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True
        )
        chain_map[info["key"]] = chain
    ## 【Step4】构建路由链router_chain
    destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
    router_template = RounterTemplate.format(destinations="\n".join(destinations))
    print("路由模板:\n", router_template)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    print("路由提示:\n", router_prompt)
    router_chain = LLMRouterChain.from_llm(
        llm,
        router_prompt,
        verbose=True
    )
    ## 【Step5】构建默认链 default_chain
    from langchain.chains import ConversationChain
    default_chain = ConversationChain(
        llm=llm,
        output_key="text",
        verbose=True
    )
    ## 【Step6】构建多提示链 MultiPromptChain
    from langchain.chains.router import MultiPromptChain
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=chain_map,
        default_chain=default_chain,
        verbose=True
    )
    # 测试1
    print(chain.run("如何为玫瑰浇水？"))


if __name__ == '__main__':
    # test1_llm_chain()
    # test2_simple_sequential_chain()
    # test3_sequential_chain()
    test4_router_Chain()
