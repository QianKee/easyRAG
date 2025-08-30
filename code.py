from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document  # Document type
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 使用新的导入方式
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    print("成功导入 langchain_huggingface 包")
except ImportError:
    print("正在安装 langchain-huggingface 包...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-huggingface"])
    from langchain_huggingface import HuggingFaceEmbeddings
#如果没有安装新的langchain-huggingface，会自动安装新版，有权限问题就要手动pip了

# 定义知识库目录
knowledge_base_path = r".\database"

# 使用DirectoryLoader加载目录下所有.txt文件，指定使用TextLoader
loader = DirectoryLoader(
    knowledge_base_path,
    glob="**/*.txt",  # 匹配所有txt文件
    loader_cls=TextLoader,  # 使用TextLoader加载
    loader_kwargs={'encoding': 'utf-8'},  # TextLoader的参数，确保UTF-8编码
    use_multithreading=True,  # 可以加速加载多个文件
    show_progress=True  # 显示加载进度
)

documents: List[Document] = loader.load()

if documents:
    print(f"成功加载 {len(documents)} 个文档。")
else:
    print("没有找到任何文档。")

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个块的最大字符数
    chunk_overlap=20,  # 相邻块之间的重叠字符数，帮助保留上下文连贯性
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],  # 中文场景
    length_function=len,  # 使用字符长度计算chunk_size
    add_start_index=True  # 在metadata中添加块在原文中的起始位置，可选
)

if documents:  # 确保前面加载成功
    document_chunks: List[Document] = text_splitter.split_documents(documents)
    print(f"\n文档被分割成 {len(document_chunks)} 个文本块.")

    if document_chunks:
        print(f"第一个文本块预览: {document_chunks[0].page_content}")
        print(f"第一个文本块元数据: {document_chunks[0].metadata}")
else:
    print("没有文档可供分割。")
    document_chunks = []  # 初始化为空列表以防后续代码出错

# 加载Embedding模型

# 如果你的机器有GPU并且安装了CUDA版本的PyTorch，可以设置为'cuda'
# 否则，sentence-transformers 会自动检测或使用 'cpu'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # BGE模型通常推荐对输出向量进行归一化
local_model_path = r"E:\embedding_model\bge-small-zh-v1.5"
embedding_model = HuggingFaceEmbeddings(
    model_name = local_model_path,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)
try:
    # 使用新的HuggingFaceEmbeddings类
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"\n成功加载Embedding模型: {local_model_path}")

    # 测试一下Embedding模型
    example_text = "这是一个示例文本，用于测试向量化功能。"
    query_vector = embedding_model.embed_query(example_text)
    print(f"示例文本的向量维度: {len(query_vector)}")
    # print(f"向量预览 (前5维): {query_vector[:5]}")
except Exception as e:
    print(f"加载Embedding模型失败: {e}")
    print("请确保已安装sentence-transformers且模型名称正确，或网络连接正常可以下载模型")
    embedding_model = None  # 标记模型加载失败
from langchain_community.vectorstores import Chroma

import shutil  # 用于清理旧的数据库目录

# 定义ChromaDB的持久化存储路径和集合名称
persist_directory = './chroma_db_store'
collection_name = 'my_rag_collection_v1'

# (可选) 清理旧的数据库目录，以便每次运行时都创建一个新的
# 设置为False则不清空，会尝试加载现有数据
if False:
    try:
        shutil.rmtree(persist_directory)
        print(f"已清理旧的数据库目录: {persist_directory}")
    except FileNotFoundError:
        print(f"数据库目录 {persist_directory} 不存在，无需清理。")
    except Exception as e:
        print(f"清理目录 {persist_directory} 失败: {e}")

vector_db = None  # 初始化

if document_chunks and embedding_model:  # 确保前面的步骤成功
    print(f"\n开始构建向量数据库和索引，使用集合名: {collection_name}...")

    try:
        vector_db = Chroma.from_documents(
            documents=document_chunks,  # 前面分割好的文本块列表
            embedding=embedding_model,  # 初始化好的Embedding模型实例
            collection_name=collection_name,
            persist_directory=persist_directory  # 指定持久化路径
        )

        # Chroma.from_documents 会自动处理持久化，但显式调用 persist() 确保写入磁盘
        # vector_db.persist() # 对于某些版本的ChromaDB或特定用法可能需要
        print(f"向量数据库 '{collection_name}' 构建完成并已持久化到 '{persist_directory}'")
        print(f"数据库中包含 {vector_db._collection.count()} 个向量条目.")
    except Exception as e:
        print(f"构建ChromaDB失败: {e}")
else:
    print("\n由于文档块列表为空或Embedding模型未加载，跳过向量数据库构建。")

from typing import List

# 假设前面的 vector_db (Chroma实例) 已经成功创建或加载
if vector_db:
    user_query = "顾溪茉是谁"
    k_results = 3  # 我们希望检索最相关的3个文档块

    # 将ChromaDB实例包装成一个Retriever对象
    # search_type默认为"similarity"，也可以指定为"mmr" (Maximal Marginal Relevance)
    # search_kwargs用于传递给底层向量存储的搜索参数，如k
    retriever = vector_db.as_retriever(search_kwargs={"k": k_results})
    print(f"\n为查询 '{user_query}' 配置的检索器将返回 {k_results} 个结果。")

    try:
        # 执行检索
        # invoke是LCEL推荐的方法, get_relevant_documents是旧版但仍可用
        relevant_docs: List[Document] = retriever.invoke(user_query)
        print(f"\n成功检索到 {len(relevant_docs)} 个相关文档块：")

        for i, doc in enumerate(relevant_docs):
            print(f"\n--- 相关文档 {i + 1} ---")
            # ChromaDB的as_retriever可能不直接在metadata中返回score，
            # 如果需要分数，要用 vector_db.similarity_search_with_score()
            # print(f"Score: {doc.metadata.get('score', 'N/A')}") # 示例，实际可能不存在
            print(f"来源: {doc.metadata.get('source', '未知')}, 块起始位置: {doc.metadata.get('start_index', '未知')}")
            print(f"内容: {doc.page_content}")

    except Exception as e:
        print(f"检索失败: {e}")
        relevant_docs = []  # 初始化为空列表
else:
    print("\n向量数据库未初始化，无法执行检索。")
    relevant_docs = []  # 初始化为空列表
from langchain_core.prompts import ChatPromptTemplate

prompt_template_str = """你是一个智能问答助手。你的任务是根据下面提供的【上下文信息】来回答问题。
请严格依据【上下文信息】进行回答，不要依赖任何外部知识或进行猜测。
如果【上下文信息】中没有足够的内容来回答【用户问题】，请直接回复："抱歉，根据我目前掌握的信息，无法回答这个问题。"
确保你的回答简洁、相关，并且直接针对【用户问题】。

【上下文信息】:
---
{context_str}
---

【用户问题】: {user_query}

【你的回答】:"""

prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

# 模拟：假设已经有了检索到的文档列表 relevant_docs 和用户查询 user_query
# (确保 user_query 和 relevant_docs 在此作用域内有效，通常它们来自上一步)
if relevant_docs and 'user_query' in locals() and user_query:
    # 将检索到的多个文档块内容合并为一个字符串
    context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    # 使用模板格式化Prompt，准备输入给LLM
    formatted_prompt_messages = prompt_template.format_messages(
        context_str=context_str,
        user_query=user_query
    )

    print("\n--- 为LLM准备的格式化Prompt ---")
    for msg in formatted_prompt_messages:
        print(f"类型: {msg.type}, 内容:\n{msg.content}")
    print("------------------------------")
else:
    print("\n相关文档列表为空或用户查询未定义，无法格式化Prompt。")
    formatted_prompt_messages = None

from langchain_core.output_parsers import StrOutputParser
import os  # 用于读取环境变量

# 初始化LLM和输出解析器
llm = None
output_parser = StrOutputParser()

# --- 方案1: 使用OpenAI API (gpt-3.5-turbo) ---
# 需要设置环境变量 OPENAI_API_KEY
# 同时可能需要设置 OPENAI_API_BASE 如果使用代理
use_openai = True  # 改为False以使用Ollama，或根据条件判断

if use_openai:
    from langchain_openai import ChatOpenAI

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # openai_api_base = os.getenv("OPENAI_API_BASE")  # 例如 "https://api.openai.com/v1"
        if not openai_api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量。")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            # openai_api_base=openai_api_base,
            model_name="gpt-3.5-turbo",
            temperature=0.1,  # 低temperature使回答更确定、更基于事实
            max_tokens=500
        )
        print("\n已配置使用 OpenAI GPT-3.5-Turbo。")
    except Exception as e:
        print(f"配置OpenAI失败: {e}. 将尝试Ollama。")
        use_openai = False  # 配置失败则尝试Ollama

# --- 方案2: 使用本地Ollama服务 (假设已运行并拉取了qwen:1.8b模型) ---
if not use_openai:  # 如果不使用OpenAI或OpenAI配置失败
    from langchain_community.llms import Ollama

    try:
        llm = Ollama(
            model="qwen:1.8b",  # 确保Ollama服务中有名为qwen:1.8b的模型
            # 可以通过 `ollama list` 查看可用模型
            # 或通过 `ollama pull qwen:1.8b` 拉取
            temperature=0.1
        )
        print("\n已配置使用本地Ollama (qwen:1.8b)。请确保Ollama服务正在运行。")
    except Exception as e:
        print(f"配置Ollama失败: {e}. LLM未能初始化。")
        llm = None  # 标记LLM初始化失败

# 调用LLM (如果Prompt和LLM都已准备好)
if 'formatted_prompt_messages' in locals() and formatted_prompt_messages and llm:
    try:
        # 使用LCEL构建链: prompt -> llm -> output_parser
        chain = prompt_template | llm | output_parser

        # 调用链
        final_answer_from_llm = chain.invoke({
            "context_str": context_str,
            "user_query": user_query
        })

        print("\n--- LLM生成的初步回答 ---")
        print(final_answer_from_llm)
        print("--------------------------")
    except Exception as e:
        print(f"LLM调用失败: {e}")
        if "OPENAI_API_KEY" in str(e).upper():
            print("提示: 如果使用OpenAI，请确保正确设置了OPENAI_API_KEY环境变量，并且有足够的API额度。")
        elif "CONNECTION REFUSED" in str(e).upper() and hasattr(llm, '__class__') and "Ollama" in str(llm.__class__):
            print("提示: 如果使用Ollama，请确保Ollama服务已在本地启动并运行。")
elif not llm:
    print("\nLLM模型未能成功初始化，无法生成答案。")
else:
    print("\n格式化Prompt未准备好，无法调用LLM。")
