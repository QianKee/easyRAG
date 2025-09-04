import os
import gchunk
import chromadb
import google.generativeai as genai

# 运行前请先设置环境变量 GOOGLE_API_KEY
# 临时设置：打开cmd,输入：set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
# 永久设置：windows的话就在环境变量里
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("错误：未能找到名为 GOOGLE_API_KEY 的环境变量。")
    exit()
#设置大模型和embedding模型，top_k
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-flash-latest"
top_k:int = 5
#设置向量数据库存储地点和数据表
chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("data_list")
#embed:把一个chunk片段变成对应的向量
def embed(text: str, is_document: bool) -> list[float]:
    task = "RETRIEVAL_DOCUMENT" if is_document else "RETRIEVAL_QUERY"
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task
    )
    return result['embedding']
#create_db:把所有chunk都变成向量
def create_db() -> None:
    for idx,c in enumerate(gchunk.get_chunks()):
        embedding:list[float]= embed(c,True)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )
#query_db:求询问语句的向量，并返回top_k个最相关的chunk

def query_db(query:str)->list[str]:
    question_embeddings:list[float]= embed(query,False)
    result=chromadb_collection.query(
        query_embeddings=question_embeddings,
        n_results=top_k
    )
    return result["documents"][0]

