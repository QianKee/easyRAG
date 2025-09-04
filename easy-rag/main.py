from google.genai.types import GenerateContentResponse
import google.generativeai as genai

import embed

if __name__ == "__main__":
    embed.create_db()  # 只要run一次就可以，多次查询时可以注释掉
    question = "Puts your question here"
    find_chunks: list[str] = embed.query_db(question)
    prompt = "You should answer the following question according to context:"
    prompt += question
    for c in find_chunks:
        prompt += f"{c}\n"
        prompt += "---------\n"

    # 初始化模型并生成内容
    model = genai.GenerativeModel(embed.LLM_MODEL)  # 使用 embed.LLM_MODEL 指定模型
    result: GenerateContentResponse = model.generate_content(prompt)

    print(result)