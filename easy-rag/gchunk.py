#chunk:原文的切片
#注意原始文本必须是utf-8编码，要么改下面的encoding，要么改原文编码
def read_dat()-> str:
    with open("data1.txt","r",encoding="utf-8") as f:
        return f.read()

def get_chunks()->list[str]:
    content:str = read_dat()
    chunks:list[str] = content.split('\n\n')
    return chunks