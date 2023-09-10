import gradio as gr
import os
from langchain.vectorstores import Chroma

from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo


def initialize_sales_bot():

    embeddings_model = OpenAIEmbeddings(
        openai_api_key=os.environ.get("API2D_API_KEY"),
        openai_api_base='https://openai.api2d.net/v1'
    )

    db = Chroma(persist_directory="./sales_chroma_db",
                embedding_function=embeddings_model)
    llm = OpenAI(
        temperature=0.1,
        verbose=True,
        openai_api_key=os.environ.get("API2D_API_KEY"),
        openai_api_base='https://openai.api2d.net/v1'
    )
    chat = ChatOpenAI(
        verbose=True,
        openai_api_key=os.environ.get("API2D_API_KEY"),
        openai_api_base='https://openai.api2d.net/v1',
        temperature=0
    )

    # 如果没有数据，则先去做一次 embedding
    n = db._collection.count()
    if n == 0:
        markdown_path = "sales_state.md"
        loader = TextLoader(markdown_path, encoding="UTF-8")
        data = loader.load()

        headers_to_split_on = [
            ("##", "industry"),
            ("###", "question"),
        ]

        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(data[0].page_content)

        chunk_size = 250
        chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(md_header_splits)
        db = Chroma.from_documents(
            splits, embeddings_model, persist_directory="./sales_chroma_db")

    # 中文支持不是很好
    metadata_field_info = [
        AttributeInfo(
            name="industry",
            description="The relevant industry about the question",
            type="string",
        ),
        AttributeInfo(
            name="question",
            description="The question asked by the user",
            type="string",
        ),
    ]
    retriever = SelfQueryRetriever.from_llm(
        llm, db, "销售话术", metadata_field_info, verbose=True
    )

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(chat,
                                            retriever=retriever,
                                            verbose=True)
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产、电脑销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
