# sales_bot

### 概述
在 [sales_chatbot](https://github.com/DjangoPeng/openai-quickstart/blob/main/langchain/sales_chatbot/sales_chatbot.py) 基础上，增加了另一个行业的语料，支持同时回答房产、电脑行业的相关问题。

### 实现方式
首先，通过 ChatGPT 获取各行业的销售话术，并使用 `markdown` 方式进行整理。然后使用 `MarkdownHeaderTextSplitter` 进行分割，从而得到对应的 metadata。  
最后通过 `SelfQueryRetriever` 通过问题中的信息来找到对应的行业内的问题，然后再基于相似性搜索找到对应的上下文。
