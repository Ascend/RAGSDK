import argparse
import base64
import io
import json
import os
import re
import shutil
from http import HTTPStatus
from typing import Optional
from pathlib import Path

import docx
import fitz
import httpx
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from pymilvus import MilvusClient
from starlette.responses import JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader
from mx_rag.knowledge import KnowledgeStore, KnowledgeDB
from mx_rag.reranker.service import TEIReranker
from mx_rag.retrievers import Retriever, FullTextRetriever
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


app = FastAPI()

upload_file_dir = os.environ.get("UPLOAD_FILE_DIR", "/home/data/files")
images_store_dir = os.environ.get("IMG_STORE_DIR", "/home/data/images")
global_config = {}

img_to_text_prompt = """Given an image containing a table or figure, please provide a structured and detailed
description in chinese with two levels of granularity:

  Coarse-grained Description:
  - Summarize the overall content and purpose of the image.
  - Briefly state what type of data or information is presented (e.g., comparison, trend, distribution).
  - Mention the main topic or message conveyed by the table or figure.

  Fine-grained Description:
  - Describe the specific details present in the image.
  - For tables: List the column and row headers, units, and any notable values, patterns, or anomalies.
  - For figures (e.g., plots, charts): Explain the axes, data series, legends, and any significant trends, outliers,
  or data points.
  - Note any labels, captions, or annotations included in the image.
  - Highlight specific examples or noteworthy details.

  Deliver the description in a clear, organized, and reader-friendly manner, using bullet points or paragraphs
  as appropriate, answer in chinese"""

text_infer_prompt = """
You are a helpful question-answering assistant. Your task is to generate a interleaved text and image response based on provided questions and quotes. Here‘s how to refine your process:

1. **Evidence Selection**:
   - From both text and image quotes, pinpoint those really relevant for answering the question. Focus on significance and direct relevance.
   - Each image quote is the description of the image.

2. **Answer Construction**:
   - Use Markdown to embed text and images in your response, avoid using obvious headings or divisions; ensure the response flows naturally and cohesively.
   - Conclude with a direct and concise answer to the question in a simple and clear sentence.

3. **Quote Citation**:
   - Cite text by adding [index]; for example, quote from the first text should be [1].
   - Cite images using the format `![{conclusion}](image index)`; for the first image, use `![{conclusion}](image1)`;The {conclusion} should be a concise one-sentence summary of the image’s content.
   - Ensure the cite of the image must strict follow `![{conclusion}](image index)`, do not simply stating "See image1", "image1 shows" ,"[image1]" or "image1".
   - Each image or text can only be quoted once.

- Do not cite irrelevant quotes.
- Compose a detailed and articulate interleaved answer to the question.
- Ensure that your answer is logical, informative, and directly ties back to the evidence provided by the quotes.
- Interleaved answer must contain both text and image response.
- Answer in chinese.
"""


def get_config(key, default=None):
    return global_config.get(key, default)


# 创建llm_chain 客户端
def create_llm_chain(base_url, model_name):
    http_client = httpx.Client()
    root_client = OpenAI(base_url=base_url, api_key="sk_fake", http_client=http_client)

    client = root_client.chat.completions

    llm = ChatOpenAI(
        api_key="sk_fake",
        client=client,
        model_name=model_name,
        temperature=0.5,
        streaming=True,
    )

    return llm


def compose_text_messages(question, text_docs, img_docs):
    # 2. Add text quotes
    user_message = "Text Quotes are:"
    for i, doc in enumerate(text_docs):
        user_message += f"\n[{i + 1}] {doc.page_content}"

    # 3. Add image quotes vlm-text or ocr-text
    user_message += "\nImage Quotes are:"
    for i, doc in enumerate(img_docs):
        user_message += f"\nimage{i + 1} is described as: {doc.page_content}"
    user_message += "\n\n"

    # 4. add user question
    user_message += f"The user question is: {question}"

    return user_message


# 从pdf文件中解析出所有的图片并存储到output_folder目录下
def extract_images_from_docx(image_out_dir, file_path):
    # 打开文档
    doc = docx.Document(file_path)

    output_folder = os.path.join(image_out_dir, os.path.basename(file_path))
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 解析文档中的图片
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_part = rel.target_part
            image_filename = os.path.basename(image_part.partname)
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_part.blob)

    logger.info(f"extract images from {file_path} successfully")


# 从pdf文件中解析出所有的图片并存储到output_folder目录下
def extract_images_from_pdf(image_out_dir, file_path):
    output_folder = os.path.join(image_out_dir, os.path.basename(file_path))
    # 打开PDF文件
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(file_path)

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        # 遍历每一张图片
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 保存图片
            image_filename = f"image_{page_num + 1}_{img_index + 1}.{image_ext}"
            image_path = f"{output_folder}/{image_filename}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

    pdf_document.close()

    logger.info(f"extract images from {file_path} successfully")


# 获取模型名
def get_model_name(base_url):
    """
    调用 /v1/models 接口获取模型列表，返回第一个可用模型的名称

    :param base_url: API 基础地址
    :return: 模型名称或 None
    """
    try:
        models_url = f"{base_url.rstrip('/')}/models"
        response = httpx.get(models_url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            # 返回第一个模型的名称
            return data["data"][0]["id"]

        return None
    except Exception as e:
        logger.error(f"Failed to get model name from {base_url}: {e}")
        return None


# 获取向量模型对象
def get_embedding(url):
    # 初始化embedding客户端对象
    return TEIEmbedding(url=url, client_param=ClientParam(use_http=True))


# 获取向量模型对象
def get_embedding_dim(url):
    # 初始化embedding客户端对象
    emb = get_embedding(url)
    return len(emb.embed_query("The capital of China is Beijing."))


# 获取向量数据库对象
def get_vector_store():
    knowledge_name = get_config("knowledge_name")
    milvus_client = MilvusClient(get_config("milvus_url"))
    vector_store = MilvusDB.create(
        client=milvus_client,
        x_dim=get_config("embedding_dim"),
        collection_name=f"{knowledge_name}_vector",
    )
    return vector_store


# 获取文本数据库对象
def get_chunk_store():
    knowledge_name = get_config("knowledge_name")
    milvus_client = MilvusClient(get_config("milvus_url"))
    return MilvusDocstore(milvus_client, collection_name=f"{knowledge_name}_chunk")


# 获取知识库对象
def get_knowledge_db():
    chunk_store = get_chunk_store()
    vector_store = get_vector_store()
    knowledge_store = KnowledgeStore(db_path="./knowledge_store_sql.db")
    knowledge_name = get_config("knowledge_name")
    knowledge_store.add_knowledge(knowledge_name, user_id="knowledge_demo")

    # 初始化知识库管理
    knowledge_db = KnowledgeDB(
        knowledge_store=knowledge_store,
        chunk_store=chunk_store,
        vector_store=vector_store,
        knowledge_name=knowledge_name,
        white_paths=["/home"],
        user_id="knowledge_demo",
    )

    return knowledge_db


# 获取文档加载器，和切分器
def get_document_loader_splitter(file_suffix):
    # 初始化文档加载切分管理器
    loader_mng = LoaderMng()

    # 注册文档加载器，可以使用mxrag提供的，也可以使用langchain提供的，同时也可实现langchain_community.document_loaders.base.BaseLoader
    # 接口类自定义实现文档解析功能
    loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt", ".md"])
    loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
    loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
    # 注册文档切分器，可自定义实现langchain_text_splitters.base.TextSplitter基类对文档进行切分
    loader_mng.register_splitter(
        splitter_class=RecursiveCharacterTextSplitter,
        file_types=[".docx", ".txt", ".md", ".pdf"],
        splitter_params={
            "chunk_size": 750,
            "chunk_overlap": 150,
            "keep_separator": False,
        },
    )

    # 根据文件后缀获取对应的文件解析器信息，包含解析类，及参数
    loader_info = loader_mng.get_loader(file_suffix)
    # 根据文件后缀获取对应的文件切分器信息，包含切分类，及参数
    splitter_info = loader_mng.get_splitter(file_suffix)

    return loader_info, splitter_info


# 根据问题从数据库中检索相似片段
def retrieve_similarity_docs(query, top_k, score_threshold):
    # 获取embedding对象
    emb = get_embedding(get_config("embedding_url"))
    # 获取文本和向量数据库对象
    chunk_store = get_chunk_store()
    vector_store = get_vector_store()

    # 配置向量检索器，
    dense_retriever = Retriever(
        vector_store=vector_store,
        document_store=chunk_store,
        embed_func=emb.embed_documents,
        k=top_k,
        score_threshold=score_threshold,
    )

    # 调用检索器从向量数据库中查找出和query最相近的tok个文档chunk
    dense_res = dense_retriever.invoke(query)

    # 配置全文检索器，其实现原理为BM25检索
    full_text_retriever = FullTextRetriever(document_store=chunk_store, k=top_k)

    full_text_res = full_text_retriever.invoke(query)

    # 合并检索结果
    docs = dense_res + full_text_res

    # 两路检索，可能检索到重复的片段，去重处理
    contents = set()
    new_docs = []
    for doc in docs:
        if doc.page_content not in contents:
            new_docs.append(doc)
            contents.add(doc.page_content)

    logger.info("retrieve similarity chunks from knowledge successfully")
    return new_docs


def generate_answer(query, q_docs):
    # 拆分知识片段分为原始文本和图片多粒度信息文本
    text_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "text"]
    img_docs = [doc for doc in q_docs if doc.metadata.get("type", "") == "image"]

    text = compose_text_messages(query, text_docs, img_docs)

    # 构造请求消息
    messages = [
        {"role": "system", "content": text_infer_prompt},
        {"role": "user", "content": text},
    ]

    # 配置大模型客户端对象
    llm_chain = create_llm_chain(base_url=get_config("llm_base_url"), model_name=get_config("llm_model_name"))

    response = ""
    try:
        response = llm_chain.invoke(messages).content
        logger.info("generate answer by llm successfully")
    except Exception as e:
        logger.error(f"call llm invoke failed:{e}")

    result = replace_image_paths(response, img_docs)

    return result


def replace_image_paths(text, image_docs):
    """
    将response中的imageN替换为image_docs中对应的image_path
    """
    if text == "":
        return text

    # 创建imageN到image_path的映射
    image_mapping = {}
    for i, doc in enumerate(image_docs):
        image_key = f"image{i + 1}"
        image_path = doc.metadata.get("image_path", "")
        image_mapping[image_key] = image_path

    # 使用正则表达式找到所有的![...](imageN)模式并替换
    def replace_match(match):
        full_match = match.group(0)  # 完整匹配的字符串
        alt_text = match.group(1)  # 图片描述文本
        image_ref = match.group(2)  # imageN

        # 如果找到对应的image_path，则替换
        if image_ref in image_mapping:
            return f"![{alt_text}]({image_mapping[image_ref]})"
        else:
            # 如果没有找到对应的路径，保持原样
            return full_match

    # 匹配 ![...](imageN) 模式
    pattern = r"!\[([^\]]*)\]\((image\d+)\)"
    updated_response = re.sub(pattern, replace_match, text)

    logger.info("Image paths replacement completed")
    return updated_response


# 调用vlm对图片进行多粒度理解
def extract_image_info_by_vlm(image_path):
    # 将图像转换为 base64 编码的字符串
    with Image.open(image_path) as img:
        width, height = img.size
        # 如果图片小于256*256，直接返回
        if width < 256 and height < 256:
            logger.warning(f"image:{image_path} size: ({width},{height}) too little, will be discarded")
            return ""

        buffer = io.BytesIO()
        if Path(image_path).suffix == ".png":
            img = img.convert("RGB")

        if width > 1024 or height > 1024:
            img = img.resize(size=(width // 2, height // 2))

        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 构造请求消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": img_to_text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image;base64,{img_str}"},
                },
            ],
        }
    ]

    vlm = create_llm_chain(base_url=get_config("vlm_base_url"), model_name=get_config("vlm_model_name"))

    try:
        return vlm.invoke(messages).content
    except Exception as e:
        logger.error(f"call vlm invoke failed:{e}")
        return ""


# 获取目录下的所有图片文件路径
def find_images_files(directory, recursive=False):
    base_path = Path(directory)
    exts = (".jpg", ".jpeg", ".png")
    files = []

    for ext in exts:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        file_list = list(base_path.glob(pattern))
        files.extend(str(p) for p in file_list)

    return files


# 将图片目录下的所有有图片调用vlm进行多粒度理解述并保存到json文件中
def extract_images_info_by_vlm(image_out_dir, file_name):
    image_dir = os.path.join(image_out_dir, file_name)

    # 避免同一个文件中的所有图片重复调用vlm 提取信息，如果image_info.json如果存在，表示已经提取，不用再次提取，降低vlm算力
    if os.path.exists(os.path.join(image_dir, "image_info.json")):
        logger.warning(f"all images in {image_dir} have been extracted, no need to repeat extraction")
        return

    logger.info(f"start to extract images info in file [{file_name}] by vlm ...")

    image_files = find_images_files(image_dir)
    info = []
    for image_file in image_files:
        logger.debug(f"start to deal {[image_file]} by vlm")
        res = extract_image_info_by_vlm(image_file)
        if res:
            info.append({"image_path": image_file, "image_description": res})

    if len(info) > 0:
        with open(os.path.join(image_dir, "image_info.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(info, indent=4, ensure_ascii=False))

    logger.info("extract images info successfully")


class RetrievalParam(BaseModel):
    query: str
    knowledge_id: str = ""
    retrieval_setting: Optional[dict] = {"top_k": 3, "score_threshold": 0.5}


@app.post("/retrieval")
async def retrieve(arg: RetrievalParam):
    logger.info("doing retrieval")

    top_k = int(arg.retrieval_setting.get("top_k", 3))

    try:
        score_threshold = arg.retrieval_setting.get("score_threshold", 0.5)
    except (ValueError, TypeError):
        score_threshold = 0.5

    text_reranker = TEIReranker(url=get_config("reranker_url"), k=top_k, client_param=ClientParam(use_http=True))

    q_docs = retrieve_similarity_docs(arg.query, top_k, score_threshold)
    if text_reranker is not None and len(q_docs) > 0:
        score = text_reranker.rerank(arg.query, [doc.page_content for doc in q_docs])
        q_docs = text_reranker.rerank_top_k(q_docs, score)

    records = []
    for doc in q_docs:
        records.append(
            {
                "content": doc.page_content,
                "score": doc.metadata.get("score", 0),
                "title": doc.metadata.get("source", ""),
                "metadata": doc.metadata,
            }
        )

    return JSONResponse(content={"records": records})


class QueryParam(BaseModel):
    query: str
    retrieval_setting: Optional[dict] = {"top_k": 3, "score_threshold": 0.5}


@app.post("/query")
async def query_answer(arg: QueryParam):
    logger.info("doing query")

    top_k = int(arg.retrieval_setting.get("top_k", 3))
    try:
        score_threshold = arg.retrieval_setting.get("score_threshold", 0.5)
    except (ValueError, TypeError):
        score_threshold = 0.5

    q_docs = retrieve_similarity_docs(arg.query, top_k, score_threshold)

    text_reranker = TEIReranker(url=get_config("reranker_url"), k=top_k, client_param=ClientParam(use_http=True))

    if text_reranker is not None:
        score = text_reranker.rerank(arg.query, [doc.page_content for doc in q_docs])
        q_docs = text_reranker.rerank_top_k(q_docs, score)

    response = generate_answer(arg.query, q_docs)

    with open("response.md", "w", encoding="utf-8") as f:
        f.write(response)

    return JSONResponse(content=response)


@app.post("/upload_file")
async def upload_file(file: UploadFile):
    logger.info(f"start to upload file: {file.filename}")

    # 文件白名单
    ALLOWED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
    # 文件大小限制：50MB
    MAX_FILE_SIZE = 50 * 1024 * 1024

    # 文件扩展名校验
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File type {file_ext} not allowed")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"},
        )

    if not os.path.exists(upload_file_dir):
        os.makedirs(upload_file_dir)

    try:
        contents = file.file.read()

        # 文件大小校验
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File size {len(contents)} exceeds limit {MAX_FILE_SIZE}")
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error_info": "File size exceeds limit (max 50MB)"},
            )

        file_path = os.path.join(upload_file_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"write file {file.filename} failed: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error_info": "Something went wrong when upload file"},
        )
    finally:
        file.file.close()

    file_obj = Path(file_path)

    if get_config("parse_image"):
        if file_obj.suffix == ".docx":
            extract_images_from_docx(images_store_dir, file_path)
        if file_obj.suffix == ".pdf":
            extract_images_from_pdf(images_store_dir, file_path)

        # 第二步 对image_out_dir下的所有图片进行vlm 多粒度理解，结果存放在image_out_dir/image_info.json中
        extract_images_info_by_vlm(images_store_dir, file.filename)

    # 根据文类型，获取loader类和splitter类信息
    loader_info, splitter_info = get_document_loader_splitter(file_obj.suffix)

    # 获取embedding对象
    emb = get_embedding(get_config("embedding_url"))
    # 获取知识库管理对象
    knowledge_db = get_knowledge_db()

    file_base_name = os.path.basename(file_path)

    # 检查当前文件是否已经入过库
    if knowledge_db.check_document_exist(file_base_name):
        logger.warning(f"file {file_base_name} exists in knowledge db")
        return

    # 创建文件解析器和切分器
    loader = loader_info.loader_class(file_path=file_obj.as_posix(), **loader_info.loader_params)
    splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
    # 解析文件并切分
    docs = loader.load_and_split(splitter)

    # 获取文档片段chunk内容和元数据信息
    texts = [doc.page_content for doc in docs if doc.page_content]
    meta_data = [{**doc.metadata, "type": "text"} for doc in docs if doc.page_content]

    if file_obj.suffix in [".docx", ".pdf"]:
        # 解析的图片存放目录
        file_image_out_dir = os.path.join(images_store_dir, file_base_name)

        # 读取图片多粒度解析信息
        try:
            with open(
                os.path.join(file_image_out_dir, "image_info.json"),
                "r",
                encoding="utf-8",
            ) as f:
                images_description = json.load(f)
        except Exception as e:
            logger.warning(f"read image info failed: {e}")
            images_description = {}

        for description in images_description:
            texts.append(description["image_description"])
            meta_data.append(
                {
                    "type": "image",
                    "source": file_path,
                    "image_path": description.get("image_path"),
                }
            )

    # 存储到文本、向量数据库中
    knowledge_db.add_file(file_obj, texts, {"dense": emb.embed_documents}, meta_data)

    logger.info(f"upload file {file.filename} to knowledge successfully")  # pylint: disable=W0101

    return JSONResponse(content={"info": f"upload file:{file.filename} successfully"})


class DeleteFileParam(BaseModel):
    file_name: str
    confirm: bool = False


@app.post("/delete_file")
async def delete_file(file: DeleteFileParam):
    logger.info(f"start to delete file:{file.file_name}")

    # 校验文件名不为空
    if not file.file_name or not file.file_name.strip():
        logger.warning("File name cannot be empty")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": "File name cannot be empty"},
        )

    # 防止路径遍历攻击
    if ".." in file.file_name or "/" in file.file_name or "\\" in file.file_name:
        logger.warning(f"Invalid file name: {file.file_name}")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": "Invalid file name"},
        )

    # 二次确认
    if not file.confirm:
        logger.warning(f"Deletion not confirmed for file: {file.file_name}")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": "Please set 'confirm: true' to confirm deletion"},
        )

    # 检查文件是否存在
    file_path = os.path.join(upload_file_dir, file.file_name)
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file.file_name}")
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"error_info": f"File '{file.file_name}' not found"},
        )

    try:
        # 删除数据库中的文档信息
        knowledge_db = get_knowledge_db()
        knowledge_db.delete_file(file.file_name)

        # 删除web上传时存放的文件
        os.remove(file_path)

        # 删除从文件解析出来的图片
        if get_config("parse_image"):
            image_dir = os.path.join(images_store_dir, file.file_name)
            if os.path.exists(image_dir):
                shutil.rmtree(image_dir)

    except Exception as e:
        logger.error(f"delete file [{file.file_name}] failed: {e}")
        return JSONResponse(content=f"delete file [{file.file_name}] failed: {e}")

    return JSONResponse(content=f"delete file [{file.file_name}] successfully")


class DeleteAllFilesParam(BaseModel):
    confirm: bool = False


@app.delete("/delete_all_files")
async def delete_all_files(param: DeleteAllFilesParam):
    logger.info("start to delete all files")

    # 二次确认检查
    if not param.confirm:
        logger.warning("Deletion not confirmed")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": "Please set 'confirm: true' to confirm deletion"},
        )

    # 检查知识库是否为空
    knowledge_db = get_knowledge_db()
    all_docs = knowledge_db.get_all_documents()
    if len(all_docs) == 0:
        logger.warning("No files to delete")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error_info": "Knowledge base is already empty"},
        )

    logger.warning(f"DELETING ALL {len(all_docs)} FILES - this action cannot be undone!")

    try:
        knowledge_db.delete_all()

        vector_store = get_vector_store()
        chunk_store = get_chunk_store()

        vector_store.drop_collection()
        chunk_store.drop_collection()

        # 删除从文件解析出来的图片
        try:
            if os.path.exists(upload_file_dir):
                shutil.rmtree(upload_file_dir)
            if get_config("parse_image") and os.path.exists(images_store_dir):
                shutil.rmtree(images_store_dir)
        except Exception as e:
            logger.info(f"delete directories failed: {e}")

    except Exception as e:
        logger.error(f"delete all files failed: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error_info": f"Failed to delete all files: {e}"},
        )

    return JSONResponse({"message": "ok"})


@app.get("/list_files")
async def list_files():
    logger.info("start to list all file")
    knowledge_db = get_knowledge_db()
    docs = knowledge_db.get_all_documents()
    files = [doc.document_name for doc in docs]
    return JSONResponse(content=files)


class QueryFileContentParam(BaseModel):
    file_name: str


@app.post("/query_file_content")
async def query_file_content(file: QueryFileContentParam):
    logger.info(f"start to query file:{file.file_name}")

    doc_id = 0
    knowledge_db = get_knowledge_db()
    documents = knowledge_db.get_all_documents()
    for doc in documents:
        if doc.document_name == file.file_name:
            doc_id = doc.document_id
            break
    if not doc_id:
        logger.error(f"there is no {file.file_name} in db")
        return JSONResponse(content="query file content failed")

    chunk_store = get_chunk_store()
    chunks = chunk_store.search_by_document_id(doc_id)
    return JSONResponse(content=[{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks])


def load_config():
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def _get_default_metavar_for_optional(self, action):
            return action.type.__name__

        def _get_default_metavar_for_positional(self, action):
            return action.type.__name__

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://192.168.9.146:1025/v1",
        help="LLM大模型服务地址",
    )
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default="http://192.168.9.146:8000/v1",
        help="VLM大模型服务地址",
    )
    parser.add_argument("--host", type=str, default="192.168.9.146", help="服务host")
    parser.add_argument("--port", type=int, default=9098, help="服务端口")
    parser.add_argument("--ssl_key_file", type=str, help="ssl秘钥文件")
    parser.add_argument("--ssl_cert_file", type=str, help="ssl证书文件")
    parser.add_argument("--ssl_ca_certs", type=str, help="ssl证书根证书文件")
    parser.add_argument("--ssl_cert_reqs", type=str, help="ssl证书验证要求")
    parser.add_argument(
        "--embedding_url",
        type=str,
        default="http://192.168.9.146:9123/v1/embeddings",
        help="向量模型服务地址",
    )
    parser.add_argument(
        "--reranker_url",
        type=str,
        default="http://192.168.9.146:9124/v1/rerank",
        help="排序模型服务地址",
    )
    parser.add_argument("--milvus_url", type=str, default="./milvus.db", help="milvus数据库服务地址")
    parser.add_argument("--knowledge_name", type=str, default="test", help="知识库名称")
    parser.add_argument("--parse_image", action="store_true", help="是否解析图片信息")

    args = parser.parse_args()
    config = vars(args)

    config["embedding_dim"] = get_embedding_dim(args.embedding_url)
    config["llm_model_name"] = get_model_name(args.llm_base_url)

    if config["parse_image"]:
        vlm_model_name = get_model_name(args.vlm_base_url)
        if vlm_model_name is None:
            raise ValueError("parse_image is True, but vlm_base_url is invalid")
        config["vlm_model_name"] = vlm_model_name

    # 应用配置到 args
    for k, v in config.items():
        setattr(args, k, v)

    return args, config


def main():
    args, config = load_config()

    global global_config
    global_config = config

    # 运行服务
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_key_file,
        ssl_certfile=args.ssl_cert_file,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )


if __name__ == "__main__":
    main()
