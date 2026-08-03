import argparse

from mx_rag.document.loader import MarkdownLoader
from mx_rag.document.splitter import MarkdownTextSplitter


def rag_demo_upload():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="要上传的文件路径",
    )
    args = parse.parse_args()
    docs = MarkdownLoader(args.file_path).load_and_split(MarkdownTextSplitter(chunk_size=200, chunk_overlap=30))

    print(f"total docs:{len(docs)}")


if __name__ == "__main__":
    rag_demo_upload()
