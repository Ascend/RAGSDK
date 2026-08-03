import argparse
import json
import os
from pathlib import Path
from loguru import logger
import torch.multiprocessing as mp
from tqdm import tqdm
from paddle.base import libpaddle  # noqa: F401

from data_preprocess import validate_graph_format, pkl2txt, DataBase
from src.run import train, test
from mx_rag.graphrag.graphs.networkx_graph import NetworkxGraph


def cmd_bool(cmdarg: str) -> bool:
    return cmdarg.lower() in ['true', '1', 'yes']


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v

    return cmd_bool(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo for GraphLP')
    # file path arguments
    parser.add_argument("--graph_path", type=str, required=True)
    # graph completion
    parser.add_argument("--tsp_dir", type=str, default="graph_completion_data")
    parser.add_argument("--del_exceed", default=False, action='store_true')
    parser.add_argument("--subgraph", type=int, default=3)
    parser.add_argument("--train_data_path", type=str, default="./graph_completion_data/train.tsv")
    parser.add_argument("--train_batch_size", type=int, default=30)
    parser.add_argument("--valid_data_path", type=str, default="./graph_completion_data/valid.tsv")
    parser.add_argument("--valid_batch_size", type=int, default=30)
    parser.add_argument("--test_data_path", type=str, default="./graph_completion_data/test.tsv")
    parser.add_argument("--test_batch_size", type=int, default=300)

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="./graph_completion_data/result.json")
    parser.add_argument("--enable_npu", type=str2bool, default=True)

    parser.add_argument("--savename", type=str, default="GraphFormers")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--token_length", type=int, default=32)
    parser.add_argument("--neighbor_num", type=int, default=16)
    parser.add_argument("--complete_threshold", type=float, default=0.5)
    parser.add_argument("--device_id", type=int, default=0)

    # model training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--log_steps", type=int, default=1000)
    parser.add_argument("--mlm", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=42)

    # turing
    parser.add_argument("--model_type", type=str, default="GraphFormers")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)

    parser.add_argument("--load_ckpt_name", type=str, default="GraphFormers-best.pt")
    parser.add_argument("--lr", type=float, default=1e-5)

    # half float
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--step", type=int, default=0)

    args = parser.parse_args()
    step = args.step

    try:
        if step == 0:
            if not validate_graph_format(args.graph_path):
                raise ValueError(f"Graph file {args.graph_path} is not in the correct format.")

            pkl2txt(args.graph_path, args.tsp_dir)
            DataBase(args)

        elif step == 1:
            args.mode = 'train'
            Path(args.model_dir).mkdir(parents=True, exist_ok=True)
            cont = False
            logger.info(f"Start training model {args.model_name_or_path}")

            if args.world_size > 1:
                mp.freeze_support()
                mgr = mp.Manager()
                end = mgr.Value('b', False)
                mp.spawn(train, args=(args, end, cont), nprocs=args.world_size, join=True)
            else:
                end = None
                train(0, args, end, cont)
        elif step == 2:
            args.mode = 'test'
            test(args)

        elif step == 3:
            threshold = args.complete_threshold
            base, ext = os.path.splitext(args.graph_path)
            save_path = f"{base}_completion{ext}"
            logger.info(f"loading graph file {args.graph_path}")
            graph = NetworkxGraph(path=args.graph_path)

            entity_nodes = {
                node for node in graph.get_nodes_by_attribute("type", "entity") if node and str(node).strip()
            }

            logger.info(f"loading data for {args.result_path} ")
            with open(args.result_path, 'r', encoding="utf-8") as f:
                items = json.load(f)

            count = 0
            skipped = 0

            for item in tqdm(items, desc="add edges"):
                prob = item.get('prob', 0)
                if prob >= threshold:
                    query = item.get("query_name")
                    candidate = item.get("key_name")

                    if query in entity_nodes and candidate in entity_nodes:
                        graph.add_edge(query, candidate, relation="语义相似")
                        count += 1
                else:
                    skipped += 1
            graph.save(save_path)
    except Exception as e:
        logger.error(f"Error: {e}")
