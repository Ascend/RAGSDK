import math
import os
import random

import torch
from loguru import logger
from collections import defaultdict as ddict
from mx_rag.graphrag.graphs.networkx_graph import NetworkxGraph


def validate_graph_format(graph_path):
    if not os.path.exists(graph_path):
        logger.error(f"Graph file not found: {graph_path}")
        return False

    logger.info(f"Validated graph format: {graph_path}")

    try:
        g1 = NetworkxGraph(path=graph_path)

        entity_count = len(g1.get_nodes_by_attribute("type", "entity"))
        event_count = len(g1.get_nodes_by_attribute("type", "event"))
        text_count = len(g1.get_nodes_by_attribute("type", "raw_text"))
        if entity_count == 0:
            logger.error(f"Graph file {graph_path} does not contain any entity nodes.")
            return False

        if event_count == 0:
            logger.error(f"Graph file {graph_path} does not contain any event nodes.")
            return False

        if text_count == 0:
            logger.error(f"Graph file {graph_path} does not contain any text nodes.")
            return False

    except Exception as e:
        logger.error(f"Error validating graph format: {e}")
        return False

    return True


def pkl2txt(graph_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    id2entities_path = os.path.join(out_dir, "id2entities.txt")
    id2relations_path = os.path.join(out_dir, "id2relations.txt")
    triple_path = os.path.join(out_dir, "triple.txt")

    g1 = NetworkxGraph(path=graph_path)

    # 1 获取实体集合
    entity_nodes = set(g1.get_nodes_by_attribute("type", "entity"))
    entity2id = {entity.strip(): idx for idx, entity in enumerate(entity_nodes)}

    # 写入实体字典
    os.makedirs(os.path.dirname(id2entities_path), exist_ok=True)
    with open(id2entities_path, 'w', encoding='utf-8') as f:
        for entity, idx in entity2id.items():
            f.write(f"{idx}\t{entity}\n")

    # 2. 只保留entity-entity的三元组
    triples = []
    relations_in_use = set()

    for u, v, attr in g1.get_edges():
        rel_name = attr.get('relation', '').strip()
        if not rel_name:
            continue

        # 只保留头尾都是实体的边
        if u in entity_nodes and v in entity_nodes:
            triples.append((u, rel_name, v))
            relations_in_use.add(rel_name)

    # 3. 构建只包含这些关系的 relation2id
    relation2id = {rel: idx for idx, rel in enumerate(sorted(relations_in_use))}
    # 写入关系字典
    with open(id2relations_path, 'w', encoding='utf-8') as f:
        for rel, idx in relation2id.items():
            f.write(f"{idx}\t{rel}\n")

    with open(triple_path, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def make_ordered_neighbor_list(center, neighbors, max_keep=None):
    out = []
    seen = set()
    if isinstance(center, str) and center.strip():
        out.append(center)
        seen.add(center)

    for n in neighbors:
        if not isinstance(n, str):
            n = str(n)
        if n in seen:
            continue
        out.append(n)
        seen.add(n)
        if max_keep is not None and len(out) >= max_keep:
            break
    return out


class DataBase:
    def __init__(self, para: object):
        self.parts = None
        self.neighborsHT = None
        self.g1 = NetworkxGraph(path=para.graph_path)
        self.out_dir = para.tsp_dir
        with open(f"{para.tsp_dir}/id2entities.txt", "r", encoding='utf-8') as e2i:
            lines = [parts for line in e2i if len(parts := line.strip().split('\t')) == 2]
            self.ent2id = {e: int(i) for i, e in lines}
        with open(f"{para.tsp_dir}/id2relations.txt", 'r', encoding='utf-8') as r2i:
            lines = [parts for line in r2i if len(parts := line.strip().split('\t')) == 2]
            self.rel2id = {r: int(i) for i, r in lines}

        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(self.rel2id)})
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = para.num_ent = len(self.ent2id)
        self.num_rel = para.num_rel = len(self.rel2id) // 2
        self.para = para
        self.data = []

        with open(f'{para.tsp_dir}/triple.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip().split('\t')) != 3:
                    continue

                sub, rel, obj = [x.strip() for x in line.strip().split('\t')]
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data.append((sub, rel, obj))

        neighbors = ddict(lambda: ddict(set))
        for h, r, t in self.data:
            neighbors[h][r].add(t)
            neighbors[t][r + self.num_rel].add(h)

        self.neighbors = {h: {r: list(t) for r, t in rt.items()} for h, rt in neighbors.items()}
        dumped_file = f'{para.tsp_dir}/subgraph.pkl'
        logger.info("extract subgraph")
        if os.path.exists(dumped_file):
            self.subgraph = torch.load(dumped_file, weights_only=True)

        else:
            self.count_parts()
            best_metric, best_subgraph = [1e9, 1e9, 1e9], None
            for i in range(5):
                logger.info(f"iter {i} to find best subgraph")
                desc = self.extract_subgraph(para.subgraph)
                logger.info(f"part {i}: {desc}")

                if best_metric[0] > desc[0]:  # desc[0]为最大子图中包含的三元组个数
                    best_metric = desc
                    best_subgraph = self.subgraph

            torch.save(best_subgraph, dumped_file)

        train_file = os.path.join(self.out_dir, 'train.tsv')
        if not os.path.exists(train_file):
            self.pkl2tsv()
        test_file = os.path.join(self.out_dir, 'test.tsv')
        if not os.path.exists(test_file):
            self.subgraph2tsv()

    def subgraph2tsv(self, max_neighbors=None):
        passage_nodes = {n for n in self.g1.get_nodes_by_attribute('type', 'original_text') if n and str(n).strip()}
        sep = "|'|"
        subgraph = self.subgraph
        q_k_neighbor = []
        for head in subgraph:
            triples = subgraph[head]
            if triples.shape[0] == 0:
                continue
            edge_index = torch.cat([triples[:, [0, 2]], triples[:, [2, 0]]], dim=0).t()
            ents = edge_index[0, :].unique()
            perdHTs = torch.cat(
                [ents.view(-1, 1).repeat(1, len(ents)).reshape(-1, 1), ents.view(-1, 1).repeat(len(ents), 1)], dim=-1
            )
            perdHTs = perdHTs[perdHTs[:, 0] != perdHTs[:, 1]]

            for HT in perdHTs:
                u, v = HT.cpu().numpy()
                u, v = self.id2ent[u], self.id2ent[v]
                if not (isinstance(u, str) and isinstance(v, str)):
                    continue
                if not u.strip() or not v.strip():
                    continue

                try:
                    neighbor_u = list(self.g1.successors(u)) + list(self.g1.predecessors(u))
                except Exception as e:
                    logger.error(f"exception happened because:{e}")
                    neighbor_u = []

                try:
                    neighbor_v = list(self.g1.successors(v)) + list(self.g1.predecessors(v))
                except Exception as e:
                    logger.error(f"exception happened because:{e}")
                    neighbor_v = []

                if v in neighbor_u:
                    continue
                if u in neighbor_v:
                    continue

                neighbor_u = [n for n in neighbor_u if n not in passage_nodes]
                neighbor_v = [n for n in neighbor_v if n not in passage_nodes]

                neighbor_u_list = make_ordered_neighbor_list(u, neighbor_u, max_neighbors)
                neighbor_v_list = make_ordered_neighbor_list(v, neighbor_v, max_neighbors)

                q_and_neighbor = sep.join(neighbor_u_list)
                k_and_neighbor = sep.join(neighbor_v_list)

                q_k_neighbor.append((q_and_neighbor, k_and_neighbor))

        file_name = os.path.join(self.out_dir, 'test.tsv')
        with open(file_name, 'w', encoding='utf-8') as f:
            for q, k in q_k_neighbor:
                f.write("{}\t{}\n".format(q, k))

    def pkl2tsv(self, max_neighbors=None):
        entity_nodes = {n for n in self.g1.get_nodes_by_attribute('type', 'entity') if n and str(n).strip()}
        entity_nodes = sorted(set(entity_nodes))
        passage_nodes = {n for n in self.g1.get_nodes_by_attribute('type', 'original_text') if n and str(n).strip()}

        triple = []
        sep = "|'|"

        for edge in self.g1.get_edges():
            u, v, attr = edge
            if u not in entity_nodes or v not in entity_nodes:
                continue

            rel_name = attr.get('relation') if isinstance(attr, dict) else None
            if not (isinstance(rel_name, str) and rel_name.strip()):
                continue

            try:
                neighbor_u = list(self.g1.successors(u)) + list(self.g1.predecessors(u))
            except Exception as e:
                logger.error(f"exception happened because:{e}")
                neighbor_u = []

            if v in neighbor_u:
                neighbor_u.remove(v)

            try:
                neighbor_v = list(self.g1.successors(v)) + list(self.g1.predecessors(v))
            except Exception as e:
                logger.error(f"exception happened because:{e}")
                neighbor_v = []

            if u in neighbor_v:
                neighbor_v.remove(u)

            neighbor_u = [n for n in neighbor_u if n not in passage_nodes]
            neighbor_v = [n for n in neighbor_v if n not in passage_nodes]

            neighbor_u_list = make_ordered_neighbor_list(u, neighbor_u, max_neighbors)
            neighbor_v_list = make_ordered_neighbor_list(v, neighbor_v, max_neighbors)
            q_and_neighbor = sep.join(neighbor_u_list)
            k_and_neighbor = sep.join(neighbor_v_list)
            triple.append((q_and_neighbor, k_and_neighbor))

        random.shuffle(triple)
        train_split = int(len(triple) * 0.7)
        train_triples = triple[:train_split]
        val_triples = triple[train_split:]

        def write_tsv(data, file_name):
            with open(file_name, 'w', encoding='utf-8') as f:
                for q, k in data:
                    f.write(f"{q}\t{k}\n")
            logger.info(f"Wrote {len(data)} triples to {file_name}")

        write_tsv(train_triples, os.path.join(self.out_dir, 'train.tsv'))
        write_tsv(val_triples, os.path.join(self.out_dir, 'valid.tsv'))

    def count_parts(self):
        self.neighborsHT = {h: {t for ts in rt.values() for t in ts} for h, rt in self.neighbors.items()}
        unseen = set(range(self.num_ent))
        self.parts = []
        while len(unseen) != 0:
            self.parts.append(set())
            queue = [random.choice(list(unseen))]  # nosec B311
            while len(queue) != 0:
                root = queue.pop()
                unseen -= {root}
                self.parts[-1].add(root)
                if root in self.neighborsHT:
                    queue.extend([t for t in self.neighborsHT[root] if t in unseen])
        self.parts.sort(key=len)
        logger.info(f"Found {len(self.parts)} parts, with sizes {[len(p) for p in self.parts]}")

    def extract_subgraph(self, hops):
        degree_ave = max(math.ceil(len(self.data) / self.num_ent), 2)
        max_ents_in_subgraph = degree_ave * math.ceil(degree_ave / 2) * math.ceil((degree_ave / 3))
        min_ents_in_subgraph = max_ents_in_subgraph // 3

        degree = {e: len(self.neighborsHT[e]) for e in self.neighborsHT.keys()}

        def select_neighbors(entity):
            nodes = [entity]
            length = [0]
            for jump in range(hops):
                length.append(len(nodes))
                for _idx in range(length[jump], length[jump + 1]):
                    parent = nodes[_idx]
                    nodes += list({_t for _t in self.neighborsHT[parent] if _t not in nodes})
                    if len(nodes) >= min_ents_in_subgraph:
                        return True

            return False

        def shift(cnt):
            return (degree_ave / cnt / 2) ** 0.5

        un_expand_ents = set(range(self.num_ent))
        subgraph_ents = ddict(lambda: [set() for _ in range(hops + 2)])
        small_subgraph_ents = ddict(set)
        idx = 0
        while idx < len(self.parts) and len(self.parts[idx]) < max_ents_in_subgraph:
            root = random.choice(list(self.parts[idx]))  # nosec B311
            while (idx < len(self.parts)) and (
                len(small_subgraph_ents[root]) + len(self.parts[idx]) < max_ents_in_subgraph
            ):
                small_subgraph_ents[root] |= self.parts[idx]
                un_expand_ents -= self.parts[idx]
                idx += 1

        choices = {e for e in un_expand_ents if select_neighbors(e)}
        while len(choices) != 0:
            logger.info(f"1.init: {len(choices)}")
            choice = list(choices)
            random.shuffle(choice)
            for ent in choice:
                if degree_ave / 2 < degree[ent] <= 2 * degree_ave:
                    break
            else:
                ent = choice[0]

            subgraph_ents[ent][0] |= {ent}
            subgraph_ents[ent][1] |= {ent}
            expend = set()
            for hop in range(1, hops + 1):
                for h in subgraph_ents[ent][hop]:
                    if h not in un_expand_ents:
                        continue
                    un_expand_ents -= {h}
                    choices -= {h}
                    expend |= {h}

                    for t in self.neighborsHT[h]:
                        if hop != 1 and (
                            t in subgraph_ents[ent][0] or random.random() > shift(len(subgraph_ents[ent][hop]))  # nosec B311
                        ):
                            continue
                        subgraph_ents[ent][hop + 1].add(t)
                        subgraph_ents[ent][0].add(t)
            if len(subgraph_ents[ent][hops + 1]) < degree_ave and len(subgraph_ents[ent][3]) < (
                degree_ave if hops == 3 else degree_ave / 2
            ):
                un_expand_ents |= expend
                del subgraph_ents[ent]

        choices = set(un_expand_ents)
        while len(choices) != 0:
            logger.info(f"2.connect {len(choices)}  end='    \n'")
            ent = random.choice(list(choices))  # nosec B311
            choices -= {ent}
            for hop in range(2, hops + 1):
                for root in sorted(subgraph_ents.keys(), key=lambda x: len(subgraph_ents[x][0])):
                    if len(self.neighborsHT[ent] & subgraph_ents[root][hop]) != 0:
                        break
                else:
                    continue

                subgraph_ents[root][hop + 1] |= {ent}
                subgraph_ents[root][0] |= {ent}
                un_expand_ents -= {ent}

                if hops == 3 and hop == 2:
                    subgraph_ents[root][hop + 2] |= self.neighborsHT[ent]
                    subgraph_ents[root][0] |= self.neighborsHT[ent]

        choices = set(un_expand_ents)

        while len(choices) != 0:
            logger.info(f"3.mor_hop {len(choices)}")
            ent = random.choice(list(choices))  # nosec B311
            choices -= {ent}
            for root in sorted(subgraph_ents.keys(), key=lambda x: len(subgraph_ents[x][0])):
                if ent in subgraph_ents[root][0]:
                    break
            else:
                continue

            un_expand_ents -= {ent}
            subgraph_ents[root][-1] -= {ent}
            subgraph_ents[root][0] |= self.neighborsHT[ent]

        try_times = len(un_expand_ents) * 5
        while len(un_expand_ents) != 0 and (try_times := try_times - 1) > 0:
            logger.info(f"4.reamin {len(un_expand_ents)}-{try_times}")
            ent = random.choice(list(un_expand_ents))  # nosec B311
            for root in sorted(subgraph_ents.keys(), key=lambda x: len(subgraph_ents[x][0])):
                if len(self.neighborsHT[ent] & subgraph_ents[root][0]) != 0:
                    break
            else:
                continue

            un_expand_ents -= {ent}
            subgraph_ents[root][-1] -= {ent}
            subgraph_ents[root][0] |= self.neighborsHT[ent] | {ent}
        sub_len = [len(x[0]) for x in subgraph_ents.values()]

        if self.para.del_exceed:
            for root in subgraph_ents:
                del_cnt = len(subgraph_ents[root][0]) - max_ents_in_subgraph
                if del_cnt > 0:
                    del_ents = list(subgraph_ents[root][-1])
                    random.shuffle(del_ents)
                    del_ents = del_ents[:del_cnt]
                    subgraph_ents[root][0] -= set(del_ents)

        ht2r = ddict(set)
        for h, r, t in self.data:
            ht2r[(h, t)].add(r)
        allHTs = set(ht2r.keys())
        subgraph = {root: subs[0] for root, subs in subgraph_ents.items()}
        subgraph.update({k: v for k, v in small_subgraph_ents.items() if len(v) != 0})
        self.subgraph = dict()
        for root, subs in subgraph.items():
            hts = torch.tensor(list(subs)).view(-1, 1)
            cnt = hts.shape[0]
            hts = torch.cat([hts.repeat(cnt, 1), hts.repeat(1, cnt).view(-1, 1)], dim=-1)
            hts = {tuple(ht) for ht in hts.tolist()} & allHTs
            self.subgraph[root] = torch.LongTensor([(ht[0], r, ht[1]) for ht in hts for r in ht2r[ht]])

        sub_triple_len = [hrt.shape[0] for hrt in self.subgraph.values()]
        return max(sub_triple_len), max(sub_len), len(self.subgraph), len(un_expand_ents)
