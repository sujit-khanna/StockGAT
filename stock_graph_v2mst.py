import dgl
from dgl.data import DGLDataset
import torch
import pandas as pd
import glob
import collections

import numpy as np
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


EDGE_THRESH = 0
LABEL_THRESH = 0.0005
DIST_THRESH = 0
TECH_IND = ["rsi", "macd", "adx"]
FUN_IND = ["rating_Score", "pe"]
START_DATE = "2017-01-01"

TRAIN, VALID, TEST = ("2017-01-01", "2018-12-31"), ("2019-01-01", "2019-08-31"), ("2019-09-01", "2021-04-31")


def minimum_spanning_tree(correl_matrix):
    corr_mat = correl_matrix.copy()
    n_vertices = corr_mat.shape[0]
    mst_edges = []
    visited_nodes = [0]
    visited = 1
    diag_indices = np.arange(n_vertices)
    corr_mat[diag_indices, diag_indices] = np.inf
    while visited != n_vertices:
        new_edge = np.argmin(corr_mat[visited_nodes], axis=None)
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_nodes[new_edge[0]], new_edge[1], corr_mat[visited_nodes[new_edge[0]], new_edge[1]]]
        mst_edges.append(new_edge)
        visited_nodes.append(new_edge[1])
        corr_mat[visited_nodes, new_edge[1]] = np.inf
        corr_mat[new_edge[1], visited_nodes] = np.inf
        visited += 1
    return np.vstack(mst_edges)


class StockNetworkNodeLabels(DGLDataset):
    def __init__(self, mode="train", save_path=None):
        self.mode = mode
        super().__init__(name='stock_network_node_classification', save_dir=save_path)

    def _align(self, nodes, fundamentals, technicals):
        """
        This method ensure all datasets are aligned together
        :return:
        """
        tickers = list(nodes.keys())
        graph_dict = collections.defaultdict(dict)
        for ticker in tickers:
            ret_df = nodes[ticker][["Date","weekly_ret"]].iloc[1:]
            ret_df.loc[ret_df["weekly_ret"]>=LABEL_THRESH, "labels"] = 2
            ret_df.loc[ret_df["weekly_ret"] <= -LABEL_THRESH, "labels"] = 0
            ret_df["labels"] = ret_df["labels"].fillna(1)

            tech_df = technicals[ticker][["Date"] + TECH_IND]
            fun_df = fundamentals[ticker][["date"] + FUN_IND]
            fun_df = fun_df.rename(columns={"date":"Date"})

            ## alter this to create left joins only for data consistency and do ffill() to remove NaNs
            full_df = pd.merge(ret_df[["Date","labels"]], tech_df, on="Date", how="left").ffill()
            full_df = pd.merge(full_df, fun_df, on="Date", how="left").ffill()
            full_df = full_df.set_index("Date")
            full_df = full_df.fillna(0)

            if self.mode=="train":
                graph_dict[ticker] = full_df.loc[(full_df.index>=TRAIN[0]) & (full_df.index<=TRAIN[1])]
            elif self.mode == "valid":
                graph_dict[ticker] = full_df.loc[(full_df.index >= VALID[0]) & (full_df.index <= VALID[1])]
            elif self.mode == "test":
                graph_dict[ticker] = full_df.loc[(full_df.index >= TEST[0]) & (full_df.index <= TEST[1])]

        return graph_dict

    def _preprocess(self, edges, graph_dict):
        df_d = graph_dict["AAPL"]
        date_list = df_d.index.tolist()

        full_graph_ids = sorted(list(edges.keys()))
        graph_ids = sorted(list(set(date_list) & set(full_graph_ids)))
        data_dict = collections.defaultdict(dict)
        for idx in graph_ids:
            wts = edges[idx]
            wts = wts.set_index("Unnamed: 0")
            wts_ary = wts.values
            ticker_list = wts.columns.tolist()


            XX, YY = np.meshgrid(np.arange(wts_ary.shape[1]), np.arange(wts_ary.shape[0]))
            table = np.vstack((wts_ary.ravel(), XX.ravel(), YY.ravel())).T
            edge_df = pd.DataFrame({"src": table[:, 1].astype(int), "dest": table[:, 2].astype(int), "wts": table[:, 0]})
            mst_ary = minimum_spanning_tree(wts_ary)
            mst_ary_copy = np.c_[mst_ary[:, 1], mst_ary[:, 0], mst_ary[:, 2]]
            edge_ary = np.append(mst_ary, mst_ary_copy, axis=0)
            edge_df = pd.DataFrame({"src": edge_ary[:, 0].astype(int), "dest": edge_ary[:, 1].astype(int), "wts": edge_ary[:, 2]})

            feat_list = []
            for ticker in ticker_list:
                df = graph_dict[ticker].loc[idx]
                feat_list.append(df.to_frame().T)
            feat_df = pd.concat(feat_list, axis=0)
            data_dict[idx] = {"wts":edge_df, "features":feat_df.fillna(method='ffill')}
        return data_dict

    def process(self):
        self.graphs = []
        stock_edges = {key.split("/")[-1].split(".")[0].split("_")[-1]: pd.read_csv(key)
                    for key in glob.glob("./data/write_files/csv/dependency/*")}

        tech_features = {key.split("/")[-1].split(".")[0]:pd.read_csv(key) for key in
                         glob.glob("./data/write_files/csv/indicators/*")}

        fun_features = {key.split("/")[-1].split(".")[0]: pd.read_csv(key) for key in
                         glob.glob("./data/write_files/csv/ratings/*")}

        node_labels = {key.split("/")[-1].split(".")[0]: pd.read_csv(key) for key in
                         glob.glob("./data/write_files/csv/labels/*")}

        graph_dict = self._align(nodes=node_labels, fundamentals=fun_features, technicals=tech_features)

        self.data_dict = self._preprocess(edges=stock_edges, graph_dict=graph_dict)

        graph_ids = list(self.data_dict.keys())

        for g in graph_ids:
            graph_data = self.data_dict[g]
            edges_src = torch.from_numpy(graph_data['wts']["src"].values)
            edges_dest = torch.from_numpy(graph_data['wts']["dest"].values)
            edges_wts = torch.from_numpy(graph_data['wts']["wts"].values)

            targets = torch.from_numpy(graph_data['features']["labels"].astype('category').cat.codes.to_numpy())
            node_labels  = torch.nn.functional.one_hot(targets.to(torch.int64), 3)

            node_rsi = torch.from_numpy(graph_data['features']["rsi"].values)
            node_macd = torch.from_numpy(graph_data['features']["macd"].values)
            node_adx = torch.from_numpy(graph_data['features']["adx"].values)
            node_rating = torch.from_numpy(graph_data['features']["rating_Score"].values)
            node_pe = torch.from_numpy(graph_data['features']["pe"].values)
            single_graph = dgl.graph((edges_src, edges_dest), num_nodes=graph_data['features'].shape[0])
            single_graph.edata["weight"] = edges_wts
            single_graph.ndata["label"] = node_labels
            single_graph.ndata["feat"] = torch.stack([node_rsi, node_macd, node_adx, node_rating, node_pe], dim=0).t()
            single_graph = dgl.add_self_loop(single_graph)
            self.graphs.append(single_graph)

    def save(self):
        graph_list_path = os.path.join(self.save_path, f'{self.mode}_dgl_graph_list.bin')
        g_path = os.path.join(self.save_path, f'{self.mode}_dgl_graph.bin')
        info_path = os.path.join(self.save_path, f'{self.mode}_info.pkl')
        save_graphs(graph_list_path, self.graphs)
        save_graphs(g_path, self.graphs[0])
        save_info(info_path, {'labels': [0,1,2], 'feats': ["3 tech", "3 fundamental"]})

    def load(self):
        graph_list_path = os.path.join(self.save_path, f'{self.mode}_dgl_graph_list.bin')
        g_path = os.path.join(self.save_path, f'{self.mode}_dgl_graph.bin')
        info_path = os.path.join(self.save_path, f'{self.mode}_info.pkl')
        self.graphs = load_graphs(graph_list_path)[0]
        g, _ = load_graphs(g_path)
        self.graph = g[0]
        info = load_info(info_path)
        self._labels = info['labels']
        self._feats = info['feats']

    @property
    def num_labels(self):
        return 3

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)


if __name__ == '__main__':

    modes = ["train", "valid", "test"]
    save_path = "./data/dgl_graphs/multifeature_mst/"
    for mode in modes:
        g = StockNetworkNodeLabels(mode=mode, save_path=save_path)
        g.save()

