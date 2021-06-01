import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from GAT import GAT
import gat_parameters as mp
# from graphs.stock_graph_v1 import StockNetworkNodeLabels
from stock_graph_v2mst import StockNetworkNodeLabels
from dgl.dataloading import GraphDataLoader
torch.manual_seed(0)


def test_network(feats, model, subgraph, labels, multilabel_loss):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = multilabel_loss(output, labels.float())
        predict = np.argmax(output.data.cpu().numpy(), axis=1)
        true_labels = np.argmax(labels.data.cpu().numpy(), axis=1)
        f1_val = f1_score(true_labels, predict, average='macro')
        accuracy = accuracy_score(true_labels,predict)
        return f1_val, loss_data.item(), accuracy, predict, true_labels


def test_model(g, test_dataloader, num_feats, num_labels, device, attention_heads, model_load_path):
    multilabel_loss = torch.nn.BCEWithLogitsLoss()
    test_f1_list, test_acc_list = [], []
    y_preds_list, y_true_list = [], []
    for batch, subgraph in enumerate(test_dataloader):
        subgraph = subgraph.to(device)
        model = GAT(g, mp.NUM_LAYERS, num_feats, mp.HIDDEN_UNITS, num_labels, attention_heads, F.elu, mp.IN_DROP,
                mp.ATTENION_DROP, mp.LEAKY_ALPHA, mp.RESIDUAL).to(device)
        model.load_state_dict(torch.load(model_load_path))
        f1_val, test_loss, test_acc, y_preds, y_true = test_network(subgraph.ndata['feat'], model, subgraph, subgraph.ndata['label'], multilabel_loss)
        test_f1_list.append(f1_val)
        test_acc_list.append(test_acc)
        y_preds_list.append(y_preds.tolist())
        y_true_list.append(y_true.tolist())

    print(f"Test Set F-1 Score {np.array(test_f1_list).mean()} Accuracy {np.array(test_acc_list).mean()}")
    y_preds_ary = np.hstack(y_preds_list)
    y_true_ary = np.hstack(y_true_list)
    class_names = ['Sell', 'Hold', 'Buy']
    print(classification_report(y_true_ary.flatten(), y_preds_ary.flatten(), target_names=class_names))


if __name__ == '__main__':

    save_path = "./data/dgl_graphs/multifeature_mst/"

    device = torch.device("cpu")

    model_save_path = f"./saved_models/stock_gat_v1_ep_{mp.EPOCHS}_lr_{mp.L_R}_batch_{mp.BATCH_SIZE}.pt"

    train_dataset = StockNetworkNodeLabels(mode='train', save_path=save_path)
    valid_dataset = StockNetworkNodeLabels(mode='valid', save_path=save_path)
    test_dataset = StockNetworkNodeLabels(mode='test', save_path=save_path)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=mp.BATCH_SIZE)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=mp.BATCH_SIZE)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=mp.BATCH_SIZE)
    g = train_dataset[0]
    num_labels = train_dataset.num_labels
    num_feats = g.ndata['feat'].shape[1]


    attention_heads = ([mp.ATTENTION_HEADS] * mp.NUM_LAYERS) + [mp.NUM_OUT_HEADS]

    test_model(g, test_dataloader, num_feats, num_labels, device, attention_heads, model_save_path)
