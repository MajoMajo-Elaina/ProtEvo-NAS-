from ruamel.yaml import YAML
from logzero import logger
from pathlib import Path
import warnings
import torch
import numpy as np
from dgl.dataloading import GraphDataLoader
from DPFunc.data_utils import get_pdb_data, get_mlb, get_inter_whole_data
from DPFunc.models import combine_inter_model
from DPFunc.objective import AverageMeter
from DPFunc.model_utils import test_performance_gnn_inter, merge_result, FocalLoss
from DPFunc.evaluation import new_compute_performance_deepgoplus
import os
import pickle as pkl
import click
from tqdm.auto import tqdm
from DPFunc.logger import gnn_architecture_performance_save,\
                                  test_performance_save,\
                                  model_save
class StackGcn(object):
    def __init__(self,architecture,data_cnf,gpu_number,epoch_number,pre_name
                 ):
        self.architecture=architecture
        self.data_cnf=data_cnf
        self.data_name=data_cnf
        self.gpu_number=gpu_number
        self.epoch_number=int(epoch_number)
        self.pre_name=pre_name
    def fit(self):
        yaml = YAML(typ='safe')
        ont = self.data_cnf
        self.data_cnf, model_cnf = yaml.load(Path('./configure/{}.yaml'.format(self.data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_name, model_name = self.data_cnf['name'], model_cnf['name'] 
        run_name = F'{model_name}-{data_name}'
        logger.info('run_name: {}'.format(run_name))

        self.data_cnf['mlb'] = Path(self.data_cnf['mlb'])
        self.data_cnf['results'] = Path(self.data_cnf['results'])
        logger.info(F'Model: {model_name}, Dataset: {data_name}')

        train_pid_list, train_graph, train_go = get_pdb_data(pid_list_file = self.data_cnf['train']['pid_list_file'],
                                                             pdb_graph_file = self.data_cnf['train']['pid_pdb_file'],
                                                             pid_go_file = self.data_cnf['train']['pid_go_file'], 
                                                             train = self.data_cnf['train']['train_file_count'])
        logger.info('train data done')
        valid_pid_list, valid_graph, valid_go = get_pdb_data(pid_list_file = self.data_cnf['valid']['pid_list_file'],
                                                             pdb_graph_file = self.data_cnf['valid']['pid_pdb_file'],
                                                             pid_go_file = self.data_cnf['valid']['pid_go_file'])
        logger.info('valid data done')
        test_pid_list, test_graph, test_go = get_pdb_data(pid_list_file = self.data_cnf['test']['pid_list_file'],
                                                          pdb_graph_file = self.data_cnf['test']['pid_pdb_file'],
                                                          pid_go_file = self.data_cnf['test']['pid_go_file'])
        logger.info('test data done')
        train_interpro = get_inter_whole_data(train_pid_list, self.data_cnf['base']['interpro_whole'], self.data_cnf['train']['interpro_file'])
        valid_interpro = get_inter_whole_data(valid_pid_list, self.data_cnf['base']['interpro_whole'], self.data_cnf['valid']['interpro_file'])
        test_interpro = get_inter_whole_data(test_pid_list, self.data_cnf['base']['interpro_whole'], self.data_cnf['test']['interpro_file']) 

        mlb = get_mlb(Path(self.data_cnf['mlb']), train_go)
        labels_num = len(mlb.classes_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_y = mlb.transform(train_go).astype(np.float32)
            valid_y = mlb.transform(valid_go).astype(np.float32)
            test_y  = mlb.transform(test_go).astype(np.float32)

        idx_goid = {}
        goid_idx = {}
        for idx, goid in enumerate(mlb.classes_):
            idx_goid[idx] = goid
            goid_idx[goid] = idx
        
        train_data = [(train_graph[i], i, train_y[i]) for i in range(len(train_y))]
        train_dataloader = GraphDataLoader(
            train_data,
            batch_size=64,
            drop_last=False,
            shuffle=True)

        valid_data = [(valid_graph[i], i, valid_y[i]) for i in range(len(valid_y))]
        valid_dataloader = GraphDataLoader(
            valid_data,
            batch_size=64,
            drop_last=False,
            shuffle=False)

        test_data = [(test_graph[i], i, test_y[i]) for i in range(len(test_y))]
        test_dataloader = GraphDataLoader(
            test_data,
            batch_size=64,
            drop_last=False,
            shuffle=False)

        del train_graph
        del test_graph
        del valid_graph
        print(len(train_dataloader))
        logger.info('Loading Data & Model')
        model = combine_inter_model(inter_size=train_interpro.shape[1], 
                                    graph_size=1280, 
                                    label_num=labels_num,
                                    architecture=self.architecture).to(device)
        logger.info(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
        loss_fn = FocalLoss()
        used_model_performance = np.array([-1.0]*3)
        max_aupr=0
        for e in range(self.epoch_number):
            model.train()
            train_loss_vals = AverageMeter()
            for batched_graph, sample_idx, labels in tqdm(train_dataloader, leave=False):
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                inter_features = (torch.from_numpy(train_interpro[sample_idx].indices).to(device).long(), 
                                torch.from_numpy(train_interpro[sample_idx].indptr).to(device).long(), 
                                torch.from_numpy(train_interpro[sample_idx].data).to(device).float())
                feats = batched_graph.ndata['x']

                logits = model(inter_features, batched_graph, feats)

                loss = loss_fn(logits, labels)
                train_loss_vals.update(loss.item(), len(labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            plus_fmax, plus_aupr, plus_t, df, valid_loss_avg = test_performance_gnn_inter(model, valid_dataloader, valid_pid_list, valid_interpro, valid_y,    idx_goid, goid_idx, ont, device)
            print('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(e, 
                                                                                                                                train_loss_vals.avg,
                                                                                                                                valid_loss_avg,
                                                                                                                                plus_fmax, 
                                                                                                                                plus_aupr, 
                                                                                                                                plus_t, 
                                                                                                                                df.shape))
            if plus_aupr > max_aupr:
                max_aupr = plus_aupr
            valid_performance=max_aupr
        gnn_architecture_performance_save(self.architecture, valid_performance, self.data_name)
        return max_aupr