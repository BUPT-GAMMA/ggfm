from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleFSManager:
    def __init__(self, class_ind, data_ind, k_shot, q_query, n_way, min_k_shot=None, min_n_way=None, task_level=None):
        self.class_ind = class_ind
        self.data_ind = data_ind
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_way = n_way
        self.min_n_way = min_n_way
        self.min_k_shot = min_k_shot
        self.graph_mode = 'graph' in task_level

    def get_few_shot_idx(self):
        if self.min_n_way is not None:
            n_way = np.random.permutation(np.arange(self.min_n_way, self.n_way))[0]
        else:
            n_way = self.n_way
        if self.min_k_shot is not None:
            k_shot = np.random.permutation(np.arange(self.min_k_shot, self.k_shot))[0]
        else:
            k_shot = self.k_shot

        target_classes_ind = self.get_target_cls_ind(n_way, k_shot)
        target_classes = self.class_ind[target_classes_ind]
        samples = []
        for idx in target_classes_ind:
            samples.append(np.random.choice(self.data_ind[idx], k_shot + self.q_query))
        return np.array(samples), target_classes

    def get_target_cls_ind(self, n_way, k_shot):
        if self.graph_mode:
            rand_class = np.random.permutation(len(self.class_ind)//2)[0]
            target_classes_ind = np.array([rand_class, rand_class + len(self.class_ind)//2])
            while min(len(self.data_ind[self.class_ind[rand_class]]), len(self.data_ind[self.class_ind[rand_class + len(self.class_ind)//2]])) < k_shot + self.q_query:
                rand_class = np.random.permutation(len(self.class_ind) // 2)[0]
                target_classes_ind = np.array([rand_class, rand_class + len(self.class_ind) // 2])
        else:
            target_classes_ind = np.random.permutation(len(self.class_ind))[:n_way]
        return target_classes_ind

