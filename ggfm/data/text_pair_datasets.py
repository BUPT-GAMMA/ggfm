"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
from ggfm.data.base_dataset import BatchIterableDataset


class TextPairDataset(BatchIterableDataset):
    def __init__(self, datasets_dir, max_length, vocab_size, mode):
        super(TextPairDataset, self).__init__(datasets_dir, mode)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def _train_data_parser(self, data):
        # 训练阶段使用
        user_id = data[0][0]
        embedding = np.array(data[0][1].split(','), dtype=np.float32)
        node_input = data[0][2]
        if len(node_input)>=1000:
            node_input=node_input[:1000]
        neighbour_input = data[0][3]
        title = data[0][4]

        text_input = 'The summary of this article is as follows:' + node_input+ '\nThere are some papers that cite this paper.' + neighbour_input

        return user_id, embedding, text_input, title

    def __len__(self):
        return self.row_count
