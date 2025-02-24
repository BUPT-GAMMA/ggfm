# GraphTranslator

  

- Paper link: [GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks | Proceedings of the ACM Web Conference 2024](https://dl.acm.org/doi/abs/10.1145/3589334.3645682)

- Author's code repo: [alibaba/GraphTranslator: GraphTranslator:Aligning Graph Model to Large Language Model for Open-ended Tasks](https://github.com/alibaba/GraphTranslator)

  

# Dataset Statics

- IMDB
- DBLP
  

# Results

<table>

  <tr>

    <th>Task</th>

    <th colspan="2">Node Classification</th>

  </tr>

  <tr>

    <th>Evaluation Matrix</th>

    <th>macro-F1</th>
    
    <th>micro-F1</th>

  </tr>

  <tr>

    <td>IMDB</td>

    <td>0.2820</td>

    <td>0.3440</td>

  </tr>
 
  <tr>

    <td>DBLP</td>

    <td>0.1099</td>

    <td>0.17</td>

  </tr>

</table>

# GraphTranslator Training Process and LLM Fine-tuning
The GraphTranslator training process does not involve fine-tuning the large language model (LLM). As a result, the model does not fully understand the semantic meaning of numerical labels. This becomes especially evident when comparing performance on different datasets, such as:

IMDB Dataset: The labels in this dataset are of text type (e.g., "positive" or "negative"), which aligns well with the language understanding capabilities of the LLM. This helps GraphTranslator perform better on IMDB.

DBLP Dataset: In contrast, DBLP uses numerical labels (e.g., 0 or 1). Since GraphTranslator doesn't fine-tune the LLM to understand numerical labels, the model struggles with interpreting these labels, resulting in lower performance on the DBLP dataset.

In summary, GraphTranslator performs better on the IMDB dataset because text labels are more aligned with the LLM's strengths, whereas numerical labels in DBLP hinder the model's performance due to the lack of fine-tuning.

# running
## translator model
Download `bert-base-uncased.zip` from link and unzip it to `./ggfm/models`.
```
cd ./ggfm/models
git lfs install
git clone git@hf.co:Hualouz/Qformer
unzip bert-base-uncased.zip
```

## ChatGLM2-6B model
Download the ChatGLM2-6B model from link and insert it to ./ggfm/models
```
cd ./ggfm/models
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```

## pretrain
In `pretrain.py`, we 
- pretrain gnn model.
- produce summary about nodes and their neighbors.
- train stage 1
- train stage 2

