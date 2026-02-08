# DSformer: A Dual-Stream Transformer for Crystal Representation Learning with Periodic Atomic Priors
The official implementation of DSformer: A Dual-Stream Transformer for Crystal Representation Learning with Periodic Atomic Priors.

![image](https://github.com/ablappno1/DSformer/blob/main/pics/PAE.png)
![image](https://github.com/ablappno1/DSformer/blob/main/pics/DS%20Attention.PNG)
![image](https://github.com/ablappno1/DSformer/blob/main/pics/DS%20Attention%20Block.PNG)
![image](https://github.com/ablappno1/DSformer/blob/main/pics/DSformer%20Architecture.PNG)

# Results
![image](https://github.com/ablappno1/DSformer/blob/main/pics/DSformer%20Result.PNG)

# Environment Setup
Use Command below:
```
conda env create -f environment.yml
```

# Dataset
You can obtain the dataset using the following commands:
```
cd /yourpath/data
python download_megnet_elastic.py
python downlad_jarvis.py
```

# Training
You can start training using the following commands:
```
cd /yourpath
CUDA_VISIBLE_DEVICES=0 python train.py -p latticeformer/default.json \
    --save_path yoursavepath \
    --n_epochs 500 \
    --experiment_name dsformer \
    --num_layers 4 \
    --value_pe_dist_real 64 \
    --target_set jarvis__megnet \
    --targets formation_energy \
    --batch_size 128 \
    --lr 0.0005 \
    --model_dim 128 \
    --embedding_dim 128 \
```

# Testing
You can test the model using the following commands:
```
. demo.sh
```
