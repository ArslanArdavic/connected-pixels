# connected-pixels
Self-Supervised Pre-Training of Models Processing Images as Graphs

# IMPORTANT
train.py accumulates gradients:
    -Batch size of 4096 is too much for GPU
    -Batch size of 256 is accumulated 16 times 

train_direct.py designed for multi-GPU case:
    -Batch size of 4096 is enough 