from types import SimpleNamespace
def parse():
    args_dict = {
        'emb_dim': 256,
        'conv_dim': 128,
        'k': 3,
        'p': 0.1,
        # 'lr': 0.01,  # 如果需要，可以取消注释
        # 'weight_decay': 5e-4  # 如果需要，可以取消注释
    }
    return SimpleNamespace(**args_dict)