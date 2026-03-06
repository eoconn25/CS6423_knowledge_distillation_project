import torch

def save_supernet(model, path="supernet_checkpoint.pth"):
    # It's best practice to save the state_dict
    torch.save({
        'state_dict': model.state_dict(),
        'width_mult_list': [0.65, 0.8, 1.0, 1.2],
        'search_space': {
            'res': [128, 160, 190, 224],
            'depth': [2, 3, 4],
            'exp': [3, 4, 6]
        }
    }, path)
    print(f"Supernet saved to {path}")