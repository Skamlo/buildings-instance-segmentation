def compute_param_norm(model):
    total = 0.0
    for p in model.parameters():
        if p is not None:
            param_norm = p.data.norm(2)
            total += (param_norm.item() ** 2)
    return total ** 0.5
