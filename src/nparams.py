from models.cdan import CDAN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = CDAN()
    print("Total number of trainable parameters = ", count_parameters(model))