from models.my_model import MyModel

def create_model(model_name):
    if model_name == 'mymodel':
        return MyModel()
    else:
        raise ValueError("model name is false, must choice from mvit_s, mvit_xs, mvit_xxs, resnet34")


