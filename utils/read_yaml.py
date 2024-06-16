import yaml
def read_yaml(fpath=None):
    with open(fpath, mode="r", encoding='utf-8') as file:
        yml = yaml.safe_load(file)
        return yml