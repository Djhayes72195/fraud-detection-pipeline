from modeling.train import train
from modeling.model_io import save

def retrain_pipeline():
    df = construct_new_training_set()
    model = train(df)
    save(model)

def construct_new_training_set():
    pass