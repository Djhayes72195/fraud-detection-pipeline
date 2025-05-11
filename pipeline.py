from data_processing.data_ingestion import load_days
from data_processing.preprocessor import preprocess
from modeling.train import train
from modeling.model_io import save


first_day = 1
last_day = 1
df = load_days(first_day, last_day)
df = preprocess(df)
model = train(df)
save(model, last_day)
# Save model off