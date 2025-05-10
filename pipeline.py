from data_processing.data_ingestion import load_days
from data_processing.preprocessor import preprocess
from model.train import train

df = load_days(1, 1)
df = preprocess(df)
model = train(df)
# Save model off