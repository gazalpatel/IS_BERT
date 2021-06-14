# Author: Gazal
# checkpoint: 17 march 2021

from torch.utils.data import DataLoader
import math, logging, re
from language_bert import models, losses
from language_bert import SentencesDataset, LoggingHandler, LanguageTransformer, InputExample
from language_bert.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import numpy as np


# -------- INPUT ---------------

# train set details: https://www.kaggle.com/rishisankineni/text-similarity
train_dataset_path = 'data/train.csv'

train_batch_size = 8
num_epochs = 1
evaluation_steps = 10
max_seq_length=128
model_name = "similarity_bert"
model_save_path  = 'model/'+ model_name





# --------- LOGS -------------------

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])



# ------------READING DATASETS ------------

# Reading dataset
df = pd.read_csv(train_dataset_path)
logging.info("data set: %s"%(df.shape[0]))
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
dev = df[~msk]

train_samples = []
for index, row in train.iterrows():
    s1 = row["description_x"].strip()
    s1 = re.sub(r'[^\w\s]','',s1)

    s2 = row["description_y"].strip()
    s2 = re.sub(r'[^\w\s]','',s2)

    label=0
    if row["same_security"]=="TRUE":
        label=1
    train_samples.append(InputExample(texts=[s1, s2], label=label))
logging.info("Train samples: %s" % (len(train_samples)))

dev_samples = []
for index, row in dev.iterrows():
    s1 = row["description_x"].strip()
    s1 = re.sub(r'[^\w\s]','',s1)

    s2 = row["description_y"].strip()
    s2 = re.sub(r'[^\w\s]','',s2)

    label=0
    if row["same_security"]=="TRUE":
        label=1

    dev_samples.append(InputExample(texts=[s1, s2], label=label))

logging.info("Dev samples: %s" % (len(dev_samples)))
    

# ------------ MODEL CONFIGS ---------------
# Read model

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
assert num_epochs>0
model_name = 'bert-base-uncased'
logging.info("Model %s will be created from based model %s" % (model_save_path, model_name))

word_embedding_model = models.Transformer(model_name)

cnn = models.CNN(in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(cnn.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = LanguageTransformer(modules=[word_embedding_model, cnn, pooling_model],  device='cpu')




# -----------PREPARE TRAIN DATASET -------------------
# create the training dataset
logging.info("Read train dataset")
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MutualInformationLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension())



# -------- PREPARE Dev evalution DATASET ------------------
#Read dataset and use it as development set
logging.info("Read dev dataset")
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')




# ---------MODEL TRAINING + SAVE ------------

# Configure the training
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

logging.info("model saved at: %s" % (model_save_path))





# --------- TEST MODEL -----------
model = LanguageTransformer(model_save_path + "_epochend_0")
logging.info("model load successful: %s"%(model_save_path + "_epochend_0"))

test = [{"sentence1":"fund transfer using credit card is prohibited",
         "sentence2":"my credit card broke while doing the fund tranfer",
         "score":0.3},
        {"sentence1":"fund transfer using credit card is prohibited",
         "sentence2":"they are requested not to use ATM card for fund transfer",
         "score":0.8}
       ]

test_samples = []
for row in test:
    test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=row['score']))
    
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)


logging.info("--- model training completed successfully ---")