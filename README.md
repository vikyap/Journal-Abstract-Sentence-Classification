# Journal-Abstract-Sentence-Classification

# Goal
Classify sentences from journal abstracts which appear in a sequential order based on their roles (Objectives, Method, Results). 

# Preprocessing 
- Split paragraph texts into individual lines 
- Split lines into individual characters
- Convert data into dataframe with columns: Target, Text, Abstract_Line_Number, Total_lines_in_abstract
- One Hot encode labels ("target" column)
- Label encode labels ("target" column) 
- Calculate 95 percentile of line lengths for output length of each text vectorized line 
- Calculate 95 percentile of character lengths for output length of each character vectorized line

# Datasets 
- Create datasets (Train, Validation, Test) of batch size 32 
- Use prefetch to decrease training time 

# Model_0 (Baseline): TF-IDF Classifier

# Model_1: Conv1D with Token Embedding
- TextVectorization layer to convert text to numbers 
- Token embedding layer to create feature vectors 

# Model_2: Feature extraction transfer learning 
- Pretrained model for transfer learning 
- TensorFlow Hub Universal Sentence Encoder 

# Model_3: Conv1D with Character Embedding
- Character Vectorization layer to convert characters to numbers 
- Character embedding layer to create feature vectors 

# Model_4: Token Embedding + Character Embedding hybrid 
- Create token embedding input 
- Create character embedding input 
- Concatenate token and character inputs

# Model_5: Token Embedding + Character Embedding + Positional Embedding 
- Create token embedding input
- Create character embedding input 
- Create positional embedding input 
- Concatenate token, character, and positional inputs 
