# Assignment 2: Transformer experiments

This is a project to implement a transformer model from first principles using the architecture discussed in the Attention Is All You Need Paper. The details of the implementation are carefully laid out in the simplicity of the models modules as they are implemented in order to build up into the MiniTransformer class at the end.

The code is split into different parts demarcated by line comments with the section names:
1. Setup: setting up from command line arguments and setting the hyperparameters.
2. Data Preparation: loading and preprocessing the dataset.
3. Model Architecture: defining the transformer model architecture, including the encoder and decoder layers.
4. Training Loop: implementing the training loop to train the transformer model on the dataset.
5. Evaluation: evaluating the performance of the trained model on a separate test dataset.
6. Inference and Saving: sample prediction and saving the model if needed

## Usage
1. Make a virtual environment ```python3 -m venv .venv``` and activate it ```source .venv/bin/activate```
2. Install pip packages ```pip3 install torch```
3. Run the transformer with default hyperparameters ```python3 mini_transformer.py```
4. Get help on the hyperparameter command-line application ```python3 mini_transformer.py -h```

## Experiment Results
All the experiment results for "ceteris paribus" hyperparameter application are in the logs subdirectory in the format ```{param}_{value}.log```. The final best version trained is in the final_train.log file.

## Note
The only language files currently loaded in are nr/Ndebele files hence any other language will not work for the time being until the files are uploaded

## Acknowledgements
Andrej Karpathy for the nanoGPT implementation which was key to my own implementation and understanding.

