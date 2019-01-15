## Control GAD

Control GAD (Generalized Anxiety Disorder), is a little application that has the goal of interpreting the level of anxiety contained in a textual phrase.

[UNIR](https://www.unir.net/) in collaboration with [Psicobótica](http://psicobotica.com/), have been developing different programs to help people to recognize and handle mental illness through the use of Artificial Intelligence (IA). In this work, as part of these efforts, we have made a concept proof to validate if it is possible to identify the presence of anxiety in the written expressions of a person, and with this information determinate the frequency of worrying thoughts, in order to diagnostic the existence of GAD. The work made is described in this [post](https://medium.com/@facarpaulina/control-gad-eb72b469cac6).

This app trained a neural network with transcripts of a set of interviews made by the Institute for Creative Technologies, that contains a set of resources applied in order to analyze people with and without mental diseases.
The database used was [DAIC-WOZ](http://dcapswoz.ict.usc.edu/), currently, the corpus of this database is being shared on a case-by-case basis by request and for research purposes. This repository contains only the compiled dataset with the words as tokens.

> This work makes use of the word representations of the unsupervised algorithm from GloVe https://nlp.stanford.edu/projects/glove/ and the pre-trained word vectors of https://code.google.com/archive/p/word2vec/<br />
> These resources are not inside of this source code.<br />
> All code described below should be executed under `app` folder<br />


This repository contains the following folders: 
.<br />
├── app       # Executable files for reproduce the experiment<br />
├── data      # Inputs files to training of models and trained models<br />
├── datasets  # Methods for dataframes generation<br />
├── graphics  # Methods for display training graphs<br />
├── model     # Methods for creation and training of models<br />
├── tests     # Methods for test best model<br />
├── util      # Utility methods for pre-processing text<br />
└── Anxiety_training_notebook.ipynb # Original notebook of experimentation<br />


### Running the best model

With a text phrase as input, we will generate an output that contains an estimation of anxiety level present in the text between five categories (none, mild, moderate, moderately severe, severe)

To evaluate a phrase you should run:

```python3 App.py```
You will be prompted:
```What are you thinking about?: ```
After you enter a phrase the application will evaluate the level of anxiety present 

In:  <br />
```I am liking to spend holidays here, it is a nice place with funny activities and a lot of relaxing time```

Out: <br />
```Expected length: 10, actual length: 10```<br />
```**************************************************```<br />
```Phrase: I am liking to spend holidays here, it is a nice place with funny activities and a lot of relaxing time```<br />
```Predictions:  none: 98%, mild: 1%, moderate: 0%, moderately severe: 0%, severe: 1%```<br />
```Anxiety level:  none```<br />        
        
### Getting Started
The whole execution comprises a set of steps in order to generate a Deep Learning model based on the database DAIC-WOZ.

#### Prerequisites

In order to reproduce the steps to train the model you will need to download:

1. The DAIC-WOZ database [DAIC-WOZ](http://dcapswoz.ict.usc.edu/)
2. The word representations of the unsupervised algorithm from [GloVe](https://nlp.stanford.edu/projects/glove/)
2. The pre-trained word vectors of Google [word2vec](https://code.google.com/archive/p/word2vec/)

#### 1. Reading the transcriptions of interviews
Once we have the database from DAIC-WOZ downloaded, the next is to generate a dataframe that pre-processing the text, cleaning it and splitting in windows of a limited size defined in GlobalConstants.py of 10 words by default.
To do this, execute the command:
  
```python3 DataSetExecutor.py```

This command will create under `/data` folder, a csv file `phrases_lp.csv` which contains the tokenized sequences and that is the input to next step and a file tokenizer.pickle that will be used to prepare the embeddings for the training.  

#### 2. Training the models
Next we need to merge the dataframe of sequences, with the dataframe of PHQ-8 scores, in this step we had two options, first train the model with the whole records of dataframe, and second, train the model only with balanced records, that is a dataframe with the same number of records for each anxiety classification.
  
We experimented with the two models options, to generate them you should run:  
```python3 ModelTraining.py```

The trained models will be saved in `app` folder.

#### 3. Using models
To use one of the trained models, you should run the `App.py` file with  `google` or `glove` as argument to select the model.
  
```python3 App.py gloogle```
```python3 App.py glove```

You will be asked to write something you are thinking about, and with your input, the algorithm will tell you if your text has any level of anxiety
