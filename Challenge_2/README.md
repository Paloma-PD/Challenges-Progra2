# Instructions for use
The first step is to clone the repository located at https://github.com/Paloma-PD/Programacion-2/tree/main/Challenge_2 using the following commands:

    git clone https://github.com/Paloma-PD/Challenges-Progra2.git
    cd Challenge_2

Once the repository is cloned, you will notice that the folder has a _txt_ file called *_*requirements_2*. This file contains the minimum libraries that the environment where the code will run must have. To install the libraries, you can do so using the command

    pip install – requirements_2.txt
** To run it, remember that you must be in the directory where the file is located.

Once executed and the libraries have been installed, or if applicable, it has been verified that they are present, the code will be executed, this code is located in the **src** folder. Entering this folder, you will find five .py files, which are described below:
1. _preprocessing_: Here, the data will be loaded and preprocessed. Also, in this module, the column of the data frame with which you will work is selected, by default the column is pros, the selected column will be transformed by cleaning punctuation marks, extra spaces, among others; in addition, a new data frame will be created only with the selected column to which a column will be added with the text without stopwords, another column with the lemmatized text, and finally, the n-grams of the text will be created, by default they will be created and graphed for a range of 2 to 5, it is important to mention that these plots will be saved in the plots folder.
2. _nlp_: In this module, an LDA model is created, which classifies the words into 4 themes (this value can be modified). In addition, sentiment analysis is performed, allowing the generation of cloud graphics for the words by theme. In the same way as the other graphics, these will be saved in the Plots folder.
3. _model_training_: This script allows the data to be divided into training and test data.
4. _evaluation_: This module, based on a list of models, adjusts and makes predictions, and also calculates metrics (accuracy, precision, recall, f1-score) for each model to finally save them in a data frame. ROC accuracy is also calculated to create the ROC curve graph and the confusion matrix graph. Both graphs are saved in the Plots folder.
5. _mlops_pipeline_: This is the *"main"* script on which the other scripts are concentrated. When it's executed, the data will be loaded, preprocessed, the models will be generated, the metrics will be calculated, and finally, the obtained metrics, report, and graphs will be recorded using the mlflow library.

Finally, it is mentioned that the code is added to the notebooks folder in an .ipynb file. This file serves as support for testing the models and/or exploring the data. For more details, see the documentation PDF.
******
*** It is important to mention that we are working with 250 records of the total, in order to make the execution of the code faster, you can always comment out line 47 of the _preprocessing_ module.
