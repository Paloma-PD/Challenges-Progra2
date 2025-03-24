# Instructions for use
The first step is to clone the repository located at https://github.com/Paloma-PD/Programacion-2/tree/main/Challenge_1 using the following commands:

    git clone https://github.com/Paloma-PD/Programacion-2/Challenge_1.git
    cd Challenge_1

Once the repository is cloned, you will notice that the folder has a _txt_ file called _requirements_. This file contains the minimum libraries that the environment where the code will run must have. To install the libraries, you can do so using the command

    pip install – requirements.txt
** To run it, remember that you must be in the directory where the file is located.

Once executed and the libraries have been installed, or if applicable, it has been verified that they are present, the code will be executed, this code is located in the **src** folder. Entering this folder, you will find four .py files, which are described below:
1. _preprocessing_: Here, the data will be loaded and preprocessed, that is, unnecessary columns will be eliminated, the independent and dependent variables will be defined, and finally the data will be scaled.
2. _model_training_: This script allows the data to be divided into training and test data, as well as initializing and training a model (RandomForest), to finally generate predictions with the model.
3. _evaluation_: This script is designed to calculate the model metrics (accuracy, precision, recall, f1-score). as well as calculating and graphing the confusion matrix and the ROC curve. It is important to mention that the created graphs will be saved in .png image format.
4. _mlops_pipeline_: This is the *"main"* script on which the other scripts are concentrated. When it's executed, the data will be loaded, preprocessed, the model will be generated, the metrics will be calculated, and finally, the obtained metrics, report, and graphs will be recorded using the mlflow library.

Finally, it is mentioned that the code is added to the notebooks folder in an .ipynb file. This file serves as support for testing the model and/or exploring the data. For more details, see the documentation PDF.
