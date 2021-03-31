# NNTI_Final_Project
The final project for the NNTI course.

All the tasks are provided as both jupyter notebooks and .py files. 

In order to run the python files for:

 - Hindi dataset: python3 Task1_Word_Embeddings.py && python3 prepare_data.py hindi && python3 classify.py hindi
 the first script trains the word2vec model on the hindi dataset, the second one gets the embedings and prepares the data for the CNN and the third
 one does the classification from scratch. 

- Bengali dataset: python3 get_embeddings_bangali.py && python3 prepare_data.py bangali && python3 classify.py bangali
 the first script trains the word2vec model on the bangali dataset, the second one gets the embedings and prepares the data for the CNN and the third
 one does the classification by fine-tuning the model trained on Hindi dataset.
 

 
 (note that the prepare_data.py and classify.py scripts are the same for both datasets and you only need to change the argument from 'hinde' to 'bangali')

