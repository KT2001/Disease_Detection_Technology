# Disease_Detection_Technology

### Introduction

* This Project is to implement Image classification by Cnn to verify if AI could be leveraged in classification of diseases. Used the pretrained Model, ResNet50 to train on, radiology scans of diseases such as Covid19, Pneumonia, and Lung Opacity.
* Libraries Used In This Project tensorflow, Matplotlib, numpy, pandas, Keras, PIL, pathlib.
* This Model has achieved 99% Accuracy on Training data, 95% accuracy on Validation data and 95% accuracy on Testing data.
* The Accuracy, F1 Score, Recall and Precision scores are 95.1818611242324 %, 0.9515821860326811, 0.951818611242324 and 0.9525182117751877 respectively.
* This is the First and Basic version Of this Project.

### Research materials used

* M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676 [Paper link](https://ieeexplore.ieee.org/document/9144185)

* Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. [Paper link](https://www.sciencedirect.com/science/article/pii/S001048252100113X)

* Narayana Darapaneni, Suma Maram, Harpreet Singh, Syed Subhani, Mandeep Kour, Sathish Nagam, Anwesh Reddy Paduri. "Prediction of COVID-19 using chest X-ray images".[Paper link](https://arxiv.org/abs/2204.03849v1)

* This notebook also took inspiiration from the *Covid-19 Radiology | VGG19| f1-score: 95%* by AHMED HAFEZ.[Link to the notebook](https://www.kaggle.com/code/ahmedtronic/covid-19-radiology-vgg19-f1-score-95)

### GPU Specifications
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   47C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

### Dataset

**COVID-19 RADIOGRAPHY DATABASE (Winner of the COVID-19 Dataset Award by Kaggle Community)**

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images and corresponding lung masks. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

The link to the dataset is [here](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Accessing the notebooks

(This project assumes that, entirity of the notebooks are being run on Google Collaboratory enviornment)

* **Download the data**
    - The data can be downloaded directly from the github reporistory by creating a clone of the reporistory on the local machine or the above mentioned Kaggle Reporistory link.
    -Upload the data in your Google Drive.

* **Training the model**
    - Please run the notebook [Disease_Detection_Technology.ipynb](https://github.com/KT2001/Disease_Detection_Technology/blob/master/DDT_Transfer_learning_Tensorflow.ipynb) either directly through Github link or in your local machine.
**(Note: It is adviced to replace any file locations in the code cells of this notebook with appropriate file location as per your system or Google drive.)**

* **Running the streamlit application** 
    - The pretrained model and model weights are already present in the reporistory, in the folder [ResNet50](https://github.com/KT2001/Disease_Detection_Technology/tree/master/ResNet50), or could be downloaded after training the model by following the above mentioned instructions.
    -Run the notebook [Application.ipynb](https://github.com/KT2001/Disease_Detection_Technology/blob/master/Application.ipynb), to run the streamlit application.
    -Alternatively, to run the application using your local machine, create an enviornment and  pip or conda install [requirements.txt](https://github.com/KT2001/Disease_Detection_Technology/blob/master/requirements.txt)
    -Run the python file [app.py](https://github.com/KT2001/Disease_Detection_Technology/blob/master/app.py) using the command **streamlit run app.py**.

### Final thoughts

This was a great oppertunity to learn and work with wonderful folks such as [Harsh Garg](https://www.linkedin.com/in/iofficialharshgarg/), [Nipun Bhardwaj](https://github.com/Nipun-Bhardwaj) and [Sanyam Garg](https://www.linkedin.com/in/sanyam-garg-2767a41b1/). 

The field of AI is rapidly evolving and has emerged as a highly promising and financially rewarding domain. By leveraging deep learning techniques, we can revolutionize the process of disease detection, making it faster and more efficient. Although this technology is currently in its developmental phase, it holds immense potential to transform the medical field. However, it will require considerable time and efforts before it becomes fully integrated and widely adopted. Nevertheless, this experiment unequivocally demonstrates that an AI-driven future is not a distant possibility. 
