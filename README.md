# [3rd Data Science's challenge of Talent Squad ](https://nuwe.io/dev/challenges/talent-squad-data-science-iii)
(10th Oct 2022 - 11th Dec 11 2022)]

# Background

#### Influences on academic performance

A study has been carried out to see if the academic performance of children is influenced by the academic level of their parents. Therefore, the academic results of the students will be evaluated based on several variables

**data**
You will be given a dataset of 800 rows for the prediction algorithm training and 200 for the test.

**variables**
'gender' : Sex of the student

'parental level of education': Educational level of the father

'lunch': If you have lunch scholarships

'Exam Preparation Course': Attend academy

'math score': Score in Maths

'reading score': Reading Comprehension

'writing score': Writing score

# Problem

The **purpose of this project** is to ....

Evaluation will be based on the f1_score from predictions of the dataset and the groundtruth (900), code quality (200) and documentation (100). (maximum puntuation 1200).

# Results

# Analysis

# Solution

# License

The MIT License (MIT)

Copyright (c) 2022-2026 Marc Humet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



## Methodology



## Tools

* Python
* Git & Github
* [Jupyter Notebook](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/main_Notebook.ipynb)
* Visual Studio Code
* Libraries: Pandas, Numpy, Seaborn, Matplotlib, Sklearn, Keras, Tensorflow, Selenium, OpenCV, Pillow, bing_image_downloader, 

## Getting Started

1. Clone this [repo](https://github.com/MarkusHumetus/Image_sports_classification) (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Use [requirements.txt](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/requirements.txt) to check all installed dependencies in the working environment. 
3. Check and adapt the paths to the working directories according where files are stored.
4. Run the Jupyter Notebook as administrator. Otherwise, permision conflict will arise when opening image files.

## Project Status

[Finished]

Project was completed and submitted for competition in 23rd of September 2022.

## Results & Conclusions

* The maximum __f1 macro score__ obtained in the classification of the test images was __0.905__. 

* Best results were obtained with __Xception__ and __InceptionResNetV2__ pretrained models with the augmented training set of images.

* No improvements were observed when leaving last 10% layers of the pretained Xception model to be trained again.

* Although __number of images was doubled by web scrapping__, the increase the number in training images didn't improve the performance of the model. 

* As forseen, the __missclassifications occurs between__ the two most similar sports: __baseball and cricket__. 


The [best model as json file](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/Im%C3%A1genes-data-science-ii/output/Xception_pretrained_aug.json) and its [wheigts as h5 file](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/Im%C3%A1genes-data-science-ii/output/Xception_pretrained_aug.h5)  are uploaded on this repository. In the [notebook](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/main_Notebook.ipynb), there is plenty of examples, how it can be loaded and tested with the test data.
Predictions of test images were stored in the file [predictions.csv](https://github.com/MarkusHumetus/Image_sports_classification/blob/main/predictions.csv)

# Contact

If you have any comment, doubt, proposal,... don't hesitate to contact me by email to marc.humet.datascience@gmail.com

If you are interested in my professional profile, please feel free to look at my [project's portfolio](https://github.com/MarkusHumetus) and my linkedin page:

[![LinkedIn][linkedin-shield]][linkedin-url]


[linkedin-url]: https://www.linkedin.com/in/marchumetmontada/

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

Requisitos de entrega
Revisar que la rama principal de entrega es main.

El código se deberá de entregar en un fichero py llamado app.py. En este tendréis todas las funciones utilizadas y el código necesario junto documentación para reproducir vuestra solución. Para los gráficos os recomendamos entregarlo en un ipynb.

Para que cuente documentación tiene que tener un README.md con la siguiente estructura y lo siguientes títulos:

background
problem
results
analysis
solution
license
Para PUNTUAR las PREDICCIONES el repositorio entregado tiene que tener un archivo:

predictions.json con este formato


# Acknowledgments

Thanks to the following organisations for the set up of such event which give the junior data scientist oportunities to learn, do networking and hopefully reach a first job in the data field.

![Barccelona_Digital_Talent](https://barcelonadigitaltalent.com/app/uploads/sites/3/2020/02/BDT-1.1-POSITIU_2-01.jpg)

![Mobil_World_Congress](https://challenges-asset-files.s3.us-east-2.amazonaws.com/companies/MWC_card.png)

![Nuwe](https://elreferente.es/wp-content/uploads/2021/12/LOGO_LETTERS_MONO-3.png)