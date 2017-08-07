# Learning Resources
### Text Books :
- Data Mining: Concepts and Techniques, 3E by J. Han et al. [Download](http://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf)
- Data Mining: Practical Machine Learning Tools and Techniques, 3E by Ian H. Witten et al. [View](https://www.amazon.com/Data-Mining-Practical-Techniques-Management/dp/0123748569)
- Pattern Recognition and Machine Learning, by Christopher Bishop [View](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)

### Learn WEKA :

- GUI based learning :
  - by Prof. Ian H. Witten [Go YouTube](https://www.youtube.com/user/WekaMOOC/playlists)
  
- Implementation based learning :
  - Implemented in Java by Dr. Noureddin Sadawi [Go YouTube](https://www.youtube.com/playlist?list=PLea0WJq13cnBVfsPVNyRAus2NK-KhCuzJ), [Go Github](https://github.com/nsadawi/WEKA-API/tree/master/src)
  - Implemented in Java by Rafsanjani Muhammod
    - version #01: [Go Github](https://github.com/mrzResearchArena/Machine-Learning-Algorithms-with-WEKA/blob/master/MainClassV1.java) & [Romeo Version :):):) -> Section: SA -> June 20, 2017 -> [Go Github](https://github.com/mrzResearchArena/Machine-Learning-Algorithms-with-WEKA/blob/master/Romeo.java)]
    - version #02: [Go Github](https://github.com/mrzResearchArena/Machine-Learning-Algorithms-with-WEKA/blob/master/MainClassV2.java)
  - GUI & Core Java by Rafsanjani Muhammod [Go YouTube](https://www.youtube.com/playlist?list=PL3BNX6CPOw7q66qsYIcF18sswBN1sJcAm)
  
### Learn scikit-learn :
- Implemented in Python (scikit-learn) by Rafsanjani Muhammod [Go Github](https://github.com/mrzResearchArena/Machine-Learning-Algorithms-with-Python)
- LazyProgrammer [Go Github](https://github.com/lazyprogrammer)
- Hands-on Machine Learning with Scikit-Learn and TensorFlow [Go Github](https://github.com/ageron/handson-ml)



### Blogs :
- An Introduction to Data Mining by Dr. Saed Sayad [Go](http://www.saedsayad.com/data_mining_map.htm)
- Analytics Vidhya [Go](https://www.analyticsvidhya.com/blog/2016/10/16-new-must-watch-tutorials-courses-on-machine-learning/)
- Soft Computing and Intelligent Information Systems [Go](http://sci2s.ugr.es/imbalanced#Introduction%20to%20Classification%20with%20Imbalanced%20Datasets)
- Comparism between classifiers [Go](http://www.devopsinfographics.com/general/programming-a-beginneraes-guide-to-machine-learning-algorithms)

Coming soon ... :)

# Public datasets for Analytics
- UCI Machine Learning Repository [Go](http://archive.ics.uci.edu/ml/datasets.html)
- KEEL [Go](http://keel.es/)
- AnalyticsVidhya [Go](https://www.analyticsvidhya.com/blog/2016/11/25-websites-to-find-datasets-for-data-science-projects/)
- Kaggle (This is mainly a contest site.) [Go](https://www.kaggle.com/datasets)
- Public datasets for Machine Learning [Go](http://homepages.inf.ed.ac.uk/rbf/IAPR/researchers/MLPAGES/mldat.htm)
- Algorithmia [Go](http://blog.algorithmia.com/machine-learning-datasets-for-data-scientists/)
- Springboar [Go](https://www.springboard.com/blog/free-public-data-sets-data-science-project/)


# Syllabus

### Key Terms :
  1. Features / Attributres
  2. Feature-values & Attributre-values
  3. Class & Class-Attributes
  4. Instances / Records / Vectors / Tuples
  5. Two-class dataset & Multi-class dataset/Multi-label datasets (when number of class-values is gretter than 2.)
  6. High-dimensional (When number of feature is gretter than 10)
  7. Balanced dataset vs Imbalanced dataset
  8. Overfitting & Underfitting of a dataset
  9. Supervised learning vs. Unsupervised learning [Go](http://dataaspirant.com/2014/09/19/supervised-and-unsupervised-learning/)
  10. Classification, Regression, Clustering
  11. Bias–variance tradeoff [Go](http://www.learnopencv.com/bias-variance-tradeoff-in-machine-learning/)
  12. Noisy Datasets & how to remove noise ?
  13. Anomaly Detection [Go](http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf)
  
    
### Preprocessing Datasets :
- Remove duplicate elements
- Handle missing elements (Can you calculate : Mean, Median, Mode, Standard Deviation etc. ?)
- Feature Scaling (Can you calculate distance using : Euclid, Manhattan, Minkowski etc. ?)

### Classification :
- Rule Classifiers
  - ZeroR Classifier
  - OneR Classifier [Go](http://www.cs.ccsu.edu/~markov/ccsu_courses/DataMining-7.html) & [Go](http://www.saedsayad.com/oner.htm)
- Logistic Regression
- KNN Classifier
- Support Vector Classifier ( Kernels : Linear, Polynomial, Gaussian, Sigmoid, etc. )
- Naive Bayes Classifier
- Decision Tree Classifier
  - Gini
  - ID3
  - C4.5 / C5.0 / J48
- Ensemble Learning
  - Bagging Classifier
  - Boosting Classifier (AdaBoost, Gradient Boosting)
  - Random Forest Classifier
- Introduction to Deep Learning (ANN, RNN, CNN, SOM, Autoencoders )  

### Regression :
- Linear Regression (Simple & Multiple)
- Polynomial Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

### Clustering :
- KMeans
- Hierarchical (Agglomerative, Divisive)

### Imbalanced Learning : [Go](https://svds.com/learning-imbalanced-classes/)
- Majority class vs Minority class
- Re-sampling : Over-sampling, Under-sampling
  - Over-sampling algorithms : ADASYN, SMOTE, Random Over-sampling
  - Under-sampling algorithms : Random Under-sampling

### Feature Selection :
- Filter methods
- Wrapper Methods

### Dimensionality Reduction :
- PCA
- Kernel PCA
- LDA

### Performance Measures : [Go](https://classeval.wordpress.com/introduction/basic-evaluation-measures/)
- Understand Confusion Matrix
- Calculate : Accuracy, Error, Sensitivity, Specificity, Precision, Recall
- ROC Curve & AUPR Curve

# Course Schedule
  - Week #1 :
    - Introduction to Pattern Recognition,
    - Current Researh Trend, 
    - Introduction to WEKA
  - Week #2 :
    - Hands-on practice on WEKA GUI
      - Understand what are the .CSV & .ARFF file
      - Data Vizualization
      - Classifier design
      - Use different machine learning algorithms
    - Evaluation options
      - use training dataset
      - supplied test dataset
      - cross-validation (KFold=10)
      - split dataset (2/3 train & 1/3 test)
    - Confusion Matrix
      - TP, FP, FN, FP
      - Performance Measure : Accuracy, Error, TPR, FPR, F-Score etc.
      - weighted mean
    - Assignment #1 : Choose all (25) datasets from WEKA & submit report on it (Week #03).
  - Week #3 :
    - Introduction to WEKA implementation in Java.
      - Assignment #2 : Two huge datasets provides & submit report on it (Week #04)
  - Week #4 :
    - Details on WEKA implementation in Java.
      - Actual vs Prediction
      - Evaluation options
      - Feature Reduction
  - Week #5 : Ensemble Learning
  - Week #6 : Midterm Exam (Classifiers : based on your both Lab & Theory courses.)
  - Week #7 : Clustering
  - Week #8 : Data Analysis with scikit-learn (Python)-I
    - Loading the datasets (using pandas, numpy)
    - Features scalling
    - Machine Learning Classifiers
    - Evaluation Matrix
    - Assignment #3 : Datasets will provide.
  - Week #9 : Data Analysis with scikit-learn (Python)-II
    - Problem solving
    - Draw curve on scikit-learn (eg. ROC Curve, AUC Curve)
    - More tricks (imblearn)
  - Week #10 : Data Analysis with scikit-learn (Python)-III
    - Introduction to Kaggle competetion [Go](https://www.kaggle.com/)
    - Problem solving
    - Assignment #4 : A dataset will provide.
  - Week #11 : Presentation based on datasets. (Individual)
    - [Sample Slide: KDD'99 Datasets](https://www.slideshare.net/RafsanjaniMuhammod/analysis-of-the-kdd-cup1999-datasets) &
    - [Sample Slide: Analysis of the Datasets](https://www.slideshare.net/RafsanjaniMuhammod/analysis-of-the-datasets) 
  - Week #12 : Final Exam.
