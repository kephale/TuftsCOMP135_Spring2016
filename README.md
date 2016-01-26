# Introduction to Machine Learning and Data Mining  

Comp 135: Introduction to Machine Learning and Data Mining  
Department of Computer Science  
Tufts University  
Spring 2016

Course Web Page (redirects to current page): [http://kephale.github.io/TuftsCOMP135_Spring2016/](http://kephale.github.io/TuftsCOMP135_Spring2016)

# Announcement(s):

None

# What is this course about?  

Machine learning is the study of algorithmic methods for learning and prediction based upon data. Approaches range from extracting patterns from large collections of data, such as social media and scientific datasets, to online learning in real-time, for applications like active robots. ML is becoming increasingly widespread because of the increase and accessibility of computational power and datasets, as well as recent advances in ML algorithms. It is now commonplace for ML to produce results that have not been achieved by humans.

As this is an introductory course, we will focus on breadth over the field of ML, but this will still require significant cognitive effort on your part. The ideal candidate for this course is an upper-level undergraduate or beginning graduate student comfortable with some mathematical techniques and a solid grounding in programming. Maths that will prove useful are: statistics, probability, calculus, and linear algebra. We will review some of the essential topics, and the only explicit requirements are previous coursework of (COMP 15) and (COMP/MATH 22 or 61) or (consent of instructor).
Comp 160 is highly recommended.  

## Class Times:  

Tu, Th 10:30AM - 11:45AM  
Tisch Library, 304-Auditorium  

## Instructor:  

Kyle Harrington [kyle@eecs.tufts.edu](mailto:kyle@eecs.tufts.edu)  
Office Hours: By appointment  

## Teaching Assistants:  

Sepideh Sadeghi  
Hao Cui  

# Grading  

- Written homework assignments (20%)  
- Quizzes (20%)  
- In-class midterm exam (20%): March 17
- Final project (40%)

### Rules for late submissions:  

All work must be turned in on the date specified. Unless there is a last minute emergency, please notify [Kyle Harrington](mailto:kyle@eecs.tufts.edu) of special circumstances at least two days in advance.

If you aren't done by the due date, then turn in what you have finished for partial credit.

# Collaboration  

On homework assignments and projects: Discussion about problems and concepts is great. Each assignment must be completed by you and only you, and is expected to be unique. Code should be written by you; writeups should be written by you. If you have collaborated (helping or being helped), just say so. There is no harm in saying so.  

On quizzes and exams: no collaboration is allowed.  

Failure to follow these guidelines may result in disciplinary action for all parties involved. For this and other issues concerning academic integrity please consult the booklet available from the office of the Dean of Student Affairs.  

# Tentative List of Topics  

- Supervised Learning basics: nearest neighbors, decision trees, linear classifiers, and simple Bayesian classifiers; feature processing and selection; avoiding over-fitting; experimental evaluation.
- Unsupervised learning: clustering algorithms; generative probabilistic models; the EM algorithm; association rules.
- Theory: basic PAC analysis for classification.
- More supervised learning: neural networks; backpropagation; dual perceptron; kernel methods; support vector machines.
- Additional topics selected from: active learning; aggregation methods (boosting and bagging); time series models (HMM); reinforcement learning

# Reference Material

We will use a mixture of primary research materials, portions of texts, and online sources. Required reading material will be listed as such. The following is a list of recommended reference material.

- Machine Learning. Tom M. Mitchell, McGraw-Hill, 1997  
- Introduction to Machine Learning, Ethem Alpaydin, 2010.  
- An introduction to support vector machines : and other kernel-based learning methods. N. Cristianini and J. Shawe-Taylor, 2000.  
- Data Mining: Practical Machine Learning Tools and Techniques. Ian H. Witten, Eibe Frank, 2005.
- Machine Learning: The Art and Science of Algorithms that Make Sense of Data. Peter Flach, 2012.  
- Pattern Classification. R. Duda, P. Hart, and D. Stork, 2001.  
- Artificial Intelligence: A Modern Approach. Stuart Russell and Peter Norvig, 2010  
- Principles of Data Mining. D. Hand, H. Mannila, and P. Smyth, 2001.
- Reinforcement Learning: an Introduction. R. Sutton and A. Barto, 1998.x

Roni Khardon's Version of [COMP-135](http://www.cs.tufts.edu/~roni/Teaching/ML/).    

# Programming and Software

[Weka](http://www.cs.waikato.ac.nz/ml/weka/) is a great machine learning package that has been around for a while. It is quite extensible, and we will be using it for some assignments. You can use weka.jar on the CS department servers through the command line. If you have trouble, there is excellent documentation on the [Weka wiki](https://weka.wikispaces.com/).

There are some languages that are particularly useful in the context of machine learning, either because of their innate capabilities or because of libraries implemented in the language. When code examples are provided in class they will likely be in one of these language:

- Python
- Java
- Julia
- Matlab
- Clojure
- R

[Jupyter](http://jupyter.org/) is a notebook-based programming environment that supports many programming languages. We will use it for numerous in-class demos, and you may want to use it for your homeworks as well.  

# Schedule

Date | Lecture | Assignments and Notes | Due Date
-----|---------|-----------------------|----------
01/21| Introduction to Machine Learning | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture01)</li><li>[Assignment 1](#assignment1)</li></ul> | 01/27
01/26| Instance learning | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture02)</li><li>[Notebook](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Lecture02/notebooks/instance_based_learning.ipynb)</li><li>[Assignment 2](#assignment2)</li><li>[kNN:Scholarpedia](http://www.scholarpedia.org/article/K-nearest_neighbor)</li><li>[Andrew Moore's KD tree tutorial](https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1991_1/moore_andrew_1991_1.pdf)</li></ul> | 02/03  
01/28| Decision trees pt 1 |
02/02| Decision trees pt 2 |
02/04| Naive bayes |
02/09| Measuring ML success pt 1 |
02/11| Measuring ML success pt 2 |
02/16| Features |
02/18| No class, Monday Schedule |
02/23| Features |
02/25| Linear threshold units pt 1 |
03/01| Linear threshold units pt 2 |
03/03| Clustering pt 1 |
03/08| Clustering pt 2 |
03/10| Unsupervised learning |
03/15| Association rules |
03/17| Midterm |
03/22| No class, Spring recess|
03/24| No class, Spring recess|
03/29| Computational learning theory |
03/31| Kernel-based methods |
04/05| Perceptron |
04/07| SVM |
04/12| Active learning |
04/14| MDPs and Reinforcement Learning |
04/19| Reinforcement learning pt 2 |
04/21| Aggregation methods |
04/26| Project presentations |
04/28| Project presentations |

# Assignments

## Assignment1

- Download [Weka](http://www.cs.waikato.ac.nz/ml/weka/)
- Download [Dataset](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/master/Assignment1/IHME_GBD_2013_LIFE_EXPECTANCY_1970_2013_Y2014M12D17.csv)  
- Open Weka, Choose Explorer models
- Load the dataset with "Open file..."
- Investigate the data through the "Classify" "Cluster" "Associate" "Visualize" tabs  
- If Weka is running slowly, then you can try the [abbreviated dataset](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/master/Assignment1/IHME_GBD_2013_LIFE_EXPECTANCY_1970_2013_Y2014M12D17_short.csv)
- Data source information is available in: [https://github.com/kephale/TuftsCOMP135_Spring2016/tree/master/Assignment1/DATA.md](https://github.com/kephale/TuftsCOMP135_Spring2016/tree/master/Assignment1/DATA.md)  

Note that it is possible to call Weka from the command line (i.e. on the homework server)

### Submission of assignment 1

Write a one paragraph description of what you can find.

- Open "Visualize" and investigate how pairs of attributes relate to each other?
- What types of clusters can you find (try "Cluster"/"Choose"/"SimpleKMeans" test with different "numClusters")
- If you're feeling adventurous, then try to build a classifier ("Classify"/"Choose"/"weka.classifiers.trees.J48" and choose a nominal attribute to classify over, like "location_name". In the case of "location_name", before building the classifier use "Preprocess" and remove all "location" attributes except "location_name". You will want to use the abbreviated dataset for this.)

## Assignment2

Git is the current standard for code sharing and collaborative coding. This course is run off of Github using git to control and track the history of changes. For this assignment, clone [this repository](https://github.com/kephale/TuftsCOMP135_Spring2016/), open up Lecture02/notebooks/instance_based_learning.ipynb, complete the assignment by adding new cells to the notebook, and submit a pull request on GitHub. The new cells should implement an exhaustive search implementation of kNN. The current version uses a KD-tree to obtain the nearest-neighbors. The current line of code that you should replace with your exhaustive search implementation is:
query_result = kdtree.query( [0.5, 0.5], k=10 )  

- For help getting going with git and GitHub checkout [GitHub guides](https://guides.github.com/activities/hello-world/)
- Setup [Jupyter](http://jupyter.org/) on your computer (use Python for this assignment. This is the default language Jupyter installs)
- See [Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture02) from Lecture 2 for information on the k-Nearest Neighbors algorithm
- We already have an existing Jupyter [Notebook](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Lecture02/notebooks/instance_based_learning.ipynb), but it is missing a classic implementation of kNN with exhaustive search!

Submission:  
- Use 'provide' to submit a text file that links to your pull request on Github  