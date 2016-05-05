# Introduction to Machine Learning and Data Mining  

Comp 135: Introduction to Machine Learning and Data Mining  
Department of Computer Science  
Tufts University  
Spring 2016

Course Web Page (redirects to current page): [https://github.com/kephale/TuftsCOMP135_Spring2016](https://github.com/kephale/TuftsCOMP135_Spring2016)  

# Announcement(s):

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

Sepideh Sadeghi, <A HREF="mailto:sepideh.sadeghi@tufts.edu"> sepideh.sadeghi@tufts.edu</A><br /> Office Hours: Mon noon-1pm, Fri 10am-noon,<br /> Location for Office Hours: Halligan 121<br />  
Hao Cui, <A HREF="mailto:Hao.Cui@tufts.edu"> Hao.Cui@tufts.edu</A><br /> Office Hours: Tue 4:30-5:30 pm, Thu 4:30-5:30 pm,<br /> Location for Office Hours: Halligan 121<br />  

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

- <b>(We will often use this one) Machine Learning. Tom M. Mitchell, McGraw-Hill, 1997</b>  
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

[Jupyter](http://jupyter.org/) is a notebook-based programming environment that supports many programming languages. We will use it for numerous in-class demos, and you may want to use it for your homework and final projects as well.

- [Jupyter quick start guide](http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/)
- [Official Python 2 tutorial](https://docs.python.org/2/tutorial/) and [Python tutorial](http://www.tutorialspoint.com/python/)

## Slides

Slides are made with Reveal.JS. This has some perks that do not exist in Powerpoint/Keynote. They embed into the web more elegantly than PDFs, and because they use HTML5/CSS support essentially all functionality that one can get in a web browser.

When browsing the slides, notice that there is also an "overview" mode (press 'o' after loading a particular set of slides). This will tile the slides in an arrangement that is encoded within the presentation file, and should facilitate rapid browsing.  

# Schedule

Date | Lecture | Assignments and Notes | Due Date
-----|---------|-----------------------|----------
01/21| Introduction to Machine Learning | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture01)</li><li>[Assignment 1](#assignment1)</li></ul> | 01/27
01/26| Instance learning | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture02)</li><li>[Notebook](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Lecture02/notebooks/instance_based_learning.ipynb)</li><li>[Bonus Assignment 2](#assignment2)</li><li>[kNN:Scholarpedia](http://www.scholarpedia.org/article/K-nearest_neighbor)</li><li>[Andrew Moore's KD tree tutorial](https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1991_1/moore_andrew_1991_1.pdf)</li></ul> | 02/03
01/28| Decision trees pt 1 | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture03)</li><li>[C4.5 for continuous values](http://www.jair.org/media/279/live-279-1538-jair.pdf)</li><li>[Scikit learn:Decision trees](http://scikit-learn.org/stable/modules/tree.html)</li><li>Chapter 3 of Mitchell</li></ul> |
02/02| Decision trees pt 2 | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture04)</li></ul>
02/04| Naive bayes | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture05)</li><li>[Chapter 6 from Mitchell](http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)</li><li>[Stanford:Intro to Probability slides](http://www.stanford.edu/class/cs109/slides/IntroProbability.pdf)</li><li>[Stanford:Conditional Probability + Bayes Theorem](http://web.stanford.edu/class/cs109/slides/ConditionalProbability.pdf)</li></ul> |
02/09| Measuring ML success pt 1 | <ul><li>Chapter 5 - Mitchell</li><li>Final project proposal <i>(See due date)</i></li><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture06)</li></ul> | 03/07
02/11| Measuring ML success pt 2 | <ul><li>[Assignment 3](http://kephale.github.io/TuftsCOMP135_Spring2016/Assignment3.pdf)<i>(See due date)</i></li><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture07)</li></ul> | 02/16
02/16| Features | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture08)</li><li>[An Introduction to Variable and Feature Selection](http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf). Isabelle Guyon and Andre Elisseeff, Journal of Machine Learning Research 3 (2003) 1157-1182 .</li><li>[Supervised and unsupervised discretization of continuous features](http://robotics.stanford.edu/~ronnyk/disc.ps). James Dougherty, Ron Kohavi, and Mehran Sahami. International Conference on Machine Learning, 1995.</li><li>Chapter 6 of Introduction to Machine Learning, Second Edition, by Ethem Alpaydin</li><li>Chapter 10 of Machine Learning: The Art and Science of Algorithms that Make Sense of Data, Peter Flach</li></ul> |
02/18| No class, Monday Schedule |
02/23| Features | <ul><li>Quiz 1</li><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture09)</li><ul>
02/25| Linear threshold units pt 1 | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture10)</li><ul>
03/01| Linear threshold units pt 2 | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture11)</li><li>[Generalized delta rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)</li><li>[L1 and L2 regularization](http://cs.nyu.edu/~rostami/presentations/L1_vs_L2.pdf)</li><li>[Stochastic Gradient Descent](http://alex.smola.org/teaching/cmu2013-10-701/slides/3_Recitation_StochasticGradientDescent.pdf)<ul>
03/08| Clustering | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture12)</li><li>[K-means](http://www.labri.fr/perso/bpinaud/userfiles/downloads/hartigan_1979_kmeans.pdf)</li><li><a href="https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Assignment4.pdf">Assignment 4</a> <i>(See due date)</i></li></ul> | 03/15
03/10| Reinforcement Learning and Games | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture13/#/)</li><li>Chapter 13 - Mitchell</li><li>[Learning to play 49 Atari games with 1 algorithm](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)</li><li><a href="http://gizmodo.com/google-ai-will-compete-against-go-world-champion-lee-se-1757289813">AlphaGo v Wold Champion Lee Sedol in Go</a></ul>
03/15| Unsupervised learning | <ul><li>[Expectation Maximization algorithm](http://www.eecs.yorku.ca/course_archive/2007-08/W/6328/Reading/EM_tutorial.pdf)</li></ul> |
03/17| Midterm | <ul><li><a href="https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/MidtermExam.zip">Exam</a></li><li><a href="http://www.r2d3.us/visual-intro-to-machine-learning-part-1/">Amazing visualization of decision trees</a></li></ul> | Due before class on 03/31
03/22| No class, Spring recess|
03/24| No class, Spring recess|
03/29| Distribution Approximation and EM | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture16/#/)</li><li>[Expectation Maximization algorithm](http://www.eecs.yorku.ca/course_archive/2007-08/W/6328/Reading/EM_tutorial.pdf)</li><li>Final project <i>(See due date)</i></li></ul> | 05/05
03/31| Boosting (Alex Lenail) |
04/05| Neural Networks | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture17/#/)</li><li>Chapter 4 - Mitchell</li><li>[Touretzky's Backpropagation Slides](https://www.cs.cmu.edu/afs/cs/academic/class/15883-f15/slides/backprop.pdf)</li><li>[Geoff Hinton's Backpropagation Slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec3.pdf)</li><li>[Send 5-slide summary of project](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture17/#/7)</li></ul> | 04/12
04/07| Support Vector Machines | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture18/#/)</li><li>[A practical guide to support vector classification](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)</li></ul>
04/12| Reinforcement Learning | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture19/#/)</li><li>[Reinforcement Learning: An Introduction by Sutton and Barto](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)</li><li><a href="https://github.com/Rochester-NRT/AlphaGo">Open-source clone of AlphaGo in progress</a></li></ul> |
04/14| Game Theory and Retrospective | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture20/#/)</li><li>[Assignment 5](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Assignment5.md) <i>(See due date)</i></li></ul> | 04/26
04/19| Project presentations | <ul><li>Quiz 2</li></ul> |
04/21| Project presentations & Random Forests | <ul><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture21/#/)</li><li>[Random Decision Forests](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)</li><li>Slides</li></ul>
04/26| Project presentations & Long Short-term Memory | <ul><li>[LSTM Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)</li><li>[Overview of LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</li></ul>
04/28| Project presentations & Automatic Optimization of ML Pipelines | <ul><li>[TPOT paper](http://arxiv.org/pdf/1601.07925v1.pdf)</li><li>[TPOT (Github)](https://github.com/rhiever/tpot)</li><li>[Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture22/#/)</li></ul>
05/05| Program Discovery | <ul><li>A Field Guide to Genetic Programming</li></ul>


# Assignments, Quizzes, and Exams

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

### This is a bonus for 10% on a quiz, not required.

Git is the current standard for code sharing and collaborative coding. This course is run off of Github using git to control and track the history of changes. For this assignment, clone [this repository](https://github.com/kephale/TuftsCOMP135_Spring2016/), open up Lecture02/notebooks/instance_based_learning.ipynb, complete the assignment by adding new cells to the notebook, and submit a pull request on GitHub. The new cells should implement an exhaustive search implementation of kNN. The current version uses a KD-tree to obtain the nearest-neighbors. The current line of code that you should replace with your exhaustive search implementation is:
query_result = kdtree.query( [0.5, 0.5], k=10 )  

- For help getting going with git and GitHub checkout [GitHub guides](https://guides.github.com/activities/hello-world/)
- Setup [Jupyter](http://jupyter.org/) on your computer (use Python for this assignment. This is the default language Jupyter installs)
- See [Slides](http://kephale.github.io/TuftsCOMP135_Spring2016/Lecture02) from Lecture 2 for information on the k-Nearest Neighbors algorithm
- We already have an existing Jupyter [Notebook](https://github.com/kephale/TuftsCOMP135_Spring2016/blob/gh-pages/Lecture02/notebooks/instance_based_learning.ipynb), but it is missing a classic implementation of kNN with exhaustive search!
- Some Python and Jupyter tutorials are linked in the programming and software section

Submission:  

- Submit a pull request to the course github repository (https://github.com/kephale/TuftsCOMP135_Spring2016/)  

### Additional instructions on submitting a pull request:

1. In order to make a pull request, you will need to "fork" the class repository (https://github.com/kephale/TuftsCOMP135_Spring2016/). On the github page, at the top right, you will see a "Fork" button. If you click this, then follow the instructions, it will create a copy of the repository under your username.  
2. You will need to clone your fork (this will download your version of the class repository).  
3. Make your changes to the file (this would involve opening Jupyter, editing the file, and resaving it). If you have already changed the file without using git, all you have to do is copy your updated version over the existing file the fork that you just downloaded.  
4. Add your changed files, commit the changes, and push to the repository.  
5. Once you have done this, you can open up the webpage for your fork and click on the "New pull request" button. Follow the instructions to send a pull request to the course's repository.  

If you have any issues with Github, then see the Github guides (https://guides.github.com/activities/hello-world/)

Nearly every major corporation (Google, Facebook, Microsoft, Twitter, etc.) and university uses git to manage code for almost all of their open-source projects, if not specifically Github. This is especially true for the open-source machine learning code being released by these corporations and universities. When it comes time to work on final projects, especially with multiple people involved, git will turn out to be one of your most powerful tools.  

## FinalProjects

The final project for this course is an opportunity to apply what you've learned about machine learning to a real-world problem, to extend an existing machine learning algorithm, or to implement and explore a cutting-edge ML technique.

Collaboration is an option, but this should be discussed with Kyle a priori. A clear delineation of work should be proposed ahead of time. Github is strongly encouraged for collaborative projects to make it easier to measure each person's contribution.

If you would like a suggestion for a project contact Kyle 2 weeks before the proposal deadline to schedule a meeting. We will talk through problems/data that may be of interest to you.  

### Proposals

Due: March 7

Submission: Email a *PDF* of your proposal to Kyle  

Write a 300-500 word abstract describing your proposed project. This should include 2-3 references of papers you expect to include in your final paper.

See an example project proposal [here](https://github.com/kephale/complexfeaturesbcell/blob/master/Proposal.md)  

### Project

Due: May 5

Turn in a 8-12 page paper. A rough outline is:
- Background on problem
- Related work
- Your method
- Results
- Conclusion and future work
- References

There should be at least 10 references.

### Resources

- Ask faculty around the department if they have datasets that might be interesting for Machine Learning
- [Google Scholar](http://scholar.google.com) - Search for articles published in "ICML", "NIPS", or "Machine Learning"; or search for keywords relevant to problems/algorithms that interest you  
- [Huge list of datasets](https://github.com/caesar0301/awesome-public-datasets?utm_content=buffer43079&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)  

## Quiz1

Quiz 1 will cover:

- kNN
- Decision trees
- Naive bayes
- Measuring success of ML algorithms

### License
