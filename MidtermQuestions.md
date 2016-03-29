# Midterm Questions

## 1

### Q

The instructions say to classify the unknowns using kNN with k = 25.
If there’s a tie - which I interpret to mean if the “majority” is the same for 2+ digits, then return them all.

Part A asks for the index of the “closest matching instance” for each unknown character. By closest matching instance, do you mean the instance of the character closest in distance to our unknown, or closest in terms of classification?

### A

Correct about the ties.

Closest matching instance is the instance of the character closest in distance to the unknown character.

### Q

For part B of the problem, what did you mean by getting the accuracy of each digit? The case is I take a row, and use the decision tree to see what will it predict then see if that prediction matches the original label, right?

### A

For each digit in the training set, what is the accuracy of your predictor for that digit, combining true positive, false positive, etc.

You should have 10 accuracies in the file, one for each digit

## 2

### Q

For question 2a, I presume that the order of output will be in BFS order, or more specifically, is similar to array heap indexing where the output row (whose line number within the file is) i has a parent at row 2 / i, a left child at 2i, and a right child at 2i + 1. Also, the left child is the decision path taken when for the current node's attribute, the sample has that attribute valued at 0, and similarly for the right child, but for value 1.

tree:

 a

/    \

b      c

/ \    / \

d  e  f   g


becomes

a.attribute_val 0

b.attribute_val 1

c.attribute_val 1

d.attribute_val 2

etc.

### A

BFS order is definitely the way to go.

### Q

If we have different class instances at a leaf (depth = 5) in our decision tree we should use the mode as the classification?

### A

Yes

## 3

### Q

For question 3a, are we using a ID3 tree with max depth of 5 for feature selection? Or are we using a max depth of 25 (all features can be used up in a branch)?

### A

Continue to use max depth = 5 when doing feature selection.

### Q

For question 3a, if a non-maximum depth node, or an internal node, doesn't split the current subset of the training data on attribute value despite being the best attribute to split on, do we end construction for that branch and set it to the mode output value (or digit) for the attribute value which captures the entire training data subset?

### A

Yes, the default behavior you'll run into for ID3-types of decision trees is to return the mode when there is still a mix of discrete labels. Use this behavior when classifying.

### Q

For question 3a, are we printing the accuracy of each classifier (all 256 of them) or the best classifier (from the 256 surveyed) for the current iteration of the forward selection wrapper method?

### A

No, just print the accuracy of the *selected* features (i.e. the best combination found for each subset size).

### Q

I'm not clear on how we use decision tree to select features. Do we look at the entropy gain from each collection of features?

###

Check the slides. Feature selection wraps around the decision tree algorithm. You use the performance of decision trees on specific sets of features in order to do the selection.

## 4

### Q

In Problem 4, part a, you ask for the accuracy of each classifier. In Problem 2, when you asked for this (on the training set), you wanted the accuracy of each digit (0 thru 9). Would you like this same breakdown for each of the folds in Problem 4? That is, would you like the kNN accuracy overall and the ID3 accuracy for each digit, for each fold? Or would you just like an overall accuracy for kNN and ID3 for each fold (2 numbers)?

### A

Problem 4 is asking for the overall accuracy.

### Q

For question 4a, are we randomly separating the data into 10 folds, or are we ascertaining that there is a relatively even distribution of digits in each fold?

### A

Use the default assumption of cross validation: randomly shuffle, and partition into 10 folds. The dataset is balanced enough, so there is no need to do any stratification.

## 5

### Q

Do we use discrete '1.0' and '0.0's to create and adjust our centroids, that is are their features one or off or do they hover between 0 and 1?

### A


Features are 0 or 1, but centroids can take floating point values between 0 and 1. 
