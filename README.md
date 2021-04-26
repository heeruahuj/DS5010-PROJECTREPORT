#DS 5010 Final Project ReadMe.md

<h3> Purpose </h3>

*The purpose of the package is to allow users to perform classification using two types of classifiers:
                <br>
                <br>
                -Naive Bayes Classifier <br>
                -Logistic Regression Classifier
                <br>
<h4> Naive Bayes Classifier</h4>
<p>
*The Naive Bayes Classifier is based on the Bayes' Theorem.
  <br>
  P(c|x) =[P(x|c).P(c)]/P(x)
  <br>
  Here, 
  P(y|X) =  P(X|y).P(y) / P(X)
  <br>
  *Naive Bayes makes the assumption that all the features are independant. So, Bayes Theorem can be re-written as:
  <br>
  P(y|X) = [P(x<sub>1</sub>|y).P(x<sub>2</sub>|y)...P(x<sub>n</sub>|y). P(y)] / P(X)<br>
  *We can eliminate the denominator. For classification, we choose the class with the highest conditional probability. 
  <br>
  Then,
  <br>
  yhat = argmax<sub>y</sub> P(x<sub>1</sub>|y).P(x<sub>2</sub>|y)...P(x<sub>n</sub>|y). P(y)<br>
  yhat = argmax<sub>y</sub> log(P(x<sub>1</sub>|y)) + log(P(x<sub>2</sub>|y)) + .... + log(P(x<sub>n</sub>|y)) + log(P(y)) 
 <br>* This is the equation implemented in the predict() in NaiveBayes.py. 
  <br>
  *Posterior_probabilities in predict is used to calculate the sum of log(P(x<sub>i</sub>|y)) <br>
            - using gaussian distribution implemented in prob_den_func()
  <br> p.s. Took the log to reduce overflow since the probability values are very small
</p>

<h4> Logistic Regression Classifier</h4>
<br>
<p> Here, we initialise the weights to 0. We run gradient descent for max number of iterations. 
<br>
 In gradient descent, we calculate the sigmoid probabilities using the sigmoid function - predicted probabilities
  <br>
  S(z) = 1/(1+exp(-z))
  <br>
  Then, we calculate the gradients using the following gradient descent formula. 
  <br>
  -y*log(wx)-(1-y)*log(1-wx). 
  <br>
  Now, we update the weights by subtracting the gradient*learning rate. 
  <br>
  For testing, we calculate the sigmoid probabilities again for X_test and compare it to a threshold. We classify based on the threshold. (>0.5 --class 1; <0.5 --class 0)
</p>

<p>The sigmoid probability is given by the formula S(z) = 1/(1+exp(-z)). It always has an output between 0 and 1, which is great for predicting classes.
The sigmoid value can be used to create a threshold. Any value of probability over 50% would make it likely that the class is 1 or present. If lower, it is categorized as 0 or not present. We can use the features(x) and the trained weights(w) as an input for the sigmoid method. This generates the predicted classes for our classifier.
The cost tells us how far from the actual label the prediction is. It is calculated with the formula -y*log(wx)-(1-y)*log(1-wx). The method for gradient descent calculates the gradient for the features with the formula x(sigmoid(wx) – y). The weights are then updated by subtracting the gradient*learning-rate. This takes the regression towards the minimum.
</p>


<h3> Organisation of the Repository </h3>

1. DS5010 Final Project Report.docx - This is the Project Report
2. README.md - This is the ReadMe File as per the requirements
3. __init__.py - This is the initial commit
4. NaiveBayesClassifier Folder - It contains all files related to Naive Bayes Classifier
-NaiveBayes.py - Naive Bayes implementation 
-ExampleNB - one tested example on the NaiveBayes implementation
-TestNB - Unit Testing of Naive Bayes Classifier
5. LogisticRegressionClassifier Folder - It contains all the files related to Logistic Regression Classifier
- LogisticRegression.py - Logistic Regression Implementation 
- ExampleLR - one tested example on the Logistic Regression implementation 
- TestLR - Unit Testing of Logistic Regression Classifier


<h3> Simple Examples of Usage </h3>

<h4> How to use </h4>

<p> The package and modules for the Logistic Regression Classifier and the Naïve Bayes Classifier can be imported into a python module. The classifiers take input of numpy arrays. This is because numpy provides efficient vectorized calculations and we have used it extensively for matrix operations. 

 <br>

After importing the module, one can create an object of the classifier and set the parameters as needed.
<br>
For example, an object of the LogisticRegressionClassifier can be created as such:
<br>
</p>
<br>

**algorithm = LogisticRegressionClassifier(learning_rate = 0.0000001, iterations = 1000)**

<br>
<p>
  We can then call the training method on training labels and predict. The fit method for Logistic Regression also has the option to set verbose to 1, which works similarly to other data science libraries. It will print the loss for every iteration it trains. For example,</p>

<br>

 **algorithm.fit(x_train, y_train)**
<br>

**predicted_y = algorithm.predict(x_test)**

<p>
  A similar workflow can be utilized for the Naïve Bayes Classifier. For Logistic Regression, one can also choose the learning rate and iterations for gradient descent.</p>
            
 <p>
  The files ‘ExampleLR.py’ and ‘exampleNB.py’ include complete examples of using both classifiers on datasets and comparing the accuracy for training iterations.
  </p>
  
  
                
         
