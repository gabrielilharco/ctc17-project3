import numpy as np


#Create a list of dictionary from tuple
def create_list_of_dict(size):
    h = size[0]
    w = size[1]
    l = [[{} for x in range(w)] for y in range(h)] 
    return l
    


class NaiveBayesClassifier(object):
    """
    Naive Bayes classifier
    """
    def __init__(self):
        pass
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_samples = np.shape(X)[0]
        self.n_features = np.shape(X)[1]
        self.prior = {}
        self.classes = np.unique(y)
        self.like_prob = create_list_of_dict((self.n_classes + 1, self.n_features))
        
        self.calculate_prior_prob_(X, y)
        self.calculate_likelihood_prob_(X, y)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        #print (n_samples)
        y_pred = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            x = X[i]
            c_res = self.classes[0]
            p_res = 0.0
            for c in self.classes: 
                p_x = self.prior[c]
                for j in range(self.n_features):
                    p_x *= self.like_prob[c][j][x[j]]
                    
                if p_x > p_res:
                    p_res = p_x
                    c_res = c
            y_pred[i] = c_res     
        return y_pred

    def score(self, X_test, Y_test, labels):
        """
        Returns the confusion matrix for the classifier in a given test set
        """
        n_samples = np.shape(X_test)[0]

        # creating the confusion matrix
        confusion_matrix = {}
        for label in labels:
            label_dict = {}
            for other_label in labels:
                label_dict[other_label] = 0
            confusion_matrix[label] = label_dict

        prediction = self.predict(X_test)
        for i in range(n_samples): 
            confusion_matrix[prediction[i,0]][Y_test[i]] += 1

        return confusion_matrix
        
    #Helper functions
    def calculate_prior_prob_(self, X, y):
        total_numb = np.size(y)
        
        for c in np.unique(y):
            self.prior[c] = len(np.extract(y == c, y)) / total_numb
            
    def calculate_likelihood_prob_(self, X, y):
        for c in np.unique(y):
            n_c = len(np.extract(y == c, y))
            for j in range(self.n_features):
                #print (j)
                for f_j in np.unique(X[:, j]):
                    self.like_prob[c][j][f_j] = 0.0
                    
                    for i in range(self.n_samples):
                        if y[i] == c and X[i, j] == f_j:
                            self.like_prob[c][j][f_j] += 1
                    self.like_prob[c][j][f_j] /= n_c
                #print (c,j, self.like_prob[c][j])