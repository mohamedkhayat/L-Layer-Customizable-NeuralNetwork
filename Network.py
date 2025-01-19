from DeviceSelector import *
import time
from EarlyStopping import EarlyStopping
from utils import create_mini_batches
from Layers import Dropout,Dense
np = get_numpy()

#import numpy as np
class NeuralNetwork():
  
  def __init__(self, n_classes, layers, learning_rate,
               criterion, lamb=None):

    self.n_classes = n_classes
    self.layers = layers
    self.learning_rate = learning_rate
    self.criterion = criterion
    
  def forward(self,X,train=None):
    
    """
    dropout is a dict, with key being layer to implement dropout and value being the keep prob
    training being a boolean to know if applying dropout is needed or not since dropout is only applied
    during training and not inference
    this function is used to make a prediction yhat for an input x 
    """
    
    if train is None:
      train = self.training
      
    output = X
    for layer in self.layers:
      
      if self.training == False and isinstance(layer,Dropout):
        continue
      
      output = layer.forward(output,train)

    return output

  def backprop(self,dA):
    """
    backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
    the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
    layers we need to multiply by their mask
    """

    for layer in reversed(self.layers):
      dA = layer.backward(dA)

  def zero_grad(self):
    
    for layer in self.layers:
      if isinstance(layer,Dense):
        layer.zero_grad()
      
  def optimize(self):
    
    for layer in self.layers:
      if hasattr(layer,"params"):
        
        for param in layer.params:
          layer.params[param] -= self.learning_rate * layer.grads['d'+param]

  def fit(self, X_train, y_train, epochs = 30, 
          batch_size = 64, shuffle = True, validation_data = None, 
          early_stopping_patience = None, early_stopping_delta = 0):
    
    History = {}
    
    train_losses = []
    test_losses = []
    
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    if early_stopping_patience is not None and early_stopping_patience >= 1:
      
      er = EarlyStopping(early_stopping_patience,early_stopping_delta)

    for epoch in range(epochs):

      self.train()
      epoch_loss = 0
      epoch_train_accuracy = 0
      num_batches = 0
      
      mini_batches =  create_mini_batches(X_train,y_train, batch_size = batch_size,
                                          shuffle = shuffle, drop_last = True)
        
      for X_batch, y_batch in mini_batches:
        self.zero_grad()
        y_pred = self.forward(X_batch, train=True)      
        
        loss = self.criterion(y_batch, y_pred)
        epoch_loss += loss
        
        dA = self.criterion.backward(y_batch, y_pred)

        self.backprop(dA)
      
        self.optimize()
      
        y_pred_labels = (y_pred > 0.5).astype(int)
        batch_accuracy = self.accuracy_score(y_pred_labels, y_batch)
        epoch_train_accuracy += batch_accuracy

        num_batches += 1
      
      avg_train_loss =  epoch_loss / num_batches
      avg_train_accuracy = epoch_train_accuracy / num_batches
      train_losses.append(avg_train_loss.item())
      train_accuracies.append(avg_train_accuracy.item())

      if validation_data is not None:
        
        X_test, y_test = validation_data
        test_loss, test_accuracy = self.evaluate(X_test, y_test, batch_size)
        
        test_losses.append(test_loss.tolist())
        
        test_accuracies.append(test_accuracy.item())

     
      if epoch % 10 == 0:
        print(f"Epoch : {epoch} : Loss : {float(test_loss):.4f}")

      if early_stopping_patience is not None and er(test_loss):
        break
        
    end_time = time.time()
    
    History = {'Train_losses':train_losses,
               'Test_losses':test_losses,
               'Train_accuracy':train_accuracies,
               'Test_accuracy':test_accuracies,
               'Time_Elapsed':end_time - start_time
               }
               
    return History
  
  def evaluate(self, X, y, batch_size=64):
    self.eval()  
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    mini_batches = create_mini_batches(X, y, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
    
    for X_batch, y_batch in mini_batches:
        y_pred = self.forward(X_batch)
        loss = self.criterion(y_batch, y_pred)
        total_loss += loss
        
        y_pred_labels = (y_pred > 0.5).astype(int)
        batch_accuracy = self.accuracy_score(y_pred_labels, y_batch)
        total_accuracy += batch_accuracy
        num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches

  
  def train(self):
    self.training = True
    
  def eval(self):
    self.training = False
  
  def predict(self,X):

    if(len(X.shape) == 1):
      X = X.reshape(-1,1)

    predictions = self.forward(X)

    return (predictions>0.5).astype(int)
  
  def accuracy_score(self,y_pred,y_true):
    
    correct = np.sum(y_pred == y_true,axis=1)
    
    return correct / y_true.shape[1]
