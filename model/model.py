import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        '''
        input_size: size of the input features
        hidden_size: the number of neurons in the hidden layer
        num_classes: The number of classes in the classification task
        RelU : This activation function is used to introduce non linearity into the network

        def forward(): This is a forward pass function, process of computing the output of the network given an input.
        we will not explicitly pass the input data 'x' to the class,
        instead we pass the input data into an instance of the class.
        Eg: model = NeuralNet(input_size, hidden_size, num_classes)
            output = model(input_data)

        

        '''

        #calling the constructor of parent class 'nn.Module'
        #Necessary to initialize the class as a pytorch module
        super(NeuralNet, self).__init__()  
        #fully connected linear layers which maps the input feature to the hidden layer
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() 
    

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)

        out = self.l2(out)
        out = self.relu(out)

        out = self.l3(out)
        #not using any activation function because we want a wide range of possible words as output
        #so we are making the output as unbounded
        return out


        
    

