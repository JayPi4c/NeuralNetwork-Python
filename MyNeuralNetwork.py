from NeuralNetwork import NeuralNetwork
from datetime import datetime
import numpy
import scipy.special


# adjust function (like the map() function in Procesing) to scale and shift the inputs
def adjust(value, istart, istop, ostart, ostop):
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

# returns the systemtime as a string
def getSystemTime():
	return str(datetime.now())



# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 500
output_nodes = 10

# learning rate
learning_rate = 0.03

# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# train the neural network

print("Training of the neural network begins at " + getSystemTime())

# load the mnist test data CSV file into a list
test_data_file_10 = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list_10 = test_data_file_10.readlines()
test_data_file_10.close()


# epochs is the number of times the training data set is used for training
epochs = 7

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        #inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        inputs = adjust(numpy.asfarray(all_values[1:]), 0, 255, 0.01, 1)
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    print("Epoch " + str(e + 1) + " of " + str(epochs) + ": ")
    print(getSystemTime())

	# scorecard for how well the network performs, initially empty
    scorecard_10 = []

	# go through all the records in the test data set
    for record in test_data_list_10:
    	# split the record by the ',' commas
        all_values_10 = record.split(',')
    	# correct answer is first value
        correct_label_10 = int(all_values_10[0])
        print("correct label: " + str(correct_label_10))
    	# scale and shift the inputs
        #inputs_10 = (numpy.asfarray(all_values_10[1:]) / 255.0 * 0.99) + 0.01
        inputs_10 = adjust(numpy.asfarray(all_values_10[1:]), 0, 255, 0.01, 1)
   		# query the network
        outputs_10 = n.query(inputs_10)
    	# the index of the highest value corresponds to the label
        label_10 = numpy.argmax(outputs_10)
        print("Network's choice: " + str(label_10))
    	# append correct or incorrect to list
        if (label_10 == correct_label_10):
        	# network's answer matches correct answer, add 1 to scorecard
       	    scorecard_10.append(1)
        else:
        	# network's answer doesn't match correct answer, add 0 to scorecard
            scorecard_10.append(0)
            pass
    
        pass

	# calculate the performance score, the fraction of correct answers
    scorecard_array_10 = numpy.asarray(scorecard_10)
    final_performance_10 = scorecard_array_10.sum() / float(scorecard_array_10.size) 
    print ("performance = " + str(final_performance_10))

    pass

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()




# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print("correct label: " + str(correct_label))
    # scale and shift the inputs
    #inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    inputs = adjust(numpy.asfarray(all_values[1:]), 0, 255, 0.01, 1)
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("Network's choice: " + str(label))
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
final_performance = scorecard_array.sum() / float(scorecard_array.size) 
print ("performance = " + str(final_performance) + " after " + str(epochs) + " Epochs")
print(getSystemTime())
print("DONE!")
