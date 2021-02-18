import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    public void softmax() {
    	// access e^Zi for each output unit i and do the softmax calculation here
    	double sum = 0.0;
    	for(int i = 0; i < outputNodes.size(); i++) { // denominator
    		sum += outputNodes.get(i).getOutput();
    	}
    	for(int j = 0; j < outputNodes.size(); j++) { // numerators
    		outputNodes.get(j).softmaxHelper((outputNodes.get(j).getOutput())/sum);
    		//System.out.println("in softmax: " + outputNodes.get(j).getOutput());
    	}
    	
    }
    
    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        // TODO: add code here
    	int idx = 0;
    	double max = 0.0;
    	forwardPass(instance);
    	for(int i = 0; i < outputNodes.size(); i++) {
    		if(outputNodes.get(i).getOutput() > max) {
    			max = outputNodes.get(i).getOutput();
    			idx = i;
    		}
    	}
        return idx;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        // TODO: add code here
    	int count = 0;
    	double totalLoss = 0.0;
    	while( count < maxEpoch) { // for each epoch
    		Collections.shuffle(trainingSet, random);
    		totalLoss = 0.0;
    		for(int i = 0; i < trainingSet.size(); i++) { // for each instance
    			Instance example = trainingSet.get(i);
    			forwardPass(example); // forward pass
    			
    			// back propagation
    			for(int k = 0; k < outputNodes.size(); k++) {
    				// calculate delta for each output node
    				outputNodes.get(k).calculateDelta(example.classValues.get(k));
    			}
    			for (int j = 0; j < hiddenNodes.size()-1; j++) {
    				// calculate delta for each hidden node
    				double sum = 0.0;
    				for(int l = 0; l < outputNodes.size(); l++) {
    					sum = sum + (outputNodes.get(l).parents.get(j).weight) * (outputNodes.get(l).delta);
    				}
    				hiddenNodes.get(j).calculateDelta(sum);
    			}
    			for(int c = 0; c < outputNodes.size(); c++) {
    				outputNodes.get(c).updateWeight(learningRate);
    			}
    			for(int d = 0; d < hiddenNodes.size()-1; d++) {
    				hiddenNodes.get(d).updateWeight(learningRate);
    			}
    		}
    		
    		for(int r = 0 ; r < trainingSet.size(); r++) {
    			totalLoss += loss(trainingSet.get(r)); // loss between target and real output
    		}
    		totalLoss = totalLoss/trainingSet.size();
    		System.out.printf("Epoch: " + count + ", Loss: %.3e\n", totalLoss);
    		//printWeight();
    		count++;
    	}
    }
    
	public void forwardPass(Instance example) {
		for (int j = 0; j < example.attributes.size(); j++) { // input# = attr# + 1(bias)
			inputNodes.get(j).setInput(example.attributes.get(j));
		}
		for (int k = 0; k < hiddenNodes.size() - 1; k++) {
			hiddenNodes.get(k).calculateOutput();
		}
		for (int t = 0; t < outputNodes.size(); t++) {
			outputNodes.get(t).calculateOutput();
		}
		softmax();
	}

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here
    	double loss = 0.0;
    	forwardPass(instance);
    	
    	for(int i = 0; i < outputNodes.size(); i++) {
    		loss += instance.classValues.get(i) * Math.log(outputNodes.get(i).getOutput());
    	}
        return (-1)*loss;
    }
    
/*    private void printWeight() {
    	System.out.println("input to hidden");
    	for (int i = 0; i < hiddenNodes.size()-1; i++) {
    		for (int j = 0; j < hiddenNodes.get(i).parents.size(); j++) {
    			System.out.println(hiddenNodes.get(i).parents.get(j).weight);
    		}
    	}
    	System.out.println("hidden to output");
    	for (int k = 0; k < outputNodes.size(); k++) {
    		for (int m = 0; m < hiddenNodes.size(); m++) {
    			System.out.println(outputNodes.get(k).parents.get(m).weight);
    		}
    	}
    }
*/
    
}
