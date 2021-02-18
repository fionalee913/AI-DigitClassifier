import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    public double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here
        	//double z = 0.0;
        	double pre = 0.0;
        	double w = 0.0;
        	this.inputValue = 0.0;
    		for (int i = 0; i < parents.size(); i++) {
    			// Z = sum of output from all connected parents * the corresponding weight
    			pre = parents.get(i).node.getOutput();
    			w = parents.get(i).weight;
    			this.inputValue += (pre * w);
    			//System.out.println("type: " + type + " parent: " + pre + " weight: " + w);
    		}
        	if(type == 2) {
        		this.outputValue = Math.max(this.inputValue, 0); // ReLU for hidden layer unit
        	}
        	if(type == 4) {
        		// e^z for output layer unit
        		this.outputValue = Math.exp(this.inputValue);
        		//System.out.println("calcOut: z = " + this.outputValue);
        	}
        }
    }
    
    public void softmaxHelper(double output) {
    	this.outputValue = output;
    }

    //Gets the output value
    public double getOutput() {
        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
	public void calculateDelta(double product) {
		if (type == 2 || type == 4) {
			// TODO: add code here
			if (type == 2) {
				if (this.inputValue <= 0) {
					this.delta = 0;
				} else {
					this.delta = product;
				}
			} 
			if (type ==4) {
				this.delta = product - outputValue;
			}
		}
	}

    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here
        	for(int i = 0; i < parents.size(); i++) {
        		 parents.get(i).weight += (learningRate * parents.get(i).node.getOutput() * this.delta);
        		 //System.out.println("update: type = " + type + " weight = " + parents.get(i).weight);
        	}
        }
    }
}


