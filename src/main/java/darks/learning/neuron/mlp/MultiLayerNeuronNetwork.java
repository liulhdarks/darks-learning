/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package darks.learning.neuron.mlp;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.SupervisedLearning;
import darks.learning.lossfunc.LossFunction;
import darks.learning.neuron.ReConstructon;

/**
 * Multiple layer neuron network. Just like back propagation neuron network.
 * 
 * @author Darks.Liu
 *
 */
public class MultiLayerNeuronNetwork implements SupervisedLearning,ReConstructon
{
    
    private static Logger log = LoggerFactory.getLogger(MultiLayerNeuronNetwork.class);
	
	public MlpConfig config = new MlpConfig();

	private HiddenLayer[] hiddenLayers;
	
	private OutputLayer outputLayer;
	
	private DoubleMatrix vInput;
	
	private DoubleMatrix labels;
	
	double lastLoss = 0;
	
	public MultiLayerNeuronNetwork()
	{
	    config.lossFunction = LossFunction.lossFunc(LossFunction.MSE, config);
	}
	
	public void initialize()
	{
		if (hiddenLayers == null || outputLayer == null)
		{
			int hiddenCount = config.hiddenLayouts.length;
			hiddenLayers = new HiddenLayer[hiddenCount];
			for (int i = 0; i < hiddenCount; i++)
			{
				hiddenLayers[i] = new HiddenLayer(config, i);
				if (i > 0)
					hiddenLayers[i].prevLayer = hiddenLayers[i - 1];
			}
			outputLayer = new OutputLayer(config);
			outputLayer.prevLayer = hiddenLayers[hiddenCount - 1];
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void trainBatch(DoubleMatrix input, DoubleMatrix output)
	{
	    this.vInput = input;
	    this.labels = output;
		initialize();
		int maxIterateCount = config.maxIterateCount;
		boolean useCount = maxIterateCount > 0;
		int iterateNumber = 1;
		while (true)
		{
			double loss = train(iterateNumber, input, output);
	        if (log.isDebugEnabled())
	        {
	            log.debug("MLP iterate number " + iterateNumber + " loss:" + Math.abs(loss) + " tolerance:" + Math.abs(loss - lastLoss));
	        }
            if (Math.abs(loss - lastLoss) < 1.0e-10) 
            {
                log.info ("Gradient Ascent: Value difference " + Math.abs(loss - lastLoss) +" below " +
                        "tolerance; arriving converged.");
                break;
            }
            lastLoss = loss;
		    if (useCount && iterateNumber >= maxIterateCount)
		    {
		        break;
		    }
            iterateNumber++;
		}
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public double train(int iterateNumber, DoubleMatrix input, DoubleMatrix output)
	{
		iterate(iterateNumber, input, output);
		return getLossValue();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix input)
	{
	    int hiddenCount = hiddenLayers.length;
        for (int i = 0; i < hiddenCount; i++)
        {
            input = hiddenLayers[i].propForward(input);
        }
        return outputLayer.propForward(input);
	}
	
	private void iterate(int iterateNumber, DoubleMatrix initInput, DoubleMatrix output)
	{
	    int hiddenCount = hiddenLayers.length;
	    DoubleMatrix input = initInput;
        for (int i = 0; i < hiddenCount; i++)
        {
            hiddenLayers[i].setNumIterate(iterateNumber);
            input = hiddenLayers[i].propForward(input);
        }
        outputLayer.setNumIterate(iterateNumber);
        outputLayer.propForward(input);
        
        DoubleMatrix error = outputLayer.propBackward(output);
        outputLayer.update(initInput);
        for (int i = hiddenCount - 1; i >= 0; i--)
        {
            error = hiddenLayers[i].propBackward(error);
            hiddenLayers[i].update(initInput);
        }
	}

    @Override
    public DoubleMatrix reconstruct()
    {
        return outputLayer.getOutput();
    }

    @Override
    public DoubleMatrix reconstruct(DoubleMatrix input)
    {
        return outputLayer.getOutput();
    }
	
    
    public double getLossValue()
    {
        config.lossFunction.setInput(labels);
        config.lossFunction.setReConstructon(this);
        double val = config.lossFunction.getLossValue();
        return -val;
    }
    
    private void print()
    {
    	for (int i = 0; i < hiddenLayers.length; i++)
		{
			System.out.println(hiddenLayers[i].getWeights());
		}
		System.out.println(outputLayer.getWeights());
    }
}
