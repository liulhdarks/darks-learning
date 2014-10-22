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

import darks.learning.SupervisedLearning;

/**
 * Multiple layer neuron network. Just like back propagation neuron network.
 * 
 * @author Darks.Liu
 *
 */
public class MultiLayerNeuronNetwork implements SupervisedLearning
{
	
	public MlpConfig config = new MlpConfig();

	private HiddenLayer[] hiddenLayers;
	
	private OutputLayer outputLayer;
	
	public MultiLayerNeuronNetwork()
	{
		
	}
	
	private void initialize()
	{
		if (hiddenLayers == null || outputLayer == null)
		{
			int hiddenCount = config.hiddenLayouts.length;
			hiddenLayers = new HiddenLayer[hiddenCount];
			for (int i = 0; i < hiddenCount; i++)
			{
				hiddenLayers[i] = new HiddenLayer(config, i);
			}
			outputLayer = new OutputLayer();
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DoubleMatrix input, DoubleMatrix output)
	{
		initialize();
		
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix predict(DoubleMatrix input)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
}
