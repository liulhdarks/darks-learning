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

import static darks.learning.common.utils.MatrixHelper.oneMinus;
import org.jblas.DoubleMatrix;

import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.gradient.GradientComputer;

/**
 * Hidden layer of multiple layers perceptron.
 * 
 * @author Darks.Liu
 *
 */
public class OutputLayer extends AbstractNeuronNetwork
{
    LayerConfig config = new LayerConfig();
    
    DoubleMatrix output;
    
    DoubleMatrix error;
    
    
    public OutputLayer()
    {
        
    }
    
    public OutputLayer(MlpConfig parentConfig)
    {
        config.setLayerSize(parentConfig.outputLayerSize)
            .setL2(parentConfig.L2)
            .setLossType(parentConfig.lossType)
            .setNormalized(parentConfig.normalized)
            .setRandomFunction(parentConfig.randomFunction)
            .setUseRegularization(parentConfig.useRegularization)
            .setUseAdaGrad(parentConfig.useAdaGrad)
            .setLearnRate(parentConfig.learnRate);
        int lastSize = parentConfig.hiddenLayouts[parentConfig.hiddenLayouts.length - 1];
        weights = DoubleMatrix.rand(lastSize, config.layerSize);
        hBias = DoubleMatrix.rand(config.layerSize);
        initialize(config);
    }


	@Override
	public GradientComputer getGradient(DoubleMatrix input)
	{
        gradComputer.setBatchSize(input.rows);
        gradComputer.setWeights(weights);
        gradComputer.sethBias(hBias);
        gradComputer.computeGradient(error.mul(output), null, error);
        return gradComputer;
	}

	@Override
	public DoubleMatrix propForward(DoubleMatrix v)
	{
	    if (v.isColumnVector())
        {
            v = v.transpose();
        }
        DoubleMatrix preProb = v.mmul(weights);
        preProb.addiRowVector(hBias);
        output = config.activateFunction.activate(preProb);
        return output;
	}

	@Override
	public DoubleMatrix propBackward(DoubleMatrix expect)
	{
	    error = output.mul(oneMinus(output)).mul(expect.sub(output));
	    return error.mmul(weights.transpose());
	}

	public void update(DoubleMatrix input)
	{
	    GradientComputer grad = getGradient(input);
	    weights.addiRowVector(grad.getwGradient().columnMeans());
	    hBias.addi(grad.gethGradient().columnMeans());
	}
	
    public DoubleMatrix getOutput()
    {
        return output;
    }

    public void setOutput(DoubleMatrix output)
    {
        this.output = output;
    }

    public DoubleMatrix getError()
    {
        return error;
    }

    public void setError(DoubleMatrix error)
    {
        this.error = error;
    }
	
	

}
