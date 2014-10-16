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
package darks.learning.neuron;

import java.util.HashMap;
import java.util.Map;

import org.jblas.DoubleMatrix;

import darks.learning.exceptions.LearningException;
import darks.learning.neuron.gradient.GradientComputer;
import darks.learning.neuron.gradient.NNGradientComputer;
import darks.learning.optimize.LearningOptimizer;
import darks.learning.optimize.LearningOptimizer.OptimizeType;
import darks.learning.optimize.LineSearchOptimizer;
import darks.learning.optimize.NoneNeuronNetworkOptimizer;

/**
 * Abstract neuronm network
 * 
 * @author Darks.Liu
 *
 */
public abstract class AbstractNeuronNetwork implements ReConstructon
{

	protected GradientComputer gradComputer;
	
	protected DoubleMatrix weights;
	
	protected DoubleMatrix vBias;
	
	protected DoubleMatrix hBias;
	
	protected DoubleMatrix vInput;
	
	protected DoubleMatrix sigma;
	
	protected DoubleMatrix hiddenSigma;
	
	private NNConfig cfg;
	
	protected LearningOptimizer optimizer;
	
	protected int numIterate;
	
	public void initialize(NNConfig cfg)
	{
		this.cfg = cfg;
		gradComputer = new NNGradientComputer(cfg);
		buildOptimizer();
	}
	
	public double getLossValue()
	{
		cfg.lossFunction.setInput(vInput);
		cfg.lossFunction.setReConstructon(this);
		return cfg.lossFunction.getLossValue();
	}

	/**
	 * Get gradient through current input values
	 * 
	 * @return GradientComputer
	 */
	public GradientComputer getGradient()
	{
		return getGradient(vInput);
	}

	/**
	 * Get gradient through specify input values
	 * 
	 * @param input Specify input values
	 * @return GradientComputer
	 */
	public abstract GradientComputer getGradient(DoubleMatrix input);

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct()
	{
		return reconstruct(vInput);
	}
	

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DoubleMatrix reconstruct(DoubleMatrix input)
	{
		return null;
	}
	
	public void addGrad(GradientComputer grad)
	{
		if (weights != null && grad.getwGradient() != null)
		{
			weights.addi(grad.getwGradient());
		}
		if (hBias != null && grad.gethGradient() != null)
		{
			hBias.addi(grad.gethGradient());
		}
		if (vBias != null && grad.getvGradient() != null)
		{
			vBias.addi(grad.getvGradient());
		}
	}
	
	public void subGrad(GradientComputer grad)
	{
		if (weights != null && grad.getwGradient() != null)
		{
			weights.subi(grad.getwGradient());
		}
		if (hBias != null && grad.gethGradient() != null)
		{
			hBias.subi(grad.gethGradient());
		}
		if (vBias != null && grad.getvGradient() != null)
		{
			vBias.subi(grad.getvGradient());
		}
	}
	
	private void buildOptimizer()
	{
		if (cfg.optimizeType == OptimizeType.NONE)
		{
			optimizer = new NoneNeuronNetworkOptimizer(this); 
		}
		else if (cfg.optimizeType == OptimizeType.LINE_SEARCH)
		{
			optimizer = new LineSearchOptimizer(this); 
		}
		else
		{
			throw new LearningException("Cannot find optimize type " + cfg.optimizeType);
		}
	}
	
	public Map<String, Object> backup()
	{
		Map<String, Object> result = new HashMap<String, Object>();
		result.put("weights", weights.dup());
		result.put("vbias", vBias.dup());
		result.put("hbias", hBias.dup());
		if (gradComputer != null)
		{
			result.put("wgrad", gradComputer.getwGradient() == null ? null : gradComputer.getwGradient().dup());
			result.put("vgrad", gradComputer.getvGradient() == null ? null : gradComputer.getvGradient().dup());
			result.put("hgrad", gradComputer.gethGradient() == null ? null : gradComputer.gethGradient().dup());
		}
		return result;
	}
	
	public void restore(Map<String, Object> pack)
	{
		Object W = pack.get("weights");
		if (W != null)
		{
			weights = ((DoubleMatrix)W).dup();
		}
		Object vbias = pack.get("vbias");
		if (vbias != null)
		{
			vBias = ((DoubleMatrix)vbias).dup();
		}
		Object hbias = pack.get("hbias");
		if (hbias != null)
		{
			hBias = ((DoubleMatrix)hbias).dup();
		}
		if (gradComputer != null)
		{
			Object grad = pack.get("wgrad");
			gradComputer.setwGradient(grad == null ? null : ((DoubleMatrix)grad).dup());
			grad = pack.get("vgrad");
			gradComputer.setvGradient(grad == null ? null : ((DoubleMatrix)grad).dup());
			grad = pack.get("hgrad");
			gradComputer.sethGradient(grad == null ? null : ((DoubleMatrix)grad).dup());
		}
	}

	public NNConfig config()
	{
		return cfg;
	}

	public GradientComputer getGradComputer()
	{
		return gradComputer;
	}

	public int getNumIterate()
	{
		return numIterate;
	}

	public void setNumIterate(int numIterate)
	{
		this.numIterate = numIterate;
		if (gradComputer != null)
		{
			gradComputer.setNumIterate(numIterate);
		}
	}

	public DoubleMatrix getWeights()
	{
		return weights;
	}

	public void setWeights(DoubleMatrix weights)
	{
		this.weights = weights;
	}

	public DoubleMatrix getvBias()
	{
		return vBias;
	}

	public void setvBias(DoubleMatrix vBias)
	{
		this.vBias = vBias;
	}

	public DoubleMatrix gethBias()
	{
		return hBias;
	}

	public void sethBias(DoubleMatrix hBias)
	{
		this.hBias = hBias;
	}

	public DoubleMatrix getvInput()
	{
		return vInput;
	}

	public void setvInput(DoubleMatrix vInput)
	{
		this.vInput = vInput;
	}

	public void setLearnRate(double learnRate)
	{
		if (gradComputer != null)
		{
			gradComputer.setLearnRate(learnRate);
		}
	}

	public double getLearnRate()
	{
		if (gradComputer != null)
		{
			return gradComputer.getLearnRate();
		}
		else
		{
			return cfg.learnRate;
		}
	}
	
}
