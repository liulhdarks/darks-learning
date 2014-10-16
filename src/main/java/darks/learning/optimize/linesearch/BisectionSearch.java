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
package darks.learning.optimize.linesearch;

import java.util.Map;
import java.util.TreeSet;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.neuron.AbstractNeuronNetwork;
import darks.learning.neuron.gradient.GradientComputer;

/**
 * Bisection line search
 * 
 * @author Darks.Liu
 *
 */
public class BisectionSearch extends LineSearch
{
	
	private static final Logger log = LoggerFactory.getLogger(BisectionSearch.class);
	
	private double learnSmallRate = 1e-3;

	private double learnBigRate = 0.98;

	public BisectionSearch(AbstractNeuronNetwork network)
	{
		super(network);
		log.debug("Initialziing bisection line search");
	}

	public BisectionSearch(AbstractNeuronNetwork network, double learnSmallRate, double learnBigRate)
	{
		super(network);
		log.debug("Initialziing bisection line search");
		this.learnSmallRate = learnSmallRate;
		this.learnBigRate = learnBigRate;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double optimize(DoubleMatrix x, int numIterate, double initStep)
	{
		double lr1 = learnBigRate;
		double lr2 = learnSmallRate;
		Map<String, Object> backup = network.backup();
		network.setLearnRate(lr1);
		GradientComputer grad = network.getGradient(x);
		network.addGrad(grad);
		double loss1 = network.getLossValue();
		double loss2 = applyLearnRate(backup, grad, lr2);
		TreeSet<LossPair> lossSet = new TreeSet<BisectionSearch.LossPair>();
		lossSet.add(new LossPair(lr1, loss1));
		lossSet.add(new LossPair(lr2, loss2));
		double lastMid = 0;
		for (int i = 0; i < maxIterateCount; i++)
		{
			double lrMid = getMidLearnRate(lossSet);
			if (lrMid == lastMid)
			{
				break;
			}
			lastMid = lrMid;
			double loss3 = applyLearnRate(backup, grad, lrMid);
			lossSet.add(new LossPair(lrMid, loss3));
			if (lossSet.size() > 2)
			{
				lossSet.pollLast();
			}
		}
		LossPair pair = lossSet.pollFirst();
		network.setLearnRate(pair.lr);
		applyLearnRate(backup, grad, pair.lr);
		return pair.lr;
	}
	
	private double getMidLearnRate(TreeSet<LossPair> lossSet)
	{
		int num = lossSet.size();
		double sum = 0;
		for (LossPair pair : lossSet)
		{
			sum += pair.lr;
		}
		return sum / num;
	}
	
	private double applyLearnRate(Map<String, Object> backup, GradientComputer grad, double lr)
	{
		network.restore(backup);
		network.setLearnRate(lr);
		grad.computeOriginGradient();
		network.addGrad(grad);
		return network.getLossValue();
	}
	
	class LossPair implements Comparable<LossPair>
	{
		
		double lr;
		
		double loss;
		
		public LossPair(double lr, double loss)
		{
			super();
			this.lr = lr;
			this.loss = Double.isNaN(loss) ? Double.NEGATIVE_INFINITY : loss;
		}

		@Override
		public int compareTo(LossPair o)
		{
			return Double.compare(o.loss, loss);
		}

		@Override
		public String toString()
		{
			return "LossPair [lr=" + lr + ", loss=" + loss + "]";
		}
		
	}

}
