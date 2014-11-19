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
package darks.learning.eval;


public class ROCPoint implements Comparable<ROCPoint>
{
	double tpr;
	
	double fpr;
	
	String label;
	
	ROCPoint prev;
	
	ROCPoint next;
	
	double k;
	
	boolean valid = true;

	public ROCPoint(double tpr, double fpr, String label)
	{
		this.tpr = tpr;
		this.fpr = fpr;
		this.label = label;
	}

	public ROCPoint(double tpr, double fpr, String label, boolean valid)
	{
		this.tpr = tpr;
		this.fpr = fpr;
		this.label = label;
		this.valid = valid;
	}


	@Override
	public String toString()
	{
		return "ROCPlot [tpr=" + tpr + ", fpr=" + fpr + ", label=" + label + ", k=" + k + "]";
	}


	@Override
	public int compareTo(ROCPoint o)
	{
		int ret = Double.compare(fpr, o.fpr);
		if (ret == 0)
		{
			ret = Double.compare(tpr, o.tpr);
			if (ret == 0)
			{
				return label.compareTo(o.label);
			}
		}
		return ret;
	}
	
}
