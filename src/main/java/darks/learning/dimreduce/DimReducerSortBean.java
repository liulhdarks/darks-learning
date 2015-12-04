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
package darks.learning.dimreduce;

public class DimReducerSortBean implements Comparable<DimReducerSortBean>
{

	String term;
	
	double value;
	
	

	public DimReducerSortBean()
	{
		super();
	}

	public DimReducerSortBean(String term, double value)
	{
		super();
		this.term = term;
		this.value = value;
	}

	public String getTerm()
	{
		return term;
	}

	public void setTerm(String term)
	{
		this.term = term;
	}

	public double getValue()
	{
		return value;
	}

	public void setValue(double value)
	{
		this.value = value;
	}

	@Override
	public int compareTo(DimReducerSortBean o)
	{
		return Double.compare(o.value, value);
	}
	
	
	
}
