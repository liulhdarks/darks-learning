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
package darks.learning.common.minispantree;

public class GraphEdge<T> implements Comparable<GraphEdge<T>>         
{
	
	protected int from;
	
	protected int to;

	protected double weight;
	
	protected T value;
	
	public GraphEdge()
	{
		
	}
	
	public GraphEdge(int from, int to, double weight)
	{
		super();
		this.from = from;
		this.to = to;
		this.weight = weight;
	}
	
	public GraphEdge(int from, int to, double weight, T value)
	{
		super();
		this.from = from;
		this.to = to;
		this.weight = weight;
		this.value = value;
	}

	public double getWeight()
	{
		return weight;
	}

	public void setWeight(double weight)
	{
		this.weight = weight;
	}

	public int getFrom()
	{
		return from;
	}

	public void setFrom(int from)
	{
		this.from = from;
	}

	public int getTo()
	{
		return to;
	}

	public void setTo(int to)
	{
		this.to = to;
	}
	
	public T getValue()
	{
		return value;
	}

	public void setValue(T value)
	{
		this.value = value;
	}

	@Override
	public String toString()
	{
		return "[" + from + ":" + to + "=" + weight + (value == null ? "" : " " + value) + "]";
	}

	@Override
	public int compareTo(GraphEdge<T> o)
	{
		return Double.compare(weight, o.weight);
	}

	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + from;
		result = prime * result + to;
		return result;
	}

	@SuppressWarnings("unchecked")
	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		GraphEdge<T> other = (GraphEdge<T>) obj;
		if (from != other.from)
			return false;
		if (to != other.to)
			return false;
		return true;
	}
	
}
