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
package darks.learning.cluster;

public class ClusterPoint<T>
{

	T object;
	
	double similar;
	
	public ClusterPoint()
	{
		
	}
	
	public ClusterPoint(T object)
	{
		this.object = object;
	}
	
	public ClusterPoint(T object, double similar)
	{
		this.object = object;
		this.similar = similar;
	}

	
	
	public T getObject()
	{
		return object;
	}

	public void setObject(T object)
	{
		this.object = object;
	}

	public double getSimilar()
	{
		return similar;
	}

	public void setSimilar(double similar)
	{
		this.similar = similar;
	}

	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + ((object == null) ? 0 : object.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		@SuppressWarnings("unchecked")
		ClusterPoint<T> other = (ClusterPoint<T>) obj;
		if (object == null)
		{
			if (other.object != null)
				return false;
		}
		else if (!object.equals(other.object))
			return false;
		return true;
	}
	
	
}
