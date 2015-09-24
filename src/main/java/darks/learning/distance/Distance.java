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
package darks.learning.distance;

public abstract class Distance<T>
{
	public static final int TYPE_DISTANCE = 1;
	
	public static final int TYPE_SIMILAR = 2;
	
	protected int type;
	
	public Distance(int type)
	{
		this.type = type;
	}

	public abstract double distance(T a, T b);
	
	public boolean compare(double src, double similar)
	{
		switch(type)
		{
		case TYPE_SIMILAR:
			return src >= similar;
		case TYPE_DISTANCE:
			return src <= similar;
		default:
			return src <= similar;
		}
	}
}
