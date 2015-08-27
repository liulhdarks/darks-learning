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

package darks.learning.common.blas;

public abstract class Matrix<T>
{
	
	protected int rowsCount;
	
	protected int columnsCount;

	/**
	 * Put element to position i:j
	 * 
	 * @param i Row number
	 * @param j Column number
	 * @param v Value object
	 */
	public abstract void put(int i, int j, T v);
	
	/**
	 * Get element from position i:j
	 * @param i Row number
	 * @param j Column number
	 * @return Value object
	 */
	public abstract T get(int i, int j);
	
	/**
	 * Check the matrix whether invalid
	 * @return if invalid, return true
	 */
	public abstract boolean checkNull();
	
	public int rows()
	{
		return rowsCount;
	}
	
	public int columns()
	{
		return columnsCount;
	}
}
