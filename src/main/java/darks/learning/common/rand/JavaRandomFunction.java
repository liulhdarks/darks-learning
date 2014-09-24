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
package darks.learning.common.rand;

import java.util.Random;

public class JavaRandomFunction implements RandomFunction
{

	private Random rand = new Random(System.currentTimeMillis());
	
	public double randDouble()
	{
		return rand.nextDouble();
	}

	public double randInt()
	{
		return rand.nextInt();
	}

	public double randInt(int max)
	{
		return rand.nextInt(max);
	}

}
