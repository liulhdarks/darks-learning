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

/**
 * Random number by word2vec random
 * 
 * @author Darks.Liu
 *
 */
public class WordRandomFunction implements RandomFunction
{

	private Random rand = new Random(System.currentTimeMillis());
	
	long nextRandom = 5;

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double randDouble()
	{
		nextRandom = nextRandom * 25214903917L + 11;
		return (nextRandom & 0xFFFF) / (double) 65536;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int randInt()
	{
		return rand.nextInt();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int randInt(int max)
	{
		return rand.nextInt(max);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public long randLong()
	{
		nextRandom = nextRandom * 25214903917L + 11;
		return nextRandom;
	}
	

}
