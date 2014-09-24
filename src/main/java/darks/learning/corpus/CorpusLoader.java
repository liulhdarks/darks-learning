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
package darks.learning.corpus;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.StringTokenizer;

import org.apache.commons.math3.stat.Frequency;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.IOUtils;
import darks.learning.exceptions.CorpusException;

/**
 * Train corpus loader
 * @author Darks.Liu
 *
 */
public class CorpusLoader
{
	
	private static Logger log = LoggerFactory.getLogger(CorpusLoader.class);

	private StopwordDictionary stopwordDictionary = new StopwordDictionary();
	
	public CorpusLoader()
	{
		
	}
	
	/**
	 * Load corpus from file
	 * @param file Corpus file
	 * @return Corpus
	 */
	public Corpus loadFromFile(File file)
	{
		if (!file.exists())
		{
			throw new CorpusException("Cannot find corpus " + file);
		}
		BufferedReader reader = null;
		try
		{
			
			Frequency wordFreq = new Frequency(String.CASE_INSENSITIVE_ORDER);
			reader = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				if ("".equals(line.trim()))
				{
					continue;
				}
				StringTokenizer token = new StringTokenizer(line, " \t");
				while (token.hasMoreTokens())
				{
					String word = token.nextToken().trim();
					if (stopwordDictionary.contain(word))
					{
						continue;
					}
					wordFreq.addValue(word);
				}
			}
			return new Corpus(file, wordFreq, stopwordDictionary);
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
			return null;
		}
		finally
		{
			IOUtils.closeStream(reader);
		}
	}
	
	/**
	 * Add stop word to dictionary
	 * @param word Stop word dictionary
	 */
	public void addStopword(String word)
	{
		stopwordDictionary.add(word);
	}
	
	/**
	 * Remove stop word from dictionary
	 */
	public void removeStopword(String word)
	{
		stopwordDictionary.remove(word);
	}
	
	/**
	 * Check whether the stop word contained in dictionary
	 * @param whether contained
	 */
	public boolean containStopword(String word)
	{
		return stopwordDictionary.contain(word);
	}

	public StopwordDictionary getStopwordDictionary()
	{
		return stopwordDictionary;
	}
	
	
}
