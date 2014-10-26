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
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.IOUtils;
import darks.learning.exceptions.CorpusException;
import darks.learning.lsa.TfIdf;

/**
 * Train corpus loader
 * @author Darks.Liu
 *
 */
public class CorpusLoader
{
	
	private static Logger log = LoggerFactory.getLogger(CorpusLoader.class);

	private StopwordDictionary stopwordDictionary = new StopwordDictionary();
	
	private List<CorpusFilter> corpusFilters = new LinkedList<CorpusFilter>();
	
	private int corpusType = Corpus.TYPE_WORD_FREQ;
	
	public CorpusLoader()
	{
		
	}
    
    public CorpusLoader(int corpusType)
    {
        this.corpusType = corpusType;
    }
    

    public Corpus loadFromFile(File file)
    {
        if (!file.exists())
        {
            throw new CorpusException("Cannot find corpus " + file);
        }

        try
        {
            Corpus corpus = loadFromReader(new BufferedReader(new FileReader(file)));
            corpus.setFile(file);
            return corpus;
        }
        catch (FileNotFoundException e)
        {
            log.error(e.getMessage(), e);
        }
        return null;
    }
    
	/**
	 * Load corpus from file
	 * @param file Corpus file
	 * @return Corpus
	 */
	public Corpus loadFromReader(BufferedReader reader)
	{
		try
		{
			long totalWordsCount = 0;
			long totalLineCount = 0;
			FreqCount<String> wordFreq = new FreqCount<String>();
			TfIdf tfidf = new TfIdf();
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				if ("".equals(line.trim()))
				{
					continue;
				}
				//line = line.trim();
				StringTokenizer token = new StringTokenizer(line, " \t");
				while (token.hasMoreTokens())
				{
					String word = token.nextToken().trim();
					if (stopwordDictionary.contain(word) || checkFilter(word))
					{
						continue;
					}
					totalWordsCount++;
					if (corpusType == Corpus.TYPE_WORD_FREQ || corpusType == Corpus.TYPE_TF_IDF_FREQ)
					{
	                    wordFreq.addValue(word);
					}
					if (corpusType == Corpus.TYPE_TF_IDF || corpusType == Corpus.TYPE_TF_IDF_FREQ)
					{
					    tfidf.addWord(line, word);
					}
				}
				totalLineCount++;
			}
			long totalUniqueCount = corpusType == Corpus.TYPE_WORD_FREQ ? wordFreq.getUniqueCount() : tfidf.getUniqueWordsCount();
			return new Corpus(wordFreq, tfidf, stopwordDictionary, totalWordsCount, totalLineCount, totalUniqueCount);
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
	
	private boolean checkFilter(String s)
	{
		for (CorpusFilter filter : corpusFilters)
		{
			if (filter.filter(s))
			{
				return true;
			}
		}
		return false;
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
	
	/**
	 * 添加停用词文件
	 * @param file 停用词文件
	 */
	public void addStopwords(File file)
	{
		BufferedReader reader = null;
		try
		{
			reader = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.startsWith("#") || "".equals(line))
				{
					continue;
				}
				addStopword(line);
			}
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
		finally
		{
			IOUtils.closeStream(reader);
		}
	}
	
	/**
	 * Add corpus filter
	 * @param filter CorpusFilter
	 */
	public CorpusLoader addFilter(CorpusFilter filter)
	{
		corpusFilters.add(filter);
		return this;
	}

	public StopwordDictionary getStopwordDictionary()
	{
		return stopwordDictionary;
	}
	
	
}
