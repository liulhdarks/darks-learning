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
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.basic.TfIdf;
import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.IOUtils;
import darks.learning.exceptions.CorpusException;

/**
 * Train corpus
 * @author Darks.Liu
 *
 */
public class Corpus
{
    
    public final static int TYPE_WORD_FREQ = 0;
    
    public final static int TYPE_TF_IDF = 1;
    
    public final static int TYPE_TF_IDF_FREQ = 3;

	private static Logger log = LoggerFactory.getLogger(Corpus.class);
	
	private File file;
	
	private FreqCount<String> wordFreq;
	
	private BufferedReader reader;
	
	private StopwordDictionary stopwordDictionary;
	
	private long totalVocabCount = 0;
	
	private long totalLineCount = 0;
	
	private long totalUniqueCount = 0;
	
	private TfIdf tfIDF;
	
	public Corpus()
	{
		
	}
	
	public Corpus(File file, FreqCount<String> wordFreq)
	{
		this.file = file;
		this.wordFreq = wordFreq;
	}
    
    public Corpus(FreqCount<String> wordFreq, TfIdf tfIDF, StopwordDictionary stopwordDictionary, 
        long totalVocabCount, long totalLineCount, long totalUniqueCount)
    {
        this.wordFreq = wordFreq;
        this.tfIDF = tfIDF;
        this.stopwordDictionary = stopwordDictionary;
        this.totalVocabCount = totalVocabCount;
        this.totalLineCount = totalLineCount;
        this.totalUniqueCount = totalUniqueCount;
    }
	
	/**
	 * Read one line from reader
	 * @return Line string
	 */
	public String readCorpusLine()
	{
		if (reader == null)
		{
			try
			{
				reader = new BufferedReader(new FileReader(file));
			}
			catch (FileNotFoundException e)
			{
				throw new CorpusException(e.getMessage(), e);
			}
		}
		try
		{
			return reader.readLine();
		}
		catch (IOException e)
		{
			log.error(e.getMessage(), e);
		}
		return null;
	}
	
	/**
	 * Close reader 
	 */
	public void closeReader()
	{
		IOUtils.closeStream(reader);
	}

	public BufferedReader getReader()
	{
		return reader;
	}

	public void setReader(BufferedReader reader)
	{
		this.reader = reader;
	}

	public FreqCount<String> getWordFreq()
	{
		return wordFreq;
	}

	public void setWordFreq(FreqCount<String> wordFreq)
	{
		this.wordFreq = wordFreq;
	}

	public StopwordDictionary getStopwordDictionary()
	{
		return stopwordDictionary;
	}

	public void setStopwordDictionary(StopwordDictionary stopwordDictionary)
	{
		this.stopwordDictionary = stopwordDictionary;
	}

	public File getFile()
	{
		return file;
	}

	public void setFile(File file)
	{
		this.file = file;
	}

	public long getTotalVocabCount()
	{
		return totalVocabCount;
	}

	public void setTotalVocabCount(long totalVocabCount)
	{
		this.totalVocabCount = totalVocabCount;
	}
	
	public TfIdf getTfIDF()
    {
        return tfIDF;
    }

    public void setTfIDF(TfIdf tfIDF)
    {
        this.tfIDF = tfIDF;
    }

    
    
    public long getTotalLineCount()
    {
        return totalLineCount;
    }

    public void setTotalLineCount(long totalLineCount)
    {
        this.totalLineCount = totalLineCount;
    }

    public long getTotalUniqueCount()
    {
        return totalUniqueCount;
    }

    public void setTotalUniqueCount(long totalUniqueCount)
    {
        this.totalUniqueCount = totalUniqueCount;
    }

    @Override
	public String toString()
	{
		return "Corpus [file=" + file + ", totalVocabCount=" + totalVocabCount + "]";
	}
	
	
}
