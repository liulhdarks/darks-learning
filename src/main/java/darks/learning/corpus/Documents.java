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
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.IOUtils;

/**
 * Corpus document, which has input or labels
 * 
 * @author Darks.Liu
 *
 */
public class Documents implements Serializable
{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8078949561869787649L;


	private static Logger log = LoggerFactory.getLogger(Documents.class);
	
	
	private Map<String, List<Document>> labelsMap = new HashMap<String, List<Document>>();
	
	private FreqCount<String> termsFreq = new FreqCount<String>();
	
	private Map<String, String> docsMap = new HashMap<String, String>();

	/**
	 * Load document from file
	 * @param input Input file
	 * @param labels Label file
	 * @return Document
	 * @throws IOException
	 */
	public static Documents loadFromFile(File input, File labels) throws IOException
	{
		return loadFromFile(input, labels, null);
	}

	/**
	 * Load from file
	 * @param input Input file
	 * @param labels Label file
	 * @param charsetName File charset name
	 * @return Document
	 * @throws IOException
	 */
	public static Documents loadFromFile(File input, File labels, String charsetName) throws IOException
	{
		BufferedReader reader = null;
		BufferedReader readerLabel = null;
		if (charsetName != null)
		{
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(input), charsetName));
			readerLabel = new BufferedReader(new InputStreamReader(new FileInputStream(labels), charsetName));
		}
		else
		{
			reader = new BufferedReader(new FileReader(input));
			readerLabel = new BufferedReader(new FileReader(labels));
		}
		try
		{
			Documents docs = new Documents();
			String inputLine = null;
			String labelsLine = null;
			while ((inputLine = reader.readLine()) != null 
					&& (labelsLine = readerLabel.readLine()) != null)
			{
				inputLine = inputLine.trim();
				labelsLine = labelsLine.trim();
				if ("".equals(inputLine) || "".equals(labelsLine))
				{
					continue;
				}
				docs.addLine(inputLine, labelsLine);
			}
			return docs;
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
		finally
		{
			IOUtils.closeStream(reader);
			IOUtils.closeStream(readerLabel);
		}
		return null;
	}
	
	void addLine(String input, String label)
	{
		List<Document> docs = labelsMap.get(label);
		if (docs == null)
		{
			docs = new LinkedList<Document>();
			labelsMap.put(label, docs);
		}
		docs.add(new Document(termsFreq, input, label, " \t\n"));
		docsMap.put(input, label);
	}

	public Map<String, List<Document>> getLabelsMap()
	{
		return labelsMap;
	}
	
	
	public FreqCount<String> getTermsFreq()
	{
		return termsFreq;
	}

	
	public Map<String, String> getDocsMap()
	{
		return docsMap;
	}



	public static class Document implements Serializable
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = 700658812752564691L;

		String source;
		
		List<String> terms = new LinkedList<String>();
		
		String label;
		
		public Document(FreqCount<String> freq, String s, String label, String delim)
		{
			this.source = s;
			StringTokenizer token = new StringTokenizer(s, delim);
			while (token.hasMoreTokens())
			{
				String term = token.nextToken().trim();
				if ("".equals(term))
				{
					continue;
				}
				terms.add(term);
				freq.addValue(term);
			}
			this.label = label;
		}

		public List<String> getTerms()
		{
			return terms;
		}

		public void setTerms(List<String> terms)
		{
			this.terms = terms;
		}

		public String getLabel()
		{
			return label;
		}

		public void setLabel(String label)
		{
			this.label = label;
		}

		public String getSource()
		{
			return source;
		}

		public void setSource(String source)
		{
			this.source = source;
		}

		@Override
		public int hashCode()
		{
			final int prime = 31;
			int result = 1;
			result = prime * result + ((source == null) ? 0 : source.hashCode());
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
			Document other = (Document) obj;
			if (source == null)
			{
				if (other.source != null)
					return false;
			}
			else if (!source.equals(other.source))
				return false;
			return true;
		}
		
	}
	
	
}
