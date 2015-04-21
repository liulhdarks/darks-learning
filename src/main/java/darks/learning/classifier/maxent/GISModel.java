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
package darks.learning.classifier.maxent;

import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GISModel extends MaxentModel
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8820339300824239323L;
	
	private static final Logger log = LoggerFactory.getLogger(GISModel.class);

    int[][] modelIndexs;
    
    Map<String, Integer> termIndexMap;
    

    public GISModel()
    {
    }
	

    public GISModel(List<String> labels, int[][] modelIndexs, Map<String, Integer> termIndexMap)
    {
        super(labels);
        this.modelIndexs = modelIndexs;
        this.termIndexMap = termIndexMap;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public boolean saveModel(OutputStream out)
    {
        ObjectOutputStream oos = null;
        try
        {
            oos = new ObjectOutputStream(out);
            oos.writeObject(this);
            return true;
        }
        catch (Exception e)
        {
            log.error(e.getMessage(), e);
        }
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean readModel(InputStream ins)
    {
        // TODO Auto-generated method stub
        return false;
    }


    public int[][] getModelIndexs()
	{
		return modelIndexs;
	}

	public Map<String, Integer> getTermIndexMap()
	{
		return termIndexMap;
	}
	
	
}
