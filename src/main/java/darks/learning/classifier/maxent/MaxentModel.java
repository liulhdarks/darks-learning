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

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.IOUtils;

public abstract class MaxentModel implements Serializable
{

    /**
	 * 
	 */
	private static final long serialVersionUID = 641755681544620064L;
	
	private static final Logger log = LoggerFactory.getLogger(MaxentModel.class);
	
	
	protected List<String> labels = new ArrayList<String>();
	


    public MaxentModel()
    {
    }

	public MaxentModel(List<String> labels)
    {
        this.labels = labels;
    }



    
    /**
     * Save maxent model to file
     * 
     * @param file Target model file
     * @return If success, return true
     */
    public boolean saveModel(File file)
    {
        OutputStream out = null;
        try
        {
            out = new BufferedOutputStream(new FileOutputStream(file));
            return saveModel(out);
        }
        catch (Exception e)
        {
            log.error(e.getMessage(), e);
            return false;
        }
        finally
        {
            IOUtils.closeStream(out);
        }
    }

    /**
     * Save maxent model to output stream
     * 
     * @param file Target model file
     * @return If success, return true
     */
    public abstract boolean saveModel(OutputStream out);
    
	

    public List<String> getLabels()
	{
		return labels;
	}
    
    
}
