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

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.IOUtils;
import darks.learning.exceptions.ModelException;

/**
 * GIS model 
 * @author Darks.Liu
 *
 */
public class GISModel extends MaxentModel
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8820339300824239323L;
	
	private static final Logger log = LoggerFactory.getLogger(GISModel.class);

    int[][] modelIndexs;
    
    Map<String, Integer> termIndexMap;
    
    DoubleMatrix lambda;
    

    public GISModel()
    {
    }
	

    public GISModel(List<String> labels, int[][] modelIndexs, Map<String, Integer> termIndexMap, DoubleMatrix lambda)
    {
        super(labels);
        this.modelIndexs = modelIndexs;
        this.termIndexMap = termIndexMap;
        this.lambda = lambda;
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
            oos.flush();
            return true;
        }
        catch (Exception e)
        {
            log.error(e.getMessage(), e);
        }
        return false;
    }
    
    /**
     * Read GIS model from target file
     * 
     * @param file Model file
     * @return GIS model
     */
    public static GISModel readModel(File file)
    {
        if (!file.exists())
            return null;
        InputStream ins = null;
        try
        {
            ins = new BufferedInputStream(new FileInputStream(file));
            return readModel(ins);
        }
        catch (Exception e)
        {
            log.error(e.getMessage(), e);
            return null;
        }
        finally
        {
            IOUtils.closeStream(ins);
        }
    }

    /**
     * Read GIS model from input stream
     * 
     * @param ins Model input stream
     * @return GIS model
     * @throws Exception
     */
    public static GISModel readModel(InputStream ins) throws Exception
    {
    	ObjectInputStream ois = new ObjectInputStream(ins);
    	GISModel model = (GISModel) ois.readObject();
    	if (model.labels == null || model.labels.isEmpty()
    			|| model.modelIndexs == null || model.modelIndexs.length == 0
    			|| model.termIndexMap == null || model.termIndexMap.isEmpty())
    	{
    		throw new ModelException("Invalid GIS model." + model);
    	}
    	return model;
    }


    public int[][] getModelIndexs()
	{
		return modelIndexs;
	}

	public Map<String, Integer> getTermIndexMap()
	{
		return termIndexMap;
	}
	
	public DoubleMatrix getLambda()
	{
		return lambda;
	}

	@Override
	public String toString()
	{
		return "GISModel [modelIndexs=" + Arrays.toString(modelIndexs) + ", termIndexMap="
				+ termIndexMap + ", labels=" + labels + "]";
	}
	
	
}
