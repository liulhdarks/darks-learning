package darks.learning.test.maxent;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;

import org.junit.Test;

import darks.learning.classifier.maxent.GISMaxent;
import darks.learning.classifier.maxent.GISModel;
import darks.learning.classifier.maxent.Maxent;
import darks.learning.classifier.maxent.MaxentModel;
import darks.learning.corpus.DocumentFilter;
import darks.learning.corpus.Documents;
import darks.learning.corpus.Documents.Document;

public class MaxentTest
{
    
    @Test
    public void testMaxentGIS()
    {
        File inputFile = new File("corpus/maxent_data3.txt");
        Documents docs;
        try
        {
        	Documents.addFilter(new DocumentFilter()
			{
				@Override
				public boolean filter(Document doc)
				{
					return "mid".equalsIgnoreCase(doc.getLabel());
				}
			});
        	docs = Documents.loadFromFile(inputFile, "UTF-8");
            Maxent maxent = new GISMaxent();
            MaxentModel model = maxent.train(docs, 1000);
            model.saveModel(new File("corpus/maxent_model.dat"));
            GISModel gisModel = GISModel.readModel(new File("corpus/maxent_model.dat"));
            maxent = new GISMaxent(gisModel);
            int count = 0;
            int totalCount = 0;
            for (Entry<String, String> entry : docs.getDocsMap().entrySet())
            {
            	if ("mid".equalsIgnoreCase(entry.getValue()))
            		continue;
                String[] terms = entry.getKey().split(" ");
                int index = maxent.predict(terms);
                String classify = maxent.getLabel(index);
                if (!classify.equals(entry.getValue()))
                {
                    System.out.println("QA:" + entry.getKey() + " output:" + classify + " expect:" + entry.getValue());
                }
                else
                {
                    count++;
                }
                totalCount++;
            }
            System.out.println("Accurancy:" + (float)count / (float)totalCount);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    
}
