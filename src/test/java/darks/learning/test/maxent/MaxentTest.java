package darks.learning.test.maxent;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;

import org.junit.Test;

import darks.learning.classifier.bayes.NaiveBayes;
import darks.learning.classifier.maxent.GISMaxent;
import darks.learning.classifier.maxent.Maxent;
import darks.learning.corpus.Documents;

public class MaxentTest
{
    
    @Test
    public void testMaxentGIS()
    {
        File input = new File("corpus/train_data.txt");
        File labels = new File("corpus/train_labels.txt");
        Documents docs;
        try
        {
            docs = Documents.loadFromFile(input, labels, "UTF-8");
            Maxent maxent = new GISMaxent();
            maxent.train(docs, 500);
            int count = 0;
            for (Entry<String, String> entry : docs.getDocsMap().entrySet())
            {
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
            }
            System.out.println("Accurancy:" + (float)count / (float)docs.getDocsMap().size());
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    
}
