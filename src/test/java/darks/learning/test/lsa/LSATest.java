package darks.learning.test.lsa;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import darks.learning.corpus.Corpus;
import darks.learning.corpus.CorpusLoader;
import darks.learning.dimreduce.lsa.LatentSemanticAnalysis;

public class LSATest
{
 
    @Test
    public void testLSA()
    {
        try
        {
            CorpusLoader loader = new CorpusLoader(Corpus.TYPE_TF_IDF);
            File file = new File("corpus/corpus_ali.txt");
            Corpus corpus = loader.loadFromFile(file, "UTF-8");
            Assert.assertNotNull(corpus);
            LatentSemanticAnalysis lsa = new LatentSemanticAnalysis();
            lsa.train(corpus);
            lsa.saveModel(new File("test/lsa.model"));
            lsa.loadModel(new File("test/lsa.model"));
            System.out.println(lsa.predict("无法 打开 出售 中 宝贝 详情 页面".split(" ")));
            System.out.println(lsa.predict("刷新 订单 数量 还是 显示 错乱 旺旺 聊天 查看 买家 信誉 不了 以前 老 版本 没有 这种 问题".split(" ")));
            System.out.println(lsa.predict("怎么 你们 千牛 软件 电脑 版 弹不出 聊天 窗口".split(" ")));
            System.out.println(lsa.predict("为什么 手机 登陆 旺旺 聊天记录 其它 客服 接待 现实 号".split(" ")));
            System.out.println(lsa.predict("电脑 登陆 时候 说 数据 不能 保存 系统盘 只有 一个盘 没有 分区".split(" ")));
            System.out.println(lsa.predict("收不到 系统 提示信息 还有 设置 信息 声音 提示 怎么 会 没有".split(" ")));
            System.out.println(lsa.predict("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事".split(" ")));
            System.out.println(lsa.predict("一家 显示 温馨 七度 店铺 解决 串号 旺旺 怎么回事 ".split(" ")));
            System.out.println(lsa.predict("商品信息 跳到 商品 管理".split(" ")));
            System.out.println(lsa.predict("升级 ios8 量子 显示 左边".split(" ")));
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        
    }

    @Test
    public void testEvalLsa()
    {
        try
        {
	    	CorpusLoader loader = new CorpusLoader(Corpus.TYPE_TF_IDF);
	        File file = new File("corpus/corpus_ali.txt");
	        Corpus corpus = loader.loadFromReader(new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8")));
	        Assert.assertNotNull(corpus);
	        LatentSemanticAnalysis lsa = new LatentSemanticAnalysis();
	        lsa.train(corpus);
	        
	        int corpusCount = (int)corpus.getTfIDF().getTotalSentenceCount();
	        System.out.println(corpusCount);
//	        DoubleMatrix labelsOne = DoubleMatrix.ones(corpusCount);
//	        DoubleMatrix labels = DoubleMatrix.diag(labelsOne);
//	        labelsOne = null;
	        int rightCount = 0;
	        for (Entry<Integer, String> entry : lsa.getSentenceColumnIndexs().entrySet())
	        {
	        	int index = entry.getKey();
	        	String s = entry.getValue();
	        	int preIndex = lsa.predictIndex(s.split(" "));
	        	if (preIndex != index)
	        	{
	        		System.out.println("source:" + s);
	        		System.out.println("target:" + lsa.getSentenceColumnIndexs().get(preIndex));
	        	}
	        	else
	        	{
	        		rightCount++;
	        	}
	        }
	        System.out.println((float) rightCount / (float) corpusCount);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        
    }
    
}
