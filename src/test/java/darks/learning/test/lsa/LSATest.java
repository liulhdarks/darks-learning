package darks.learning.test.lsa;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import org.junit.Assert;
import org.junit.Test;

import darks.learning.corpus.Corpus;
import darks.learning.corpus.CorpusLoader;
import darks.learning.lsa.LatentSemanticAnalysis;

public class LSATest
{
 
    @Test
    public void testLSA()
    {
        try
        {
            CorpusLoader loader = new CorpusLoader(Corpus.TYPE_TF_IDF);
            File file = new File("corpus/corpus_ali_train.txt");
            Corpus corpus = loader.loadFromReader(new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8")));
            //Corpus corpus = loader.loadCorpus(new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8")));
            Assert.assertNotNull(corpus);
            LatentSemanticAnalysis lsa = new LatentSemanticAnalysis();
            lsa.train(corpus);
            System.out.println(lsa.predict("无法 打开 出售 中 宝贝 详情 页面".split(" ")));
            System.out.println(lsa.predict("刷新 订单 数量 还是 显示 错乱 旺旺 聊天 查看 买家 信誉 不了 以前 老 版本 没有 这种 问题".split(" ")));
            System.out.println(lsa.predict("怎么 你们 千牛 软件 电脑 版 弹不出 聊天 窗口".split(" ")));
            System.out.println(lsa.predict("为什么 手机 登陆 旺旺 聊天记录 其它 客服 接待 现实 号".split(" ")));
            System.out.println(lsa.predict("电脑 登陆 时候 说 数据 不能 保存 系统盘 只有 一个盘 没有 分区".split(" ")));
            System.out.println(lsa.predict("收不到 系统 提示信息 还有 设置 信息 声音 提示 怎么 会 没有".split(" ")));
            System.out.println(lsa.predict("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事".split(" ")));
            System.out.println(lsa.predict("千 牛 电脑 端 怎么 登陆 打开 聊天 窗口 一直 响应 黑 然后 怎么回事 求 回应 已经 反馈 几次 现在 不了 点 开 就是".split(" ")));
            System.out.println(corpus.getTfIDF().getSentenceMap().size());
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        
    }
    
}
