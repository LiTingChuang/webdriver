#使用jieba分析中文字
import jieba
import jieba.analyse

#匯入文字雲模組
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from scipy.ndimage import gaussian_gradient_magnitude

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
import re
import jieba.posseg as psg


#讀入資料
txtfile = "情緒分類-積極.txt" # 剛才下載存的文字檔 

Text = open(txtfile,"r",encoding="utf-8").read()

#設定jieba繁體中文字典
#file_path = 'cn_stopwords.txt'
#jieba.set_dictionary(file_path)

#進行斷詞
seg_list = jieba.cut(Text, cut_all=True, HMM=True) # 預設模式。cut_all=True- 全切模式 切的很碎

#傳回list結構
seg_list = jieba.lcut(Text, cut_all=True ,HMM=True)
print (seg_list)


#抽取關鍵詞
jieba.analyse.extract_tags(Text, topK=10, withWeight=False, allowPOS=())

#利用分析出來的結果，利運pandas做字數頻率統計
import pandas as pd
def count_segment_freq(seg_list):
  seg_df = pd.DataFrame(seg_list,columns=['seg'])
  seg_df['count'] = 1
  sef_freq = seg_df.groupby('seg')['count'].sum().sort_values(ascending=False)
  sef_freq = pd.DataFrame(sef_freq)
  return sef_freq

sef_freq = count_segment_freq(seg_list)
sef_freq.head()
print (sef_freq)

#設定中文字型，沒設置會出現亂碼
#font_path = 'TaipeiSansTCBeta-Regular.ttf' # 
font_path = 'NotoSansSC-VariableFont_wght.ttf' # 字型路徑

seg_lists=' '.join(seg_list) #做成list

with open('jieba_stopWords.txt','r', encoding='utf8') as sw:
    stopword = [line.strip() for line in sw.readlines()]

#文字雲繪製參數設定
wc = WordCloud(
  background_color='white',        #   背景顏色
  max_words=200,                   #   最大分詞數量
  #mask=mask_image,                       #   背景圖片
  max_font_size=None,              #   顯示字體的最大值
  font_path=font_path,             #   若為中文則需引入中文字型(.TTF)
  random_state=None,               #   隨機碼生成各分詞顏色
  prefer_horizontal=0.9,             #   調整分詞中水平和垂直的比例
  stopwords = stopword,
  width=800,
  height=600,
  margin=5,
)
# 生成詞雲
wc.generate(seg_lists)

#利用wordcloud內的ImageColorGenerator()上色
#image_colors = ImageColorGenerator(mask_color)
#wc.recolor(color_func=image_colors) #依照原圖上色

#重要詞頻結果儲存
file_name='重要詞頻結果儲存.txt'
#file_name='pts_ptscontent2_clean_cloud-1_result.txt'
with open(file_name, 'w',encoding='utf-8') as f: 
    for key, value in wc.words_.items(): 
        f.write('%s:%s\n' % (key, value))

#繪製成圖形

#plt.imshow(wc,interpolation="bilinear")
#plt.axis('off') #關掉座標
#plt.show() #展示圖片
wc.to_file("第一種類.jpg")
#plt.clf()

print('開始第二部分')

def data_process(str_data):
    # 去除换行、空格、网址、参考文献
    data_after_process =  re.sub(r'\n+', '', str_data)
    data_after_process =  re.sub(r'\s+', '', data_after_process)
    data_after_process =  re.sub(r'[a-zA-z]+://[^\s]*', '', data_after_process)
    data_after_process =  re.sub(r'\[([^\[\]]+)\]', '', data_after_process)

    # 删除日期：YY/MM/DD YY-MM-DD YY年MM月DD日
    data_after_process = re.sub(r'\d{4}-\d{1,2}-\d{1,2}','',data_after_process)
    data_after_process = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', data_after_process)

    # 删除标点符号
    punctuation = """＂!！?？＃＄％＆＇()（）＊＋－／：；＜＝＞＠［＼］＾＿｀●｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛*°▽〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    re_punctuation = "[{}]+".format(punctuation)
    data_after_process =  re.sub(re_punctuation, '', data_after_process)

    return data_after_process

def getText(filename):
    fp = open(r'./' + filename, 'r', encoding='utf-8')
    sentences = fp.readlines()
    fp.close()
    for i in range(len(sentences)):
        sentences[i] = data_process(sentences[i])
    sentences = " ".join(sentences).split('。')
    return sentences


def getWordFrequency(sentences):
    print('計算詞頻')
    words_dict = {}
    for text in sentences:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        # 结巴分词
        wordGen = psg.cut(text)
        # 结巴分词的输出结果为一个生成器，把生成器转换为 list
        for word, attr in wordGen:
            if attr in ['n', 'nr', 'nz']:
                if word in words_dict.keys():
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1
    return words_dict


def getTFIDF(sentences):
    print('計算權重')
    corpus = []
    for text in sentences:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        # 结巴分词
        wordGen = psg.cut(text)
        # 结巴分词的输出结果为一个生成器,把生成器转换为list
        cut_list = []
        for word, attr in wordGen:
            if attr in ['n', 'nr', 'nz']:
                cut_list.append(word)
        corpus.append(" ".join(cut_list))

    words_dict = {}
    # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    vectorizer = CountVectorizer()
    # 计算每个词语的tf-idf权值  
    transformer = TfidfTransformer()
    # 将文本转为词频矩阵
    matrix = vectorizer.fit_transform(corpus)
    # 计算tf-idf
    tfidf = transformer.fit_transform(matrix)
    # 获取词袋模型中的所有词语  
    word = vectorizer.get_feature_names_out()
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重 
    weight = tfidf.toarray() 
    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重  
    for i in range(len(weight)):
        for j in range(len(word)):  
            if word[j] in words_dict:
                words_dict[word[j]] += weight[i][j]
            else:
                words_dict[word[j]] = weight[i][j]
    return words_dict

def showCloud(wordDict, filename):
    print('開始繪圖')
    # 根据图片创建 graph 为 nd-array 类型
    #image = Image.open("./pikachu.jpg")
    #graph = np.array(image)
    # 创建 wordcloud 对象，背景图片为 graph，背景色为白色
    wc = WordCloud(
        background_color='white',
        max_words=200,                 
        max_font_size=None,          
        font_path=font_path,          
        random_state=None,        
        prefer_horizontal=0.9,          
        stopwords = stopword,
        width=800,
        height=600,
        margin=5,
        )
    # 生成词云
    wc.generate_from_frequencies(wordDict)
    # 根据 graph 生成颜色
    # image_color = ImageColorGenerator(graph)
    #plt.imshow(wc,interpolation="bilinear") #对词云重新着色
    #plt.axis('off')
    # 显示词云图，并保存为 jpg 文件
    #plt.show()
    wc.to_file("第二種類.jpg")
    #plt.clf()

showCloud(getWordFrequency(getText(txtfile)), "xiamen")
showCloud(getTFIDF(getText(txtfile)), "xiamen_tfidf")
print('繪圖結束')

