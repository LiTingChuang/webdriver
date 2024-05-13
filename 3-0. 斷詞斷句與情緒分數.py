from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from cemotion import Cemotion
from DataRecorder import Recorder
from datetime import date
from tqdm import tqdm
import csv


# Initialize drivers
#斷詞
ws_driver  = CkipWordSegmenter(model="bert-base")
#詞性標記
pos_driver = CkipPosTagger(model="bert-base")
#實體辨識
#ner_driver = CkipNerChunker(model="bert-base")

# Use CPU
ws_driver = CkipWordSegmenter(device=-1)

rows = []
with open("詳細內容All.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)


note_info = []
for i in range(len(rows)):
    note_info.append(rows[i][1])
    note_info.append(rows[i][2])
#print(note_info)


# Run pipeline
ws  = ws_driver(note_info)
pos = pos_driver(ws)
#ner = ner_driver(note_info)

c = Cemotion()

all_sentences = []
all_analyzed = []
all_points = []

with open('jieba_stopWords.txt','r', encoding='utf8') as sw:
    stopwords = [line.strip() for line in sw.readlines()]

# Pack word segmentation and part-of-speech results
def pack_ws_pos_sentece(sentence_ws, sentence_pos):
   assert len(sentence_ws) == len(sentence_pos)
   res = []
   for word_ws, word_pos in zip(sentence_ws, sentence_pos):
      if word_ws in stopwords:
            pass
      else:
                res.append(f"{word_ws}({word_pos})")
   return "\u3000".join(res)



# Show results

for sentence, sentence_ws, sentence_pos in zip(note_info, ws, pos):
#    #print('原句：　',sentence)
#    #print('分詞後詞性標注：　',pack_ws_pos_sentece(sentence_ws, sentence_pos))
#    # for entity in sentence_ner:
#    #    print('entity',entity)
#    # print()
#    # print(sentence_ws)
        for i in range(len(sentence_ws)):
            #print(f'"{sentence_ws}"\n预测值:{c.predict(sentence_ws)}\n')
                    all_sentences.append(sentence_ws[i])
                    all_analyzed.append(c.predict(sentence_ws[i]))
        print('第 ', i, ' 筆資料已分析完成，仍在分析中，下一筆為第 ', i+1)
print('all_sentences: ',all_sentences)
print('all_analyzed: ',all_analyzed)


current_date = date.today()
# 保存excel文件
r = Recorder('情緒分析.csv')
#数据写入缓存
for i in range(len(all_sentences)):
        new_senana = { 
                "單詞": all_sentences[i], 
                "情緒分數": all_analyzed[i],
                }
        r.add_data(new_senana)

r.record(f"{current_date}-情緒分析.csv")
print('儲存結束')
