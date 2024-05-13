import csv
from datetime import date
from DataRecorder import Recorder
from tqdm import tqdm


current_date = date.today()
# 保存excel文件
r = Recorder('情緒分析分數.csv')

rows = []
with open("情緒分析-2024-05-03_1.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)



odata_words = []
data_points = []
data_scopes = []
for i in tqdm(range(len(rows))):
        odata_words.append(rows[i][0])
        data_points.append(rows[i][1])
        if float(rows[i][1]) >= 0.8 and float(rows[i][1]) <= 1 :
                data_scopes.append('非常積極')
        elif float(rows[i][1]) < 0.8 and float(rows[i][1]) >= 0.6 :
                data_scopes.append('積極')
        elif float(rows[i][1]) < 0.6 and float(rows[i][1]) >= 0.4 :
                data_scopes.append('普通')
        elif float(rows[i][1]) < 0.4 and float(rows[i][1]) >= 0.2 :
                data_scopes.append('消極')
        else :
                data_scopes.append('非常消極')
data_words=[]
with open('jieba_stopWords.txt','r', encoding='utf8') as sw:
    stopwords = [line.strip() for line in sw.readlines()]
for i in range(len(odata_words)):
       if odata_words[i] in stopwords:
            pass
       else:
            data_words.append(i)
print (data_words)
       

for i in tqdm(range(len(data_words))):
        new_senana = { 
                    "單詞": data_words[i], 
                    "情緒分數": data_points[i],
                    "情緒等級": data_scopes[i]
                    }
        r.add_data(new_senana)
r.record(f"{current_date}-情緒分析分數.csv")
print('儲存結束')