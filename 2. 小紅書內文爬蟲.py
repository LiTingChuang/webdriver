from datetime import date
import json
from DrissionPage import ChromiumPage
from DataRecorder import Recorder
from tqdm import tqdm
import time

def sign_in():
    sign_in_page = ChromiumPage()
    sign_in_page.get("https://www.xiaohongshu.com")
    print("请扫码登录")
    # 第一次运行需要扫码登录
    time.sleep(20)

def read_urls_from_txt(path):
    with open(path, "r") as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

def open_url(url):
    global page
    page = ChromiumPage()
    # page.set.load_mode.eager()
    page.get(f"{url}")
    time.sleep(10)

def get_note_page_info(url):
    time.sleep(15)
    # 访问url
    open_url(url)
    time.sleep(15)
    page.set.NoneElement_value("0")
    container = page.ele(".outer-link-container")
    sections = container.eles(".note-container")
    for section in sections:
        footer = section.ele(".author-container", timeout=0)
        auther_info = footer.ele(".author-wrapper", timeout=0)
        auther_name = auther_info.ele(".name", timeout=0).text

        note_info = section.ele(".note-scroller", timeout=0)
        note_content = note_info.ele(".note-content",timeout=0)
        note_title = note_content.ele(".title", timeout=0).text
        note_desc = note_content.ele(".desc", timeout=0).text

        date = note_info.ele(".bottom-container", timeout=0).text

        count_info = section.ele(".interact-container", timeout=0)
        rlike_count = count_info.ele(".like-wrapper like-active",timeout=0).text
        if 'w' in rlike_count and '.' in rlike_count :
                Nlikes = rlike_count.replace('w', '') + '000'
                like_count = Nlikes.replace('.', '')
        elif 'w' in rlike_count :
                like_count = rlike_count.replace('w', '') + '000'
        else:
                like_count = rlike_count

        rcollect_count = count_info.ele(".collect-wrapper",timeout=0).text
        if 'w' in rcollect_count and '.' in rcollect_count :
                Nlikes = rcollect_count.replace('w', '') + '000'
                collect_count = Nlikes.replace('.', '')
        elif 'w' in rcollect_count :
                collect_count = rcollect_count.replace('w', '') + '000'
        else:
                collect_count = rcollect_count

        rchat_count = count_info.ele(".chat-wrapper",timeout=0).text
        if 'w' in rchat_count and '.' in rchat_count :
                Nlikes = rchat_count.replace('w', '') + '000'
                chat_count = Nlikes.replace('.', '')
        elif 'w' in rchat_count :
                chat_count = rchat_count.replace('w', '') + '000'
        else:
                chat_count = rchat_count

        
        

        return auther_name, note_title, note_desc, date, like_count, collect_count, chat_count
        



if __name__ == "__main__":
    # 第1次运行需要登录，后面不用登录，可以注释掉
    sign_in()

    # 新建一个excel表格，用来保存数据
    r = Recorder(path="詳細內容.xlsx", cache_size=20)

    # 设置要采集的笔记链接
    # 多篇小红书笔记的url地址放在txt文件里，每行放1个url
    note_urls_file_path = "要蒐集的連結.txt"

    # 从txt文件读取urls
    note_urls = read_urls_from_txt(note_urls_file_path)
    new_note_contents_dict = {}
    all_note_contents=[]
    for note_url in tqdm(note_urls):
        # 采集笔记详情，返回一个note_contents字典
        note_contents = get_note_page_info(note_url)
        all_note_contents.append(note_contents)

        # 将note_contents字典转换为字符串
        note_contents = json.dumps(note_contents, separators=(",", ":"), ensure_ascii=False)
        print(type(note_contents), "笔记详情：", note_contents)


    # 获取当前日期
    current_date = date.today()

    # 保存excel文件
    r = Recorder('詳細內容.csv')
    #数据写入缓存
    for i in range(len(all_note_contents)):
        print(all_note_contents[i])
        new_note_contents_dict = { 
                "作者": all_note_contents[i][0], 
                "標題": all_note_contents[i][1], 
                "內文": all_note_contents[i][2], 
                "發佈時間": all_note_contents[i][3], 
                "按讚數": all_note_contents[i][4], 
                "收藏數": all_note_contents[i][5], 
                "討論數": all_note_contents[i][6],
                }
        r.add_data(new_note_contents_dict)

    r.record(f"{current_date}-詳細內容.csv")
    print('儲存結束')

    