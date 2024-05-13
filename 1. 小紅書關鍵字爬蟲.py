from datetime import date
from DrissionPage import ChromiumPage
from urllib.parse import quote
import pandas as pd
from tqdm import tqdm
import time
import random

def sign_in():
    sign_in_page = ChromiumPage()
    sign_in_page.get('https://www.xiaohongshu.com')
    print("請掃碼登入")
    # 第一次執行需要掃碼登錄
    time.sleep(20)

def search(keyword):
    global page
    page = ChromiumPage()
    page.get(f'https://www.xiaohongshu.com/search_result?keyword={keyword}&source=web_search_result_notes')

def get_info():
        time.sleep(5)
        # 定位包含筆記資訊的sections
        page.set.NoneElement_value('0')
        container = page.ele('.feeds-container')
        sections = container.eles('.note-item')
        for section in sections:
            # 定位文章鏈接
            note_link = section.ele('.cover ld mask', timeout=0).link
            if note_link is not None:
                print(note_link)
                pass
            # 定位標題、作者、按讚
            footer = section.ele('.footer', timeout=0)
            if footer is not None:
                pass
            rtitle = footer.ele('.title', timeout=0).text
            if rtitle is not None:
                title = rtitle.strip()
                print(title)
            else:
                title = rtitle
            # 定位作者
            author_wrapper = footer.ele('.author-wrapper')
            if author_wrapper is not None:
                pass
            author = author_wrapper.ele('@class=name').text
            if author is not None:
                print(author)
                pass
            # 定位作者主頁網址
            author_link = author_wrapper.ele('tag:a', timeout=0).link
            if author_link is not None:
                pass
            # 定位作者頭像
            author_img = author_wrapper.ele('tag:img', timeout=0).link
            if author_img is not None:
                pass
            # 定位按讚 
            like = footer.ele('.like-wrapper like-active').text
            if 'w' in like and '.' in like :
                Nlikes = like.replace('w', '') + '000'
                nlikes = Nlikes.replace('.', '')
                likes = int(nlikes)
            elif 'w' in like :
                Nlikes = like.replace('w', '') + '000'
                likes = int(Nlikes)
            else:
                likes = int(like)
                print(likes)
            # print(likes)         
            # contents列表用来存放所有爬取到的信息
            contents.append([title, author, note_link, likes])
            #contents.append([title, author, note_link, author_link, author_img, likes])


def page_scroll_down():
    print("********下滑頁********")
    # 產生一個隨機時間
    random_time = random.uniform(0.5, 1.5)
    # 暫停
    time.sleep(random_time)
    # time.sleep(1)
    # page.scroll.down(5000)
    page.scroll.to_bottom()

def craw(times):
    for i in tqdm(range(1, times + 1)):
        get_info()
        page_scroll_down()

def save_to_excel(contents,excel_path):
    print("開始儲存資料")
    name = ['title', 'author', 'note_link', 'likes']
    #name = ['title', 'author', 'note_link', 'author_link', 'author_img', 'likes']
    df = pd.DataFrame(columns=name, data=contents)
    #df.dropna(how = 'all')
    #Signature: df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df['likes'] = df['likes'].astype(int)
    # 删除重复行
    df = df.drop_duplicates()
    # 按点赞 降序排序
    df = df.sort_values(by='likes', ascending=False)
    df.to_csv(excel_path, na_rep='NA')
    print("已儲存",df)

# def auto_resize_column(excel_path):
#     print("開始調整格式")
#     """自適應列寬度"""
#     wb = openpyxl.load_workbook(excel_path)
#     worksheet = wb.active
#     # 循環遍歷工作表中的1-2列
#     for col in worksheet.iter_cols(min_col=1, max_col=2):
#         max_length = 0
#         # 列名稱
#         column = col[0].column_letter
#         # 迴圈遍歷列中的所有儲存格
#         for cell in col:
#             try:
#                 # 如果目前儲存格的值長度大於max_length，則更新 max_length 的值
#                 if len(str(cell.value)) > max_length:
#                     max_length = len(str(cell.value))
#             except:
#                 pass
#         # 計算調整後的列寬度
#         adjusted_width = (max_length + 2) * 2
#         # 使用 worksheet.column_dimensions 屬性設定列寬度
#         worksheet.column_dimensions[column].width = adjusted_width

#         # 循環遍歷工作表中的3-5列
#         for col in worksheet.iter_cols(min_col=3, max_col=5):
#             max_length = 0
#             column = col[0].column_letter # Get the column name

#             # 使用 worksheet.column_dimensions 屬性設定列寬度
#             worksheet.column_dimensions[column].width = 15

#     wb.save(excel_path)
    
#     print("已調整完成",excel_path)




if __name__ == '__main__':
    # contents列表用來存放所有爬取到的信息
    contents = []

    # 搜尋關鍵字: 美妝、彩妝、makeup、化妝
    keyword = "化妝"
    # 設定向下翻頁爬取次數
    times = 50

    # 第1次執行需要登錄，後面不用登錄，可以註解掉
    sign_in()

    # 關鍵字轉為 url 編碼
    keyword_temp_code = quote(keyword.encode('utf-8'))
    keyword_encode = quote(keyword_temp_code.encode('gb2312'))

    # 根據關鍵字搜尋小紅書文章
    search(keyword_encode)

    # 根據設定的次數，開始爬取數據
    craw(times)
    
    # 爬到的資料儲存到本機excel文件
    excel_path = 'Ｍakeup4Data.csv'
    save_to_excel(contents, excel_path)
    # auto_resize_column(excel_path)