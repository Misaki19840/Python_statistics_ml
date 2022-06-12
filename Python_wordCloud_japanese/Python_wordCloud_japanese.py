from wordcloud import WordCloud
import numpy as np
from PIL import Image
import MeCab
import unidic

# read text
text_org_path = "tweet_ID87027640_20220611181929.txt"
text_org = open("./" + text_org_path, encoding="utf8").read()

tagger = MeCab.Tagger()
tagger.parse('')
node = tagger.parseToNode(text_org)

word_list = []
while node:
    word_type = node.feature.split(',')[0]
    if word_type == '名詞':
        word_list.append(node.surface)
    node = node.next
word_chain = ' '.join(word_list)

# read Mask Img
imgMask_path = "hir_mask.jpg"
imgMask = np.array(Image.open( imgMask_path ).convert('L'))
imgMask[imgMask>=128] = 128
imgMask[imgMask<=10] = 255
imgMask[imgMask==128] = 0

wordcloud = WordCloud(width=640, height=480, background_color='white', mask=imgMask, font_path='C:\Windows\Fonts\yumin.ttf').generate(word_chain)
wordcloud.to_file("./" + "WordCloudImg_" + text_org_path + ".png")