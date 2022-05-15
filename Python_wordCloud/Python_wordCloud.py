from wordcloud import WordCloud
 
fname = "the_great_dictator_speech.txt"
 
text = open("./" + fname, encoding="utf8").read()
wordcloud = WordCloud(max_font_size=40).generate(text)
wordcloud.to_file("./" + "WordCloudImg_" + fname + ".png")