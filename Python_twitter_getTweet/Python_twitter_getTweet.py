from xml.etree.ElementTree import tostring
from requests_oauthlib import OAuth1Session
import datetime
import json

### input
## from your developer acount
AK = 'xx'
AS = 'xx'
AT = 'xx'
ATS = 'xx'

## twitter ID to find
TW_ID = 'xxxxxxxx'

## num of tweet to get
numTweet = 100
###

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
## output file name
outfileName = 'tweet_ID' + str(TW_ID) + '_' + '{:%Y%m%d%H%M%S}'.format(now) + '.txt'

twitter = OAuth1Session(AK, AS, AT, ATS)

url = "https://api.twitter.com/2/users/" + TW_ID + "/tweets"

params = {
  'expansions'  : 'author_id',
  'tweet.fields': 'created_at,public_metrics',
  'user.fields' : 'name',
  'max_results' : numTweet,
  }

res = twitter.get(url, params = params)

print(res.content)

tl = json.loads(res.text)

print(f"name : {tl['includes']['users'][0]['name']}")
print(f"user : {tl['includes']['users'][0]['username']}")
print('----------------------------')

f = open(outfileName, 'w', encoding='UTF-8')

for l in tl['data']:
  print(l['text'])
  print(l['created_at'])
  print('----------------------------')

  f.write(l['text'])

f.close()