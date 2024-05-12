# しゃべくりミリオネア

2024年5月12日　大阪24時間AIハッカソンの作品 <br>
過去の偉人の名言から名言度を判定します。
<br>
### 1.機械学習<br>
learning.py で 偉人名言データセットquotes.csv を機械学習<br>
TensorFlow 利用 <br>
以下を同じディレクトリに生成<br>
- great_person_quotes_model.h5 （偉人名言モデル）
- tokenizer.pickle

### 2.フロント　（FLASK）
- app.py を実行
- ブラウザで以下を開く <br>
  http://localhost:5000/predict <br>
  ※index.htmlはtemplateの中にあります

テキストを入力すると、その名言度を判定してくれます。<br>

しかし、現状何を入力しても同じ判定度になり、うまくいっておりません。


