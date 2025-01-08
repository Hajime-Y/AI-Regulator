# AI-Regulator

AI-Regulatorは、金融機関の規定改定作業を自動化・効率化するためのAIツールです。LLMを活用して、既存の規定文書と変更情報を分析し、新しい規定案を生成します。

## 謝辞
このプロロェクトは、[SakanaAI](https://github.com/SakanaAI)の[AI Scientist](https://github.com/SakanaAI/AI-Scientist)プロジェクトを参考に作成されました。
革新的なアイデアと実装に深く感謝いたします。

## 主な機能
- 規定文書の分析（規定一覧目次を利用した改定候補の選出・個別確認）
- LLMによる規定改定案の作成
- 規定改定案の品質レビューと改善作業
- 改定レポートの生成（MD/PDF形式）

## 利用方法

### 1. リポジトリのクローンとセットアップ
まず、リポジトリをクローンし、必要なパッケージをインストールします：

```
# clone
git clone https://github.com/Hajime-Y/AI-Regulator.git
cd AI-Regulator

# PDFの生成に必要なパッケージのインストール
sudo apt-get install wkhtmltopdf fonts-noto-cjk

# Pythonパッケージのインストール
pip install -r requirements.txt
```

### 2. OpenAI APIキーの設定
環境変数にOpenAI APIキーを設定します：

```
export OPENAI_API_KEY="YOUR KEY HERE"
```

### 3. 規定改定の実行
以下のコマンドで規定改定処理を実行します：

```
python launch_regulator.py \
    --regulations-dir ./regulations/external_regulations/ \
    --base-dir ./template/投資信託規定集改定/ \
    --model gpt-4o-2024-05-13 \
    --num-reflections 3 \
    --draft \
    --review \
    --improvement
```

## regulations/templateフォルダについて

### regulations
- 三菱UFJ銀行の外部向け[規定一覧](https://www.bk.mufg.jp/regulation/index.html)をPDF化して保存しています
- templateフォルダの改定情報を反映する前の状態の規定が格納されています

### template
- 2024年の規定改定情報を含むフォルダです
- 三菱UFJ銀行の[お知らせ](https://www.bk.mufg.jp/info/sonota.html)から、主要な改定情報をピックアップしています