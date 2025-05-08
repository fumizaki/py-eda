# PyEDA(app)

データ分析と可視化のためのStreamlitアプリケーション。様々な種類のグラフ描画や組み込みデータセットの探索が可能です。

## Features

このアプリケーションは、以下の機能を提供します。

-   複数の組み込みデータセット（Iris, Titanic, Boston Housing, Wineなど）のロードと表示。
-   ロードしたデータセットの概要統計量や情報表示。
-   以下の様々な種類のグラフや分析結果の描画:
    -   棒グラフ (Bar Chart)
    -   箱ひげ図 (Box Plot)
    -   相関行列 (Correlation Matrix)
    -   カウントプロット (Count Plot)
    -   度数分布表 (Frequency Table)
    -   ヒストグラム (Histogram)
    -   折れ線グラフ (Line Plot)
    -   ペアプロット (Pair Plot)
    -   並行座標プロット (Parallel Coordinates Plot)
    -   QQ プロット (QQ Plot)
    -   散布図 (Scatter Plot)
    -   バイオリンプロット (Violin Plot)

## Directory Structure

```
.
├── components/   # 各種グラフ描画やUIコンポーネントのモジュール
│   ├── [graph_type]/ # 例: barchart, boxplot など
│   │   ├── draw.py      # グラフ描画ロジック
│   │   └── instruction.py # コンポーネントの説明など
│   └── ...
├── datasets/     # アプリケーションで使用する組み込みデータセット (CSV形式)
│   ├── BostonHousing.csv
│   ├── Iris.csv
│   ├── Titanic.csv
│   └── Wine.csv
├── services/     # データ処理や分析関連のサービスモジュール
│   ├── dataset/    # データセットロード関連
│   └── eda/        # 探索的データ分析関連
├── main.py       # Streamlit アプリケーションのエントリポイント
├── pyproject.toml # 依存関係管理 (Poetry または uv/pip)
└── uv.lock      # このファイル
```