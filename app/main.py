import streamlit as st
from services.eda.dataframe import EDADataFrame
from services.eda.models.lightgbm.model import LightGBMModel

from components.histogram import draw_histogram, get_histogram_instruction
from components.qqplot import draw_qqplot, get_qqplot_instruction
from components.boxplot import draw_boxplot, get_boxplot_instruction
from components.violinplot import draw_violinplot, get_violinplot_instruction
from components.frequency_table import draw_freqtable, get_freqtable_instruction
from components.scatterplot import draw_scatterplot, get_scatterplot_instruction
from components.correlation_matrix import draw_correlation_matrix, cmap_options, fmt_options, get_correlation_matrix_instruction
from components.pairplot import draw_pairplot, diagnoal_kind_options, get_pairplot_instruction
from components.barchart import draw_barchart

st.title('PyEDA')

def render_sidebar_dataset_options(eda: EDADataFrame) -> None:
    with st.sidebar:
        st.header("データセットのオプション")
        upload_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        if upload_file:
            eda.load_from_csv(upload_file)
        else:
            st.write('OR')
            dataset_name = st.selectbox(
                "データセットを選択",
                options=eda.dataset_options(),
                index=0,
            )
            # データセットの読み込み
            eda.load_from_option(dataset_name)


def render_dataset_summary(eda: EDADataFrame) -> None:
    st.markdown("## データセット")
    st.markdown("### プレビュー")
    st.dataframe(eda.df, hide_index=True)

    st.markdown("### データ形状")
    col1, col2 = st.columns(2)
    rows = len(eda.df)
    cols = len(eda.columns)
    col1.metric("行数", f"{len(eda.df):,}")
    col2.metric("列数", f"{len(eda.columns):,}")

    st.markdown("### データ型")

    st.markdown("#### 数値型")
    st.write("- カラム数:", len(eda.numeric_columns))
    st.write("- カラム名:", f"`{', '.join(eda.numeric_columns)}`" if len(eda.numeric_columns) > 0 else "なし")
    st.markdown("#### カテゴリ型")
    st.write("- カラム数:", len(eda.categorical_columns))
    st.write("- カラム名:", f"`{', '.join(eda.categorical_columns)}`" if len(eda.categorical_columns) > 0 else "なし")
    st.markdown("#### その他")
    st.write("- カラム数:", len(eda.other_columns))
    st.write("- カラム名:", f"`{', '.join(eda.other_columns)}`" if len(eda.other_columns) > 0 else "なし")

    st.markdown("### 欠損値")
    st.write("- カラム数:", len(eda.missing_columns))
    st.write("- カラム名:", f"`{', '.join(eda.missing_columns)}`" if eda.missing_columns else "なし")
    total_missing_percentage = (eda.df.isnull().sum().sum() / (rows * cols)) * 100
    st.write("- 合計欠損率:", f"{total_missing_percentage:.2f}%")

    st.markdown("### 重複")
    st.write("- カラム数:", len(eda.duplicated_columns))
    st.write("- カラム名:", f"`{', '.join(eda.duplicated_columns)}`" if eda.duplicated_columns else "なし")

    st.markdown("### ユニーク")
    st.write("- カラム数:", len(eda.unique_columns))
    st.write("- カラム名:", f"`{', '.join(eda.unique_columns)}`" if eda.unique_columns else "なし")


    st.markdown("### 基本統計量")
    df_stats = eda.stats(include='all')
    st.dataframe(df_stats.T)
    st.write("""
        **数値変数:**
        - `mean` (平均) と `50%` (中央値) の差が大きい場合、分布が偏っている可能性があります。
        - `std` (標準偏差) が大きい、または `min`/`max` が他の四分位数から大きく離れている場合、スケールの違いや外れ値の可能性があります。

        **カテゴリカル変数:**
        - `unique` (ユニークな値の数) が多いカテゴリ変数は、取り扱いに注意が必要な場合があります。
        - `top` (最頻値) と `freq` (その頻度) から、カテゴリの偏りを確認できます。
        """)


def render_histogram_section(eda: EDADataFrame) -> None:
    st.markdown("#### ヒストグラム")
    st.write(get_histogram_instruction())
    if eda.numeric_columns:
        col = st.selectbox("カラム", eda.numeric_columns, index=0, key="histgram_col")
        with st.expander("カスタマイズ", expanded=False):
            title = st.text_input("タイトル", f"Histogram({col})", key="histgram_title")
            bins = st.slider("区間", 1, 100, 30, key="histgram_bins")
        draw_histogram(
            df=eda.df,
            col=col,
            title=title,
            bins=bins,
        )
    else:
        st.info("数値型のカラムがありません。")


def render_qqplot_section(eda: EDADataFrame) -> None:
    st.markdown("#### QQプロット")
    st.write(get_qqplot_instruction())
    if eda.numeric_columns:
        # カラムの選択
        col = st.selectbox("カラム", eda.numeric_columns, index=0, key="qqplot_col")
        # カスタマイズオプション
        with st.expander("カスタマイズ", expanded=False):
            title = st.text_input("タイトル", f"QQ Plot({col})", key="qqplot_title")

        # 描画
        draw_qqplot(
            df=eda.df,
            col=col,
            title=title,
        )
    else:
        st.info("数値型のカラムがありません。")


def render_violinplot_section(eda: EDADataFrame) -> None:
    st.markdown("#### バイオリンプロット")
    st.write(get_violinplot_instruction())
    if eda.numeric_columns:
        col = st.selectbox("カラム", eda.numeric_columns, index=0, key="violinplot_col")
        with st.expander("カスタマイズ", expanded=False):
            title = st.text_input("タイトル", f"Violin Plot({col})", key="violineplot_title")
        draw_violinplot(
            df=eda.df,
            y_col=col,
            title=title,
        )
    else:
        st.info("数値型のカラムがありません。")


def render_boxplot_section(eda: EDADataFrame) -> None:
    st.markdown("#### 箱ひげ図")
    st.write(get_boxplot_instruction())
    if eda.numeric_columns:
        col = st.selectbox("カラム", eda.numeric_columns, index=0, key="boxplot_col")
        with st.expander("カスタマイズ", expanded=False):
            title = st.text_input("タイトル", f"Box Plot({col})", key="boxplot_title")
        draw_boxplot(
            df=eda.df,
            y_col=col,
            title=title,
        )
    else:
        st.info("数値型のカラムがありません。")


def render_freqtable_section(eda: EDADataFrame) -> None:
    st.markdown("#### 度数分布表")
    st.write(get_freqtable_instruction())
    if eda.categorical_columns:
        col = st.selectbox("カラム", eda.categorical_columns, index=0, key="freqtable_col")
        with st.expander("カスタマイズ", expanded=False):
            title = st.text_input("タイトル", f"Frequency Table({col})", key="freqtable_title")
        draw_freqtable(
            df=eda.df,
            col=col,
            title=title
        )
    else:
        st.info("カテゴリ型のカラムがありません。")
    

def render_scatterplot_section(eda: EDADataFrame) -> None:
    st.markdown("#### 散布図")
    st.write(get_scatterplot_instruction())
    if len(eda.numeric_columns) >= 2:
        col1, col2 = st.columns(2)

        # X軸とY軸の選択 (数値カラムから)
        x_col = col1.selectbox("X軸", eda.numeric_columns, index=0, key="scatter_x")
        # Y軸のデフォルトインデックスをX軸と異なるように調整
        default_y_index = 1 if len(eda.numeric_columns) > 1 and eda.numeric_columns[0] != eda.numeric_columns[1] else 0
        if x_col == eda.numeric_columns[default_y_index]: # X軸とY軸のデフォルトが同じ場合はずらす
             default_y_index = (default_y_index + 1) % len(eda.numeric_columns) if len(eda.numeric_columns) > 1 else 0

        y_col = col2.selectbox("Y軸", eda.numeric_columns, index=default_y_index, key="scatter_y")

        # カスタマイズオプション
        with st.expander("カスタマイズ(散布図)", expanded=False):
            title = st.text_input("タイトル", f"Scatter Plot({x_col}-{y_col})", key="scatter_title")
            alpha = st.slider("透過度", 0.1, 1.0, 0.7, key="scatter_alpha")

            color_option = st.checkbox("色分け", value=True, key="scatter_color_option")
            color_col = None
            if color_option and len(eda.columns) > 0: # 色分け対象のカラムが存在する場合
                 # 全てのカラムから選択可能（カテゴリカルデータも含む）
                 # デフォルトで最後のカラムを選択 (カテゴリカルが多いと仮定)
                 default_color_index = len(eda.columns)-1 if len(eda.columns) > 0 else 0
                 color_col = st.selectbox("カラム(色)", eda.columns, index=default_color_index, key="scatter_color")
            elif color_option and len(eda.columns) == 0:
                 st.warning("色分けに使えるカラムがありません。")


            size_option = st.checkbox("サイズ分け", value=False, key="scatter_size_option")
            size_col = None
            if size_option and len(eda.numeric_columns) > 0: # サイズ分け対象のカラムが存在する場合
                 size_col = st.selectbox("カラム(サイズ)", eda.numeric_columns, key="scatter_size")
            elif size_option and len(eda.numeric_columns) == 0:
                 st.warning("サイズ分けに使える数値カラムがありません。")

        # 描画
        draw_scatterplot(
            df=eda.df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            size_col=size_col,
            title=title,
            alpha=alpha
        )
    else:
        st.info("散布図を描画するには２つ以上の数値カラムが必要です。")


def render_corrmatrix_section(eda: EDADataFrame) -> None:
    st.markdown("#### 相関行列")
    st.write(get_correlation_matrix_instruction())

    if len(eda.numeric_columns) >= 2:
        heatmap_cols = st.multiselect(
            "カラム",
            options=eda.numeric_columns,
            default=eda.numeric_columns,
            key="heatmap_cols"
        )

        if len(heatmap_cols) >= 2:
            with st.expander("カスタマイズ", expanded=False):
                cmap = st.selectbox(
                    "カラーマップ",
                    cmap_options(),
                    index=0,
                    key="heatmap_cmap"
                )
                annot = st.checkbox("データ表示", value=True, key="heatmap_annot")
                fmt = st.selectbox(
                    "フォーマット",
                    fmt_options(),
                    index=0,
                    key="heatmap_fmt"
                )

            # 描画
            correlation_df = eda.get_correlation_matrix(heatmap_cols)
            draw_correlation_matrix(
                df=correlation_df,
                cmap=cmap,
                annot=annot,
                fmt=fmt
            )
            # 高い相関を持つペアの表示
            st.markdown("##### 高い相関を持つペア")
            st.dataframe(eda.get_high_correlations(heatmap_cols), hide_index=True)

        else:
            st.warning("相関行列には２つ以上のカラムが必要です。")
    else:
        st.info("相関行列を描画するには２つ以上の数値カラムが必要です。")


def render_pairplot_section(eda: EDADataFrame) -> None:
    st.markdown("#### ペアプロット")
    st.write(get_pairplot_instruction())

    if len(eda.numeric_columns) >= 2:
        pairplot_cols = st.multiselect(
            "カラム(ペアプロット)",
            options=eda.numeric_columns,
            default=eda.numeric_columns,
            key="pairplot_cols"
        )

        if len(pairplot_cols) >= 2:
            with st.expander("カスタマイズ(ペアプロット)", expanded=False):
                title = st.text_input("タイトル", "Pair Plot", key="pairplot_title")
                # hue に使えるのはカテゴリカルカラムまたは None
                hue_options = [None] + eda.categorical_columns # None オプションを追加
                hue = st.selectbox("色分け", hue_options, index=0, key="pairplot_hue")

                diag_kind_opts = diagnoal_kind_options()
                diag_kind = st.selectbox(
                    "分布形状",
                    diag_kind_opts,
                    index=diag_kind_opts.index("auto") if "auto" in diag_kind_opts else 0, # デフォルトを 'auto' に設定 (あれば)
                    key="pairplot_diag_kind"
                )

            # 描画
            draw_pairplot(
                df=eda.df,
                columns=pairplot_cols,
                hue=hue,
                diag_kind=diag_kind,
                title=title,
            )
        else:
            st.warning("ペアプロットには２つ以上のカラムが必要です。")
    else:
        st.info("ペアプロットを描画するには２つ以上の数値カラムが必要です。")


def render_numeric_variable_analysis(eda: EDADataFrame) -> None:


    tab1, tab2, tab3, tab4 = st.tabs(['ヒストグラム', 'QQプロット', 'バイオリンプロット', '箱ひげ図'])

    with tab1:
        render_histogram_section(eda)
    with tab2:
        render_qqplot_section(eda)
    with tab3:
        render_violinplot_section(eda)
    with tab4:
        render_boxplot_section(eda)


def render_categorical_variable_analysis(eda: EDADataFrame) -> None:
    render_freqtable_section(eda)


def render_multivariate_analysis(eda: EDADataFrame) -> None:
    tab1, tab2, tab3 = st.tabs(['散布図', '相関行列', 'ペアプロット'])

    with tab1:
        render_scatterplot_section(eda)
    with tab2:
        render_corrmatrix_section(eda)
    with tab3:
        render_pairplot_section(eda)


def render_variable_analysis(eda: EDADataFrame) -> None:
    """
    変数特性分析セクションをレンダリングする。
    """
    st.markdown("## 変数特性")

    st.markdown("### 数値変数")    
    render_numeric_variable_analysis(eda)

    st.markdown("### カテゴリ変数")
    render_categorical_variable_analysis(eda)

    st.markdown("### 多変数分析")
    render_multivariate_analysis(eda)


def render_dataset_tab(eda: EDADataFrame) -> None:
    """
    Dataset タブの内容をレンダリングする。
    """
    if eda.df is not None and not eda.df.empty:
        render_dataset_summary(eda)
        st.markdown("---") # 区切り線
        render_variable_analysis(eda)
    else:
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。")


def render_lgbm_tab(eda: EDADataFrame) -> None:
    st.markdown("## LightGBM")
    if eda.df is not None and not eda.df.empty:
        lgbm = LightGBMModel()

        lgbm_task = st.selectbox("タスク種別", lgbm.task_options(), index=0, key="lgbm_task_type")

        if len(eda.numeric_columns) > 0:
             default_lgbm_features = eda.numeric_columns
             lgbm_features = st.multiselect(
                     "特徴量カラム",
                     options=eda.numeric_columns,
                     default=default_lgbm_features,
                     key="lgbm_features"
                 )
        else:
             lgbm_features = []
             st.warning("LightGBMの特徴量に使える数値カラムがありません。")

        if len(eda.categorical_columns) > 0:
             # デフォルトのターゲットインデックスを、可能な範囲で0に設定
             default_target_index = 0 if len(eda.categorical_columns) > 0 else None
             lgbm_target = st.selectbox("ターゲットカラム", eda.categorical_columns, index=default_target_index, key="lgbm_target")
        else:
             lgbm_target = None
             st.warning("LightGBMのターゲットに使えるカテゴリカラムがありません。")


        if lgbm_target and lgbm_features: # ターゲットと特徴量が選択されていれば実行可能
             if st.button("モデル学習 & 評価", key="lgbm_train"):
                 try:
                    lgbm.load(lgbm_task, eda.df, lgbm_features, lgbm_target, test_size=0.3) # test_size も引数で調整可能にしても良い

                    # ---- 学習・評価 ----
                    with st.spinner("モデルを学習しています..."):
                         lgbm.train(early_stopping_rounds=10) # early_stopping_rounds も引数で調整可能にしても良い
                         eval_df = lgbm.evaluate()
                         feature_importances = lgbm.get_feature_importances()

                    # ---- 評価結果表示 ----
                    st.subheader("モデル評価結果 (テストデータ)")
                    st.dataframe(eval_df.style.format("{:.3f}"))

                    # ---- 特徴量重要度 ----
                    if feature_importances is not None and not feature_importances.empty:
                         st.subheader("特徴量重要度 (Feature Importances)")

                         draw_barchart(
                            df=feature_importances,
                            x_col="Importance",
                            y_col="Feature",
                            orient="v",
                            title="Feature Importances",
                        )

                    elif feature_importances is None:
                         st.info("特徴量重要度を取得できませんでした。")


                 except Exception as e:
                     st.error(f"LightGBMの処理中にエラーが発生しました: {e}")
        elif not lgbm_features:
            st.warning("LightGBMを実行するには特徴量カラムを選択してください。")
        elif not lgbm_target:
            st.warning("LightGBMを実行するにはターゲットカラムを選択してください。")

    else:
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。LightGBMを使用するにはデータが必要です。")

def render_page() -> None:
    eda = EDADataFrame()

    render_sidebar_dataset_options(eda)

    if eda.df is not None and not eda.df.empty:
        tab1, tab2 = st.tabs(['Dataset', 'LightGBM'])
        with tab1:
            render_dataset_tab(eda)
        with tab2:
            render_lgbm_tab(eda)
    else:
        st.info("アプリケーションを開始するには、サイドバーからデータセットを選択するか、CSVファイルをアップロードしてください。")


if __name__ == '__main__':
    render_page()