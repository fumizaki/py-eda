import streamlit as st
import pandas as pd
import numpy as np
from services.eda.dataframe import EDADataFrame
from services.eda.enum import ImputeMethod, DetectOutlierMethod, TreatOutlierMethod, ScalingMethod, EncodingMethod
from services.eda.models.lightgbm.pipeline import LGBMPipeline, TaskType

from components.histogram import draw_histogram, get_histogram_instruction
from components.qqplot import draw_qqplot, get_qqplot_instruction
from components.boxplot import draw_boxplot, get_boxplot_instruction
from components.violinplot import draw_violinplot, get_violinplot_instruction
from components.frequency_table import draw_freqtable, get_freqtable_instruction
from components.scatterplot import draw_scatterplot, get_scatterplot_instruction
from components.correlation_matrix import draw_correlation_matrix, cmap_options, fmt_options, get_correlation_matrix_instruction
from components.pairplot import draw_pairplot, diagnoal_kind_options, get_pairplot_instruction
from components.barchart import draw_barchart


def render_delete_missing_value(eda: EDADataFrame) -> None:
    if len(eda.missing_columns) > 0:
        with st.expander("欠損値(削除)", expanded=False):
            delete_target = st.selectbox(
                "削除対象のカラムを選択",
                eda.missing_columns,
                key='delete_target'
            )
            if st.button("実行", key='delete_button'):    
                try:
                    eda.delete_missing_value(
                        column=delete_target,
                    )
                    st.success(f"カラム '{delete_target}' に欠損値を含む行の削除が完了しました")
                except ValueError as e:
                    st.error(f"欠損値削除エラー: {e}")
                except Exception as e:
                    st.error(f"予期しないエラーが発生しました: {e}")

    else:
        st.info("欠損値はありません")

def render_impute_missing_value(eda: EDADataFrame) -> None:
    if len(eda.missing_columns) > 0:
        with st.expander("欠損値(補完)", expanded=False):
            impute_target = st.selectbox(
                "補完対象のカラムを選択",
                eda.missing_columns,
                key='impute_target'
            )

            impute_option = st.selectbox(
                "補完オプションを選択",
                options=['静的補完', '動的補完'],
                key='impute_option'
            )

            impute_value = None
            impute_method = None
            groupby_cols = None

            if impute_option == '静的補完':
                impute_value_str = st.text_input(
                    "補完する値を入力",
                    key='impute_textinput_value'
                )
                # 入力値を適切な型に変換を試みる
                if impute_value_str:
                    try:
                        # まずintに変換を試みる
                        impute_value = int(impute_value_str)
                    except ValueError:
                        try:
                            # intでなければfloatに変換を試みる
                            impute_value = float(impute_value_str)
                        except ValueError:
                            # floatでもなければ文字列として扱う
                            impute_value = impute_value_str
                    # Check if value is None after conversion (e.g. empty string was entered)
                    if impute_value == "":
                        impute_value = None

            elif impute_option == '動的補完':
                impute_method = st.selectbox(
                    '補完方法を選択',
                    options=[
                        ImputeMethod.MEAN,
                        ImputeMethod.MEDIAN,
                        ImputeMethod.MODE,
                        ImputeMethod.FFILL,
                        ImputeMethod.BFILL
                    ],
                    format_func=lambda x: {
                        ImputeMethod.MEAN: "平均値",
                        ImputeMethod.MEDIAN: "中央値",
                        ImputeMethod.MODE: "最頻値",
                        ImputeMethod.FFILL: "前の値",
                        ImputeMethod.BFILL: "後の値"
                    }[x],
                )

                if impute_method in [ImputeMethod.MEAN, ImputeMethod.MEDIAN, ImputeMethod.MODE]:
                    available_groupby_cols = [col for col in eda.columns if col != impute_target]
                    if available_groupby_cols:
                        groupby_cols = st.multiselect(
                            "グループ化(複数, 任意)",
                            available_groupby_cols,
                            key='impute_groupby'
                        )
                        if not groupby_cols:
                            groupby_cols = None
                    else:
                        st.info("グループ化に使用できる他のカラムがありません。")


            if st.button("実行", key='impute_button'):
                if impute_method is None and impute_value is None:
                    st.warning("補完方法を選択するか、補完する値を入力してください。")
                else:
                    try:
                        # Call the imputation method
                        eda.impute_missing_value(
                            column=impute_target,
                            value=impute_value if impute_method is None else None, # Only pass value if method is None
                            method=impute_method,
                            groupby=groupby_cols
                        )
                        st.success(f"カラム '{impute_target}' の欠損値補完が完了しました（方法: {impute_method.value if impute_method else '定数'}, groupby: {groupby_cols}）。")
                    except ValueError as e:
                        st.error(f"欠損値補完エラー: {e}")
                    except Exception as e:
                        st.error(f"予期しないエラーが発生しました: {e}")

    else:
        st.info("欠損値はありません")


def render_outlier_handling(eda: EDADataFrame) -> None:

    # Only show if there are numeric columns
    if len(eda.numeric_columns) > 0:
        with st.expander("外れ値", expanded=False):
            # Column selection
            outlier_target = st.selectbox(
                "外れ値を処理するカラムを選択",
                eda.numeric_columns,
                key='outlier_target'
            )

            # Detection method selection
            detection_method = st.selectbox(
                "外れ値検出方法を選択",
                [
                    DetectOutlierMethod.IQR, 
                    DetectOutlierMethod.ZSCORE, 
                    DetectOutlierMethod.PERCENTILE
                ],
                format_func=lambda x: {
                    DetectOutlierMethod.IQR: "IQR法",
                    DetectOutlierMethod.ZSCORE: "Z-スコア法",
                    DetectOutlierMethod.PERCENTILE: "パーセンタイル法"
                }[x],
                key='detection_method'
            )

            # Threshold input based on detection method
            if detection_method == DetectOutlierMethod.IQR:
                threshold = st.slider(
                    "IQR閾値", 
                    min_value=0.5, 
                    max_value=3.0, 
                    value=1.5, 
                    step=0.1,
                    help="IQRの何倍を外れ値とみなすか",
                    key='iqr_threshold'
                )
            elif detection_method == DetectOutlierMethod.ZSCORE:
                threshold = st.slider(
                    "Z-スコア閾値", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=3.0, 
                    step=0.5,
                    help="標準偏差の何倍を外れ値とみなすか",
                    key='zscore_threshold'
                )
            else:  # PERCENTILE
                col_lower, col_upper = st.columns(2)
                with col_lower:
                    lower_percentile = st.number_input(
                        "下限(%)", 
                        min_value=0.0, 
                        max_value=49.9, 
                        value=1.0,
                        step=0.1,
                        help="この値未満を外れ値とみなす",
                        key='lower_percentile'
                    )
                with col_upper:
                    upper_percentile = st.number_input(
                        "上限(%)", 
                        min_value=50.1, 
                        max_value=100.0, 
                        value=99.0,
                        step=0.1,
                        help="この値を超える値を外れ値とみなす",
                        key='upper_percentile'
                    )
                threshold = (lower_percentile, upper_percentile)

            # Treatment method selection
            treatment_method = st.selectbox(
                "外れ値の処理方法を選択",
                [
                    TreatOutlierMethod.REMOVE, 
                    TreatOutlierMethod.CLIP, 
                    TreatOutlierMethod.REPLACE
                ],
                format_func=lambda x: {
                    TreatOutlierMethod.REMOVE: "削除",
                    TreatOutlierMethod.CLIP: "境界値に丸める",
                    TreatOutlierMethod.REPLACE: "指定値に置換"
                }[x],
                key='treatment_method'
            )

            # Replace value input (only for REPLACE method)
            replace_value = None
            if treatment_method == TreatOutlierMethod.REPLACE:
                replace_value_str = st.text_input(
                    "置換する値を入力",
                    key='replace_value'
                )
                # 入力値を適切な型に変換を試みる
                if replace_value_str:
                    try:
                        # まずintに変換を試みる
                        replace_value = int(replace_value_str)
                    except ValueError:
                        try:
                            # intでなければfloatに変換を試みる
                            replace_value = float(replace_value_str)
                        except ValueError:
                            # floatでもなければ文字列として扱う
                            replace_value = replace_value_str
                    # Check if value is None after conversion (e.g. empty string was entered)
                    if replace_value == "":
                        replace_value = None


            # Execute button
            if st.button("実行", key='outlier_handle_button'):
                try:
                    # Perform outlier handling
                    if treatment_method == TreatOutlierMethod.REPLACE:
                        if replace_value is None:
                            st.warning("置換する値を入力してください。")
                            return
                        
                        eda.handle_outlier(
                            column=outlier_target, 
                            detection_method=detection_method, 
                            threshold=threshold, 
                            treatment_method=treatment_method,
                            replace_value=replace_value
                        )
                    else:
                        eda.handle_outlier(
                            column=outlier_target, 
                            detection_method=detection_method, 
                            threshold=threshold, 
                            treatment_method=treatment_method
                        )
                    
                    # Success message
                    st.success(f"カラム '{outlier_target}' の外れ値処理が完了しました。")
                
                except ValueError as e:
                    st.error(f"外れ値処理エラー: {e}")
                except Exception as e:
                    st.error(f"予期しないエラーが発生しました: {e}")

    else:
        st.info("数値型のカラムがありません")


def render_encoding(eda: EDADataFrame) -> None:
    """
    カテゴリカルデータのエンコーディングのためのUIを描画する
    
    Args:
        eda (EDADataFrame): 処理対象のEDADataFrameインスタンス
    """
    # カテゴリカル列がある場合のみ表示
    if len(eda.categorical_columns) > 0:
        with st.expander("エンコーディング", expanded=False):
            
            # エンコードするカラムの複数選択
            encoding_target = st.selectbox(
                "エンコードするカラムを選択",
                eda.categorical_columns,
                key='encoding_target'
            )
            
            # エンコーディング方法の選択
            encoding_method = st.selectbox(
                "エンコーディング方法を選択",
                [
                    EncodingMethod.LABEL,
                    EncodingMethod.ONEHOT,
                    EncodingMethod.ORDINAL,
                    EncodingMethod.BINARY
                ],
                format_func=lambda x: {
                    EncodingMethod.LABEL: "ラベルエンコーディング",
                    EncodingMethod.ONEHOT: "OneHotエンコーディング",
                    EncodingMethod.ORDINAL: "オーディナルエンコーディング",
                    EncodingMethod.BINARY: "バイナリエンコーディング"
                }[x],
                key='encoding_method'
            )
            
            # 追加オプション
            drop_original = st.checkbox("元のカラムを削除", key='drop_original')
            
            
            # OneHotエンコーディングの場合のプレフィックスオプション
            if encoding_method == EncodingMethod.ONEHOT:
                prefix = st.text_input(
                    "カラム名のプレフィックス (空白の場合は元の列名が使用されます)",
                    key='onehot_prefix'
                )
                if prefix == "":
                    prefix = None
            else:
                prefix = None
            
            # 実行ボタン
            if st.button("実行", key='encoding_execute_button'):
                if not encoding_target:
                    st.warning("エンコードするカラムを少なくとも1つ選択してください。")
                else:
                    try:
                        with st.spinner("エンコーディング処理中..."):
                            eda.encode_column(
                                column=encoding_target,
                                method=encoding_method,
                                drop_original=drop_original,
                                prefix=prefix
                            )
                            
                        # 成功メッセージ
                        st.success(f"選択されたカラムのエンコーディングが完了しました: {', '.join(encoding_target)}")
                    
                    except ValueError as e:
                        st.error(f"エンコーディング処理エラー: {e}")
                    except Exception as e:
                        st.error(f"予期しないエラーが発生しました: {e}")



def render_scaling(eda: EDADataFrame) -> None:
    """
    Render a Streamlit UI for scaling numeric columns
    
    Args:
        eda (EDADataFrame): The EDA DataFrame instance to process
    """
    # Only show if there are numeric columns
    if len(eda.numeric_columns) > 0:
        with st.expander("スケーリング", expanded=False):
            
            # Multi-select for columns to scale
            scaling_columns = st.multiselect(
                "スケーリングするカラムを選択",
                eda.numeric_columns,
                default=eda.numeric_columns,  # Default to all numeric columns
                key='scaling_columns'
            )
        
            # Scaling method selection
            scaling_method = st.selectbox(
                "スケーリング方法を選択",
                [
                    ScalingMethod.STANDARD, 
                    ScalingMethod.MINMAX, 
                    ScalingMethod.ROBUST, 
                    ScalingMethod.MAXABS
                ],
                format_func=lambda x: {
                    ScalingMethod.STANDARD: "標準化 (Z-スコア)",
                    ScalingMethod.MINMAX: "最小-最大スケーリング",
                    ScalingMethod.ROBUST: "ロバストスケーリング",
                    ScalingMethod.MAXABS: "最大絶対値スケーリング"
                }[x],
                key='scaling_method'
            )
            
            # Additional options based on scaling method
            if scaling_method == ScalingMethod.MINMAX:
                st.markdown("#### 最小-最大スケーリングオプション")
                col_min, col_max = st.columns(2)
                with col_min:
                    min_range = st.number_input(
                        "最小値", 
                        value=0.0, 
                        step=0.1,
                        key='minmax_min'
                    )
                with col_max:
                    max_range = st.number_input(
                        "最大値", 
                        value=1.0, 
                        step=0.1,
                        key='minmax_max'
                    )
            else:
                # For other methods, use default values
                min_range, max_range = 0, 1
            
            
            # Execute button
            if st.button("実行", key='scaling_execute_button'):
                try:
                    # Perform actual scaling
                    if scaling_method == ScalingMethod.MINMAX:
                        eda.scale_columns(
                            columns=scaling_columns, 
                            method=scaling_method,
                            scale_range=(min_range, max_range)
                        )
                    else:
                        eda.scale_columns(
                            columns=scaling_columns, 
                            method=scaling_method
                        )
                    
                    # Success message
                    st.success(f"選択されたカラムのスケーリングが完了しました: {', '.join(scaling_columns)}")
                
                except ValueError as e:
                    st.error(f"スケーリング処理エラー: {e}")
                except Exception as e:
                    st.error(f"予期しないエラーが発生しました: {e}")

    else:
        st.info("数値型のカラムがありません")


def render_preprosessing(eda: EDADataFrame) -> None:
    st.markdown("### 前処理")
    # 欠損値
    render_delete_missing_value(eda)
    render_impute_missing_value(eda)
    # 外れ値
    render_outlier_handling(eda)
    # エンコーディング
    render_encoding(eda)
    # スケーリング
    render_scaling(eda)
    st.markdown("#### 履歴")
    st.dataframe(eda.history, hide_index=True)
    
    


def render_sidebar_dataset_options(eda: EDADataFrame) -> None:
    with st.sidebar:
        st.header("データセットのオプション")
        # キーを設定して状態を維持
        upload_file = st.file_uploader("CSVファイルをアップロード", type=["csv"], key="sidebar_csv_uploader")

        # ファイルがアップロードされた場合
        if upload_file is not None:
             # ボタンを追加して明示的にロードをトリガー
             if st.button("アップロードしたCSVをロード", key="sidebar_load_uploaded_csv"):
                 try:
                      # セッションステートのedaオブジェクトに対してロード
                      eda.load_from_csv(upload_file)
                      st.success("CSVファイルをロードしました。")
                      st.rerun() # ロード成功したら再実行
                 except Exception as e:
                      st.error(f"CSVファイルのロードに失敗しました: {e}")

        else:
            st.write('OR')
            # キーを設定して状態を維持
            dataset_name = st.selectbox(
                "データセットを選択",
                options=[""] + eda.dataset_options(), # 初期値として空文字列またはNoneを追加
                index=0,
                key="sidebar_dataset_select"
            )
            # データセットの選択とロードボタン
            if dataset_name and dataset_name != "": # 何か選択されているか確認
                if st.button(f"{dataset_name} をロード", key="sidebar_load_selected_dataset"):
                    try:
                        # セッションステートのedaオブジェクトに対してロード
                        eda.load_from_option(dataset_name)
                        st.success(f"データセット '{dataset_name}' をロードしました。")
                        st.rerun() # ロード成功したら再実行
                    except Exception as e:
                        st.error(f"データセットのロードに失敗しました: {e}")

        st.markdown("---")
        if eda.df is not None and not eda.df.empty:
            render_preprosessing(eda)
            



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
    st.write("- カラム名:", f"`{', '.join(eda.missing_columns)}`" if len(eda.missing_columns) > 0 else "なし")
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
        lgbm = LGBMPipeline()

        lgbm_task = st.selectbox(
            "タスク種別",
            [
                TaskType.BINARY,
                TaskType.MULTICLASS,
                TaskType.REGRESSION,    
            ],
            format_func=lambda x: {
                TaskType.BINARY: "二値分類",
                TaskType.MULTICLASS: "多クラス分類",
                TaskType.REGRESSION: "回帰"
            }[x],
            index=0,
            key="lgbm_task_type"
        )

        if len(eda.numeric_columns) > 0:
            lgbm_features = st.multiselect(
                "特徴量カラム",
                options=eda.numeric_columns,
                default=eda.numeric_columns,
                key="lgbm_features"
            )
        else:
            lgbm_features = []
            st.warning("LightGBMの特徴量に使える数値カラムがありません。")

        default_target_index = 0 if len(eda.categorical_columns) > 0 else None
        lgbm_target = st.selectbox("ターゲットカラム", eda.columns, index=default_target_index, key="lgbm_target")

        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05)
        early_stopping = st.number_input("Early Stopping Rounds", value=10, min_value=1)

        if lgbm_target and lgbm_features:
            if st.button("モデル学習 & 評価", key="lgbm_train"):
                try:
                    lgbm.prepare_data(
                        task=lgbm_task,
                        df=eda.df,
                        features=lgbm_features,
                        target=lgbm_target,
                        test_size=0.3
                    )
                    
                    with st.spinner("モデルを学習しています..."):
                        lgbm.fit(early_stopping_rounds=early_stopping)
                        eval_df = lgbm.evaluate()
                        feature_importances = lgbm.feature_importances()
                        y_pred = lgbm.predict(eda.df[lgbm_features][0:lgbm.y_test.shape[0]])
                        # クラス予測ラベルに変換
                        if lgbm.task == TaskType.MULTICLASS:
                            y_pred_label = np.argmax(y_pred, axis=1)
                        else:
                            y_pred_label = y_pred  # BINARY や REGRESSION の場合はそのまま

                        # 比較用DataFrameを作成
                        pred_df = pd.DataFrame({
                            "予測値": y_pred_label,
                            "実際の値": lgbm.y_test.reset_index(drop=True)
                        })

                    st.subheader("モデル評価結果")
                    st.dataframe(eval_df.style.format("{:.3f}"))

                    st.subheader("予測結果")
                    st.dataframe(pred_df.style.format("{:.3f}"))

                    if feature_importances is not None and not feature_importances.empty:
                        st.subheader("特徴量重要度 (Feature Importances)")
                        draw_barchart(
                            df=feature_importances,
                            x_col="importance",
                            y_col="feature",
                            orient="v",
                            title="Feature Importances",
                        )
                    elif feature_importances is None:
                        st.info("特徴量重要度を取得できませんでした。")

                except Exception as e:
                    st.error(f"LightGBMの処理中にエラーが発生しました: {e}")

            if st.button("交差検証の実行", key="lgbm_cv"):
                try:
                    lgbm.prepare_data(
                        task=lgbm_task,
                        df=eda.df,
                        features=lgbm_features,
                        target=lgbm_target,
                        test_size=test_size
                    )

                    with st.spinner("交差検証を実行中..."):
                        cv_result = lgbm.cross_validate(cv=5)

                    st.subheader("交差検証結果")
                    st.dataframe(cv_result.style.format("{:.3f}"))

                except Exception as e:
                    st.error(f"交差検証中にエラーが発生しました: {e}")

        elif not lgbm_features:
            st.warning("LightGBMを実行するには特徴量カラムを選択してください。")
        elif not lgbm_target:
            st.warning("LightGBMを実行するにはターゲットカラムを選択してください。")

    else:
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。LightGBMを使用するにはデータが必要です。")


def render_page() -> None:
    st.set_page_config(layout="centered", page_title="PyEDA")
    st.title('PyEDA')
    
    # Streamlitのセッションステートを利用してEDADataFrameインスタンスを保持
    # スクリプトの再実行時も、このインスタンスは維持される
    if 'eda' not in st.session_state:
        st.session_state.eda = EDADataFrame()

    # edaオブジェクトを取得
    eda: EDADataFrame = st.session_state.eda

    render_sidebar_dataset_options(eda)

    if eda.df is not None and not eda.df.empty:
        tab1, tab2 = st.tabs(['Dataset', 'LightGBM'])
        with tab1:
            render_dataset_tab(eda)
        with tab2:
            render_lgbm_tab(eda)
    else:
        st.info("EDAを開始するには、サイドバーからデータセットを選択するか、CSVファイルをアップロードしてください。")


if __name__ == '__main__':
    render_page()