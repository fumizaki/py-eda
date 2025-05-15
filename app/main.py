import streamlit as st
import pandas as pd
import numpy as np
from services.eda.dataframe import EDADataFrame
from services.eda.enum import ImputeMethod, DetectOutlierMethod, TreatOutlierMethod, ScalingMethod, EncodingMethod
from services.eda.models.lightgbm.pipeline import LGBMPipeline, LGBMTask
from services.eda.models.mlp.pipeline import MLPPipeline, MLPTask
from services.eda.models.randomforest.pipeline import RandomForestPipeline, RFTask
from services.eda.models.logisticregression.pipeline import LogisticRegressionPipeline, LogiRegTask
from services.eda.models.linearregression.pipeline import LinearRegressionPipeline, LinRegTask
from services.eda.models.svm.pipeline import SVMPipeline, SVMTask


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
                        st.success(f"選択されたカラムのエンコーディングが完了しました: {encoding_target}")
                    
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
                LGBMTask.BINARY,
                LGBMTask.MULTICLASS,
                LGBMTask.REGRESSION,    
            ],
            format_func=lambda x: {
                LGBMTask.BINARY: "二値分類",
                LGBMTask.MULTICLASS: "多クラス分類",
                LGBMTask.REGRESSION: "回帰"
            }[x],
            index=0,
            key="lgbm_task"
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
            tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])
            with tab1:
                st.write("LightGBMモデルを学習させ、テストデータで評価を行います。")
                if st.button("実行", key="lgbm_train"):
                    try:
                        lgbm.load_task(lgbm_task)
                        lgbm.load_data(
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
                            if lgbm.task == LGBMTask.MULTICLASS:
                                y_pred_label = np.argmax(y_pred, axis=1)
                            else:
                                y_pred_label = y_pred  # BINARY や REGRESSION の場合はそのまま

                            # 比較用DataFrameを作成
                            pred_df = pd.DataFrame({
                                "予測値": y_pred_label,
                                "実際の値": lgbm.y_test.reset_index(drop=True)
                            })

                        st.subheader("モデル評価結果")
                        st.write("精度や再現率などの評価指標を確認できます。")
                        st.dataframe(eval_df.style.format("{:.3f}"))

                        st.subheader("予測結果")
                        st.write("テストデータに対する予測値と実測値の比較です。")
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

            with tab2:
                st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="lgbm_cv_folds_slider")
                if st.button("実行", key="lgbm_cv"):
                    try:
                        lgbm.load_task(lgbm_task)
                        lgbm.load_data(
                            df=eda.df,
                            features=lgbm_features,
                            target=lgbm_target,
                        )

                        with st.spinner("交差検証を実行中..."):
                            cv_result = lgbm.cross_validate(cv = cv_folds)

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

def render_mlp_tab(eda: EDADataFrame) -> None:
    st.markdown("## MLP") # MLPタブのヘッダー

    # データセットがロードされているかを確認
    if eda.df is not None and not eda.df.empty:
        # MLPPipeline インスタンスを作成
        # UIで選択されたタスク種別は後ほど load_task() で設定します
        mlp = MLPPipeline()

        # --- タスク、特徴量、ターゲットのカラム選択UI ---
        mlp_task = st.selectbox(
            "タスク種別",
            [
                MLPTask.BINARY,
                MLPTask.MULTICLASS,
                MLPTask.REGRESSION,
            ],
            format_func=lambda x: {
                MLPTask.BINARY: "二値分類",
                MLPTask.MULTICLASS: "多クラス分類",
                MLPTask.REGRESSION: "回帰"
            }[x],
            index=0,
            key="mlp_task_select" # ユニークなキーを設定
        )

        # 特徴量カラムの選択 (MLPは通常スケーリングされた数値データに適しています)
        # LightGBMの例に倣い、デフォルトで数値カラムを選択肢とします
        available_features = eda.numeric_columns
        if not available_features:
             st.warning("MLPの特徴量として使用できる数値カラムがデータセットにありません。")
             mlp_features = [] # 数値カラムがない場合は特徴量を空リストにする
        else:
            mlp_features = st.multiselect(
                "特徴量カラム",
                options=available_features,
                default=available_features, # デフォルトで全ての数値カラムを選択
                key="mlp_features_multiselect" # ユニークなキーを設定
            )

        # ターゲットカラムの選択 (分類ならエンコードされた数値、回帰なら数値)
        # LightGBMの例に倣い、全てのカラムを選択肢とします
        available_targets = eda.columns
        default_target_index = 0 # デフォルトのインデックス

        # 分類タスクでカテゴリカルカラムが存在する場合、最初のカテゴリカルをデフォルトに試みる
        if mlp_task in [MLPTask.BINARY, MLPTask.MULTICLASS] and eda.categorical_columns:
             try:
                 default_target_index = available_targets.index(eda.categorical_columns[0])
             except ValueError:
                 # 見つからなければ0のまま
                 pass
        # 回帰タスクで数値カラムが存在する場合、最初の数値をデフォルトに試みる
        elif mlp_task == MLPTask.REGRESSION and eda.numeric_columns:
             try:
                  default_target_index = available_targets.index(eda.numeric_columns[0])
             except ValueError:
                  # 見つからなければ0のまま
                  pass

        # 選択肢が空でないことを確認し、デフォルトインデックスを調整
        if not available_targets:
             mlp_target = None
             st.warning("ターゲットカラムとして選択できるカラムがデータセットにありません。")
        else:
            # デフォルトインデックスが範囲内であることを保証
            default_target_index = min(default_target_index, len(available_targets) - 1) if available_targets else 0
            mlp_target = st.selectbox(
                 "ターゲットカラム",
                 options=available_targets,
                 index=default_target_index,
                 key="mlp_target_selectbox" # ユニークなキーを設定
             )


        # データ分割割合のスライダー
        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05, key="mlp_test_size_slider") # ユニークなキーを設定


        


        # --- 特徴量とターゲットが選択されているか確認 ---
        if mlp_target and mlp_features:
            # --- モデル学習/評価 と 交差検証 のためのタブ ---
            tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])

            with tab1:
                st.write("MLPモデルを学習させ、テストデータで評価を行います。")
                if st.button("実行", key="mlp_train_button"): # ユニークなキーを設定
                    try:
                        # --- UIで選択されたパラメータを使ってMLPパイプラインを更新 ---
                        mlp.load_task(mlp_task)
                        # 更新されたパラメータでモデルを再初期化
                        mlp.model = mlp.get_model()

                        # データをロードし、学習用・テスト用に分割
                        mlp.load_data(
                            df=eda.df,
                            features=mlp_features,
                            target=mlp_target,
                            test_size=test_size,
                        )

                        # --- モデルを学習 ---
                        with st.spinner("モデルを学習しています..."):
                            # mlp.fit() はパラメータを受け取らないように設計されています
                            # パラメータはモデル初期化時に渡されています
                            mlp.fit()

                        # --- モデルを評価 ---
                        eval_df = mlp.evaluate()

                        # --- 予測結果の取得と表示（テストデータに対するもの） ---
                        # テストデータに対する予測値または予測確率を取得
                        y_pred_proba_or_value = mlp.predict(mlp.X_test)

                        # タスク種別に応じて、比較用の予測ラベル/値に変換
                        if mlp.task == MLPTask.MULTICLASS:
                            # 多クラス分類の場合、確率の中から最も高いクラスのインデックスを取得
                            y_pred_label_or_value = np.argmax(y_pred_proba_or_value, axis=1)
                        elif mlp.task == MLPTask.BINARY:
                             # 二値分類の場合、model.predict()で直接クラスラベルを取得
                             y_pred_label_or_value = mlp.model.predict(mlp.X_test)
                        else: # Regression (回帰)
                            # 回帰の場合、predict()の結果がそのまま予測値
                            y_pred_label_or_value = y_pred_proba_or_value

                        # テストデータの実測値を取得し、インデックスをリセットして予測値と揃える
                        actual_values = mlp.y_test.reset_index(drop=True)

                        # 予測結果と実測値の比較DataFrameを作成
                        pred_df = pd.DataFrame({
                            "予測値": y_pred_label_or_value,
                            "実際の値": actual_values
                        })


                        # --- 結果を表示 ---
                        st.subheader("モデル評価結果")
                        st.write("テストデータに対する精度や再現率などの評価指標を確認できます。")
                        st.dataframe(eval_df.style.format("{:.3f}"))

                        st.subheader("予測結果")
                        st.write("テストデータに対する予測値と実測値の比較です。")
                        # データが多い場合は先頭の一部のみ表示
                        if pred_df.shape[0] > 100:
                             st.dataframe(pred_df.head(100).style.format("{:.3f}"))
                             st.write(f"... 他 {pred_df.shape[0] - 100} 件を表示していません。")
                        else:
                            st.dataframe(pred_df.style.format("{:.3f}"))

                        # MLPは標準では特徴量重要度を提供しないため、表示部分はありません。

                    except Exception as e:
                        st.error(f"MLPの学習または評価中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示（開発時推奨）
                        import traceback
                        st.error(traceback.format_exc())


            with tab2:
                st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="mlp_cv_folds_slider") # ユニークなキーを設定

                if st.button("実行", key="mlp_cv_button"): # ユニークなキーを設定
                    try:
                        # --- UIで選択されたパラメータを使ってMLPパイプラインを更新 (学習タブと同じ処理) ---
                        mlp.load_task(mlp_task)
                        # データをロード (特徴量/ターゲットの設定のために必要。CV自体はdf全体を使う)
                        # test_sizeの値はCVの分割には直接影響しませんが、load_dataの引数として必要です。
                        mlp.load_data(
                            df=eda.df,
                            features=mlp_features,
                            target=mlp_target,
                        )

                        # --- 交差検証を実行 ---
                        with st.spinner(f"{cv_folds}分割交差検証を実行中..."):
                            cv_result = mlp.cross_validate(cv=cv_folds)

                        # --- 結果を表示 ---
                        st.subheader("交差検証結果")
                        st.dataframe(cv_result.style.format("{:.3f}"))

                    except Exception as e:
                        st.error(f"MLPの交差検証中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示
                        import traceback
                        st.error(traceback.format_exc())


        # --- 特徴量やターゲットが選択されていない場合の警告 ---
        elif not mlp_features:
            st.warning("MLPモデルの学習には特徴量カラムが必要です。")
        elif mlp_target is None:
            st.warning("MLPモデルの学習にはターゲットカラムが必要です。")

    else:
        # データセットがロードされていない場合の情報メッセージ
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。MLPモデルを使用するにはデータが必要です。")


def render_rf_tab(eda: EDADataFrame) -> None:
    """
    Random Forestモデルの学習、評価、交差検証、特徴量重要度のためのStreamlit UIを描画します。
    パラメータ設定UIは含まれず、デフォルトパラメータが使用されます。

    Parameters
    ----------
    eda : EDADataFrame
        処理対象のEDADataFrameインスタンス。
    """
    st.markdown("## Random Forest") # Random Forestタブのヘッダー

    # データセットがロードされているかを確認
    if eda.df is not None and not eda.df.empty:
        # RandomForestPipeline インスタンスを作成
        # UIで選択されたタスク種別は後ほど load_task() で設定します
        rf = RandomForestPipeline()

        # --- タスク、特徴量、ターゲットのカラム選択UI ---
        # RFTask Enum を使用
        rf_task = st.selectbox(
            "タスク種別",
            [
                RFTask.CLASSIFICATION,
                RFTask.REGRESSION,
            ],
            format_func=lambda x: {
                RFTask.CLASSIFICATION: "分類",
                RFTask.REGRESSION: "回帰"
            }[x],
            index=0,
            key="rf_task_select" # ユニークなキーを設定
        )

        # 特徴量カラムの選択
        available_features = eda.numeric_columns
        if not available_features:
             st.warning("Random Forestの特徴量として使用できる数値またはエンコード済みカラムがデータセットにありません。")
             rf_features = [] # 数値カラムがない場合は特徴量を空リストにする
        else:
            rf_features = st.multiselect(
                "特徴量カラム",
                options=available_features,
                default=available_features, # デフォルトで全ての利用可能なカラムを選択
                key="rf_features_multiselect" # ユニークなキーを設定
            )

        # ターゲットカラムの選択
        available_targets = eda.columns
        default_target_index = 0 # デフォルトのインデックス

        # 分類タスクでカテゴリカルカラムが存在する場合、最初のカテゴリカルをデフォルトに試みる
        if rf_task == RFTask.CLASSIFICATION and eda.categorical_columns:
             try:
                 default_target_index = available_targets.index(eda.categorical_columns[0])
             except ValueError:
                 pass # 見つからなければ0のまま
        # 回帰タスクで数値カラムが存在する場合、最初の数値をデフォルトに試みる
        elif rf_task == RFTask.REGRESSION and eda.numeric_columns:
             try:
                  default_target_index = available_targets.index(eda.numeric_columns[0])
             except ValueError:
                  pass # 見つからなければ0のまま

        # 選択肢が空でないことを確認し、デフォルトインデックスを調整
        if not available_targets:
             rf_target = None
             st.warning("ターゲットカラムとして選択できるカラムがデータセットにありません。")
        else:
            default_target_index = min(default_target_index, len(available_targets) - 1) if available_targets else 0
            rf_target = st.selectbox(
                 "ターゲットカラム",
                 options=available_targets,
                 index=default_target_index,
                 key="rf_target_selectbox" # ユニークなキーを設定
             )

        # データ分割割合のスライダー
        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05, key="rf_test_size_slider") # ユニークなキーを設定


        # --- 特徴量とターゲットが選択されているか確認 ---
        if rf_target and rf_features:
            # --- モデル学習/評価 と 交差検証 のためのタブ ---
            tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])

            with tab1:
                st.write("Random Forestモデルをデフォルトパラメータで学習させ、テストデータで評価を行います。")
                if st.button("実行", key="rf_train_button"): # ユニークなキーを設定
                    try:
                        # --- Random Forestパイプラインの準備 ---
                        # まずタスクを設定 (これにより、内部でデフォルトパラメータがロードされます)
                        rf.load_task(rf_task)
                        # パラメータUIはないため、rf.paramsはget_rf_paramsのデフォルト値のまま使用されます。

                        # モデルを初期化 (デフォルトパラメータが使われます)
                        rf.model = rf.get_model()

                        # データをロードし、学習用・テスト用に分割
                        rf.load_data(
                            df=eda.df,
                            features=rf_features,
                            target=rf_target,
                            test_size=test_size,
                        )

                        # --- モデルを学習 ---
                        with st.spinner("モデルを学習しています..."):
                            rf.fit() # デフォルトパラメータで学習

                        # --- モデルを評価 ---
                        eval_df = rf.evaluate()

                        # --- 予測結果の取得と表示（テストデータに対するもの） ---
                        # テストデータに対する予測値または予測確率を取得
                        # 分類タスクの場合、predict_proba()の結果を予測ラベルに変換して比較用に表示
                        if rf.task == RFTask.CLASSIFICATION:
                            y_pred_proba = rf.predict(rf.X_test)
                            y_pred_label_or_value = np.argmax(y_pred_proba, axis=1) # 確率 -> ラベル
                        else: # Regression (回帰)
                            y_pred_label_or_value = rf.predict(rf.X_test) # predict()の結果がそのまま予測値

                        # テストデータの実測値を取得し、インデックスをリセットして予測値と揃える
                        actual_values = rf.y_test.reset_index(drop=True)

                        # 予測結果と実測値の比較DataFrameを作成
                        pred_df = pd.DataFrame({
                            "予測値": y_pred_label_or_value,
                            "実際の値": actual_values
                        })


                        # --- 結果を表示 ---
                        st.subheader("モデル評価結果")
                        st.write("テストデータに対するデフォルトパラメータでの評価指標を確認できます。")
                        st.dataframe(eval_df.style.format("{:.3f}"))

                        st.subheader("予測結果")
                        st.write("テストデータに対する予測値と実測値の比較です。")
                        # データが多い場合は先頭の一部のみ表示
                        if pred_df.shape[0] > 100:
                             st.dataframe(pred_df.head(100).style.format("{:.3f}"))
                             st.write(f"... 他 {pred_df.shape[0] - 100} 件を表示していません。")
                        else:
                            st.dataframe(pred_df.style.format("{:.3f}"))

                        # --- 特徴量重要度を表示 ---
                        try:
                            feature_importances_df = rf.feature_importances()
                            if feature_importances_df is not None and not feature_importances_df.empty:
                                st.subheader("特徴量重要度 (Feature Importances)")
                                # draw_barchart コンポーネントを使用して描画
                                draw_barchart(
                                    df=feature_importances_df,
                                    x_col="importance",
                                    y_col="feature", # 特徴量名をY軸に
                                    orient="v", # 横向き棒グラフが見やすい
                                    title="Feature Importances",
                                    # 必要に応じて色やサイズなどを追加
                                )
                            else:
                                st.info("特徴量重要度を取得できませんでした（モデルが訓練されていない、または重要度が全て0など）。")
                        except ValueError as e:
                             st.warning(f"特徴量重要度の取得に失敗しました: {e}")
                        except Exception as e:
                             st.error(f"特徴量重要度の描画中に予期しないエラーが発生しました: {e}")


                    except Exception as e:
                        st.error(f"Random Forestの学習または評価中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示（開発時推奨）
                        import traceback
                        st.error(traceback.format_exc())


            with tab2:
                st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="rf_cv_folds_slider") # ユニークなキーを設定

                if st.button("実行", key="rf_cv_button"): # ユニークなキーを設定
                    try:
                        # --- Random Forestパイプラインの準備 ---
                        # まずタスクを設定 (これにより、内部でデフォルトパラメータがロードされます)
                        rf.load_task(rf_task)
                        # パラメータUIはないため、rf.paramsはget_rf_paramsのデフォルト値のまま使用されます。

                        # データをロード (特徴量/ターゲットの設定のために必要。CV自体はdf全体を使う)
                        rf.load_data(
                            df=eda.df,
                            features=rf_features,
                            target=rf_target,
                        )

                        # --- 交差検証を実行 ---
                        with st.spinner(f"{cv_folds}分割交差検証を実行中..."):
                            # get_model() は cross_validate の内部で呼び出される
                            # その際、rf.params に設定されているデフォルトパラメータが使用されます
                            cv_result = rf.cross_validate(cv=cv_folds)

                        # --- 結果を表示 ---
                        st.subheader("交差検証結果")
                        st.write("デフォルトパラメータでの交差検証結果を確認できます。")
                        st.dataframe(cv_result.style.format("{:.3f}"))

                    except Exception as e:
                        st.error(f"Random Forestの交差検証中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示
                        import traceback
                        st.error(traceback.format_exc())


        # --- 特徴量やターゲットが選択されていない場合の警告 ---
        elif not rf_features:
            st.warning("Random Forestモデルの学習には特徴量カラムが必要です。")
        elif rf_target is None:
            st.warning("Random Forestモデルの学習にはターゲットカラムが必要です。")


    else:
        # データセットがロードされていない場合の情報メッセージ
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。Random Forestモデルを使用するにはデータが必要です。")

def render_logireg_tab(eda: EDADataFrame) -> None:
    """
    Logistic Regressionモデルの学習、評価、係数、交差検証のためのStreamlit UIを描画します。
    パラメータ設定UIは含まれず、デフォルトパラメータが使用されます。

    Parameters
    ----------
    eda : EDADataFrame
        処理対象のEDADataFrameインスタンス。
    """
    st.markdown("## Logistic Regression") # Logistic Regressionタブのヘッダー

    # データセットがロードされているかを確認
    if eda.df is not None and not eda.df.empty:
        # LogisticRegressionPipeline インスタンスを作成
        # Logistic Regressionは分類専用なのでタスクは固定
        lr = LogisticRegressionPipeline(task=LogiRegTask.CLASSIFICATION)

        # --- タスク、特徴量、ターゲットのカラム選択UI ---
        # Logistic Regressionは分類タスク専用のため、タスク選択はUI上不要ですが、
        # 他のモデルとのUI構造を合わせるため、固定表示とします。
        st.markdown(f"**タスク種別:** {LogiRegTask.CLASSIFICATION.value}")
        lr_task = LogiRegTask.CLASSIFICATION # コード内部で使用するためタスクを定義


        # 特徴量カラムの選択 (LRは数値/エンコード済みを推奨)
        # eda.encoded_categorical_columns は、エンコーディング機能で生成されたカラムのリストをedaが保持していると仮定
        available_features = eda.numeric_columns
        if not available_features:
             st.warning("Logistic Regressionの特徴量として使用できる数値またはエンコード済みカラムがデータセットにありません。")
             lr_features = [] # 利用可能なカラムがない場合は特徴量を空リストにする
        else:
            lr_features = st.multiselect(
                "特徴量カラム",
                options=available_features,
                default=available_features, # デフォルトで全ての利用可能なカラムを選択
                key="lr_features_multiselect" # ユニークなキーを設定
            )

        # ターゲットカラムの選択
        available_targets = eda.columns
        default_target_index = 0 # デフォルトのインデックス

        # カテゴリカルカラムが存在する場合、最初のカテゴリカルをデフォルトに試みる
        if eda.categorical_columns:
             try:
                 default_target_index = available_targets.index(eda.categorical_columns[0])
             except ValueError:
                 pass # 見つからなければ0のまま

        # 選択肢が空でないことを確認し、デフォルトインデックスを調整
        if not available_targets:
             lr_target = None
             st.warning("ターゲットカラムとして選択できるカラムがデータセットにありません。")
        else:
            default_target_index = min(default_target_index, len(available_targets) - 1) if available_targets else 0
            lr_target = st.selectbox(
                 "ターゲットカラム",
                 options=available_targets,
                 index=default_target_index,
                 key="lr_target_selectbox" # ユニークなキーを設定
             )

        # データ分割割合のスライダー
        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05, key="lr_test_size_slider") # ユニークなキーを設定

        # --- 特徴量とターゲットが選択されているか確認 ---
        # ロジスティック回帰は分類専用なので、ターゲットが数値型であることも重要
        if lr_target and lr_features:
            # ターゲットカラムのデータ型を確認 (load_dataでもチェックするが、UIでも表示)
            if lr_target in eda.df.columns and not pd.api.types.is_numeric_dtype(eda.df[lr_target]):
                 st.warning(f"選択されたターゲットカラム '{lr_target}' は数値型ではありません（データ型: {eda.df[lr_target].dtype}）。Logistic Regressionは数値ラベルを期待します。前処理でラベルエンコーディングを行ってください。")
                 is_target_type_ok = False
            else:
                 is_target_type_ok = True


            if is_target_type_ok:
                # --- モデル学習/評価 と 交差検証 のためのタブ ---
                tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])

                with tab1:
                    st.write("Logistic Regressionモデルをデフォルトパラメータで学習させ、テストデータで評価を行います。特徴量の係数を確認できます。")
                    if st.button("実行", key="lr_train_button"): # ユニークなキーを設定
                        try:
                            # --- Logistic Regressionパイプラインの準備 ---
                            # タスクは固定 (CLASSIFICATION)
                            # load_task() でデフォルトパラメータがロードされます
                            lr.load_task(lr_task)

                            # モデルを初期化 (デフォルトパラメータが使われます)
                            lr.model = lr.get_model()

                            # データをロードし、学習用・テスト用に分割
                            # load_data の中でターゲットの数値型チェックも再度行われます
                            lr.load_data(
                                df=eda.df,
                                features=lr_features,
                                target=lr_target,
                                test_size=test_size,
                            )

                            # --- モデルを学習 ---
                            with st.spinner("モデルを学習しています..."):
                                lr.fit() # デフォルトパラメータで学習

                            # --- モデルを評価 ---
                            eval_df = lr.evaluate()

                            # --- 予測結果の取得と表示（テストデータに対するもの） ---
                            # テストデータに対する予測ラベルを取得
                            y_pred_label = lr.model.predict(lr.X_test)

                            # テストデータの実測値を取得し、インデックスをリセットして予測値と揃える
                            actual_values = lr.y_test.reset_index(drop=True)

                            # 予測結果と実測値の比較DataFrameを作成
                            pred_df = pd.DataFrame({
                                "予測値": y_pred_label,
                                "実際の値": actual_values
                            })


                            # --- 結果を表示 ---
                            st.subheader("モデル評価結果")
                            st.write("テストデータに対するデフォルトパラメータでの評価指標を確認できます。")
                            st.dataframe(eval_df.style.format("{:.3f}"))

                            st.subheader("予測結果")
                            st.write("テストデータに対する予測値と実測値の比較です。")
                            # データが多い場合は先頭の一部のみ表示
                            if pred_df.shape[0] > 100:
                                 st.dataframe(pred_df.head(100).style.format("{:.3f}"))
                                 st.write(f"... 他 {pred_df.shape[0] - 100} 件を表示していません。")
                            else:
                                st.dataframe(pred_df.style.format("{:.3f}"))

                            # --- 特徴量の係数を表示 ---
                            try:
                                # feature_coefficients メソッドを呼び出す
                                coefficients_df = lr.feature_coefficients()
                                if coefficients_df is not None and not coefficients_df.empty:
                                    st.subheader("特徴量の係数 (Coefficients)")
                                    st.write("各特徴量の予測への寄与度（正の値はクラス予測値を増加、負の値は減少させる傾向）を示します。")
                                    # draw_barchart コンポーネントを使用して描画
                                    draw_barchart(
                                        df=coefficients_df,
                                        x_col="coefficient", # 係数値をX軸に
                                        y_col="feature",     # 特徴量名をY軸に
                                        orient="h",          # 横向き棒グラフが見やすい
                                        title="Feature Coefficients",
                                        # 係数には正負があるため、棒グラフの向きで表現されます
                                    )
                                else:
                                    st.info("特徴量の係数を取得できませんでした（モデルが訓練されていないなど）。")
                            except NotImplementedError as e:
                                 st.warning(f"特徴量の係数表示エラー: {e}")
                                 st.info("現在、多クラス分類の係数表示はこの形式ではサポートされていません。")
                            except ValueError as e:
                                 st.warning(f"特徴量の係数取得に失敗しました: {e}")
                            except Exception as e:
                                 st.error(f"特徴量の係数描画中に予期しないエラーが発生しました: {e}")


                        except Exception as e:
                            st.error(f"Logistic Regressionの学習または評価中にエラーが発生しました: {e}")
                            # より詳細なデバッグ情報のためにトレースバックを表示（開発時推奨）
                            import traceback
                            st.error(traceback.format_exc())


                with tab2:
                    st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                    cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="lr_cv_folds_slider") # ユニークなキーを設定

                    if st.button("実行", key="lr_cv_button"): # ユニークなキーを設定
                        try:
                            # --- Logistic Regressionパイプラインの準備 ---
                            # タスクは固定 (CLASSIFICATION)
                            # load_task() でデフォルトパラメータがロードされます
                            lr.load_task(lr_task)

                            # データをロード (特徴量/ターゲットの設定のために必要。CV自体はdf全体を使う)
                            # load_data の中でターゲットの数値型チェックも再度行われます
                            lr.load_data(
                                df=eda.df,
                                features=lr_features,
                                target=lr_target,
                            )

                            # --- 交差検証を実行 ---
                            with st.spinner(f"{cv_folds}分割交差検証を実行中..."):
                                # get_model() は cross_validate の内部で呼び出される
                                # その際、lr.params に設定されているデフォルトパラメータが使用されます
                                cv_result = lr.cross_validate(cv=cv_folds)

                            # --- 結果を表示 ---
                            st.subheader("交差検証結果")
                            st.write("デフォルトパラメータでの交差検証結果を確認できます。")
                            st.dataframe(cv_result.style.format("{:.3f}"))

                        except Exception as e:
                            st.error(f"Logistic Regressionの交差検証中にエラーが発生しました: {e}")
                            # より詳細なデバッグ情報のためにトレースバックを表示
                            import traceback
                            st.error(traceback.format_exc())


            # --- ターゲットが数値型でない場合の警告（UIの選択後に表示） ---
            # これは load_data 内のチェックと重複しますが、UI上での早期フィードバックとして表示
            # 上記の if is_target_type_ok: ブロックの外に置く
            # pass # is_target_type_ok == False の場合の警告は既に上で表示されている

        # --- 特徴量やターゲットが選択されていない場合の警告 ---
        elif not lr_features:
            st.warning("Logistic Regressionモデルの学習には特徴量カラムが必要です。")
        elif lr_target is None:
            st.warning("Logistic Regressionモデルの学習にはターゲットカラムが必要です。")
        # is_target_type_ok == False の警告は、上記の elif を通過した後に表示されるように配置

    else:
        # データセットがロードされていない場合の情報メッセージ
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。Logistic Regressionモデルを使用するにはデータが必要です。")

def render_linreg_tab(eda: EDADataFrame) -> None:
    """
    Linear Regressionモデルの学習、評価、係数、交差検証のためのStreamlit UIを描画します。
    パラメータ設定UIは含まれず、デフォルトパラメータが使用されます。

    Parameters
    ----------
    eda : EDADataFrame
        処理対象のEDADataFrameインスタンス。
    """
    st.markdown("## Linear Regression") # Linear Regressionタブのヘッダー

    # データセットがロードされているかを確認
    if eda.df is not None and not eda.df.empty:
        # LinearRegressionPipeline インスタンスを作成
        # Linear Regressionは回帰専用なのでタスクは固定
        linreg = LinearRegressionPipeline(task=LinRegTask.REGRESSION)

        # --- タスク、特徴量、ターゲットのカラム選択UI ---
        # Linear Regressionは回帰タスク専用のため、タスク選択はUI上不要ですが、
        # 他のモデルとのUI構造を合わせるため、固定表示とします。
        st.markdown(f"**タスク種別:** {LinRegTask.REGRESSION.value}")
        linreg_task = LinRegTask.REGRESSION # コード内部で使用するためタスクを定義


        # 特徴量カラムの選択 (LRは数値/エンコード済みを推奨)
        # eda.encoded_categorical_columns は、エンコーディング機能で生成されたカラムのリストをedaが保持していると仮定
        available_features = eda.numeric_columns
        if not available_features:
             st.warning("Linear Regressionの特徴量として使用できる数値またはエンコード済みカラムがデータセットにありません。")
             linreg_features = [] # 利用可能なカラムがない場合は特徴量を空リストにする
        else:
            linreg_features = st.multiselect(
                "特徴量カラム",
                options=available_features,
                default=available_features, # デフォルトで全ての利用可能なカラムを選択
                key="linreg_features_multiselect" # ユニークなキーを設定
            )

        # ターゲットカラムの選択 (回帰タスクなので、数値カラム必須)
        available_targets = eda.columns
        default_target_index = 0 # デフォルトのインデックス

        # 数値カラムが存在する場合、最初の数値をデフォルトに試みる
        if eda.numeric_columns:
             try:
                 default_target_index = available_targets.index(eda.numeric_columns[0])
             except ValueError:
                 pass # 見つからなければ0のまま

        # 選択肢が空でないことを確認し、デフォルトインデックスを調整
        if not available_targets:
             linreg_target = None
             st.warning("ターゲットカラムとして選択できるカラムがデータセットにありません。")
        else:
            default_target_index = min(default_target_index, len(available_targets) - 1) if available_targets else 0
            linreg_target = st.selectbox(
                 "ターゲットカラム",
                 options=available_targets,
                 index=default_target_index,
                 key="linreg_target_selectbox" # ユニークなキーを設定
             )

        # データ分割割合のスライダー
        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05, key="linreg_test_size_slider") # ユニークなキーを設定

        # --- 特徴量とターゲットが選択されているか確認 ---
        # 線形回帰は回帰専用なので、ターゲットが数値型であることも重要
        if linreg_target and linreg_features:
            # ターゲットカラムのデータ型を確認 (load_dataでもチェックするが、UIでも表示)
            if linreg_target in eda.df.columns and not pd.api.types.is_numeric_dtype(eda.df[linreg_target]):
                 st.warning(f"選択されたターゲットカラム '{linreg_target}' は数値型ではありません（データ型: {eda.df[linreg_target].dtype}）。Linear Regressionは数値ターゲットを期待します。データ型を確認してください。")
                 is_target_type_ok = False
            else:
                 is_target_type_ok = True


            if is_target_type_ok:
                # --- モデル学習/評価 と 交差検証 のためのタブ ---
                tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])

                with tab1:
                    st.write("Linear Regressionモデルをデフォルトパラメータで学習させ、テストデータで評価を行います。特徴量の係数を確認できます。")
                    if st.button("実行", key="linreg_train_button"): # ユニークなキーを設定
                        try:
                            # --- Linear Regressionパイプラインの準備 ---
                            # タスクは固定 (REGRESSION)
                            # load_task() でデフォルトパラメータがロードされます
                            linreg.load_task(linreg_task)

                            # モデルを初期化 (デフォルトパラメータが使われます)
                            linreg.model = linreg.get_model()

                            # データをロードし、学習用・テスト用に分割
                            # load_data の中でターゲットの数値型チェックも再度行われます
                            linreg.load_data(
                                df=eda.df,
                                features=linreg_features,
                                target=linreg_target,
                                test_size=test_size,
                                random_state=42 # load_data に渡す random_state を固定
                            )

                            # --- モデルを学習 ---
                            with st.spinner("モデルを学習しています..."):
                                linreg.fit() # デフォルトパラメータで学習

                            # --- モデルを評価 ---
                            eval_df = linreg.evaluate()

                            # --- 予測結果の取得と表示（テストデータに対するもの） ---
                            # テストデータに対する予測回帰値を取得
                            y_pred = linreg.predict(linreg.X_test)

                            # テストデータの実測値を取得し、インデックスをリセットして予測値と揃える
                            actual_values = linreg.y_test.reset_index(drop=True)

                            # 予測結果と実測値の比較DataFrameを作成
                            pred_df = pd.DataFrame({
                                "予測値": y_pred,
                                "実際の値": actual_values
                            })


                            # --- 結果を表示 ---
                            st.subheader("モデル評価結果")
                            st.write("テストデータに対するデフォルトパラメータでの評価指標を確認できます。")
                            st.dataframe(eval_df.style.format("{:.3f}"))

                            st.subheader("予測結果")
                            st.write("テストデータに対する予測値と実測値の比較です。")
                            # データが多い場合は先頭の一部のみ表示
                            if pred_df.shape[0] > 100:
                                 st.dataframe(pred_df.head(100).style.format("{:.3f}"))
                                 st.write(f"... 他 {pred_df.shape[0] - 100} 件を表示していません。")
                            else:
                                st.dataframe(pred_df.style.format("{:.3f}"))

                            # --- 特徴量の係数を表示 ---
                            try:
                                # feature_coefficients メソッドを呼び出す
                                coefficients_df = linreg.feature_coefficients()
                                if coefficients_df is not None and not coefficients_df.empty:
                                    st.subheader("特徴量の係数 (Coefficients)")
                                    st.write("各特徴量の予測への寄与度（正の値はターゲット値を増加、負の値は減少させる傾向）を示します。")
                                    # draw_barchart コンポーネントを使用して描画
                                    draw_barchart(
                                        df=coefficients_df,
                                        x_col="coefficient", # 係数値をX軸に
                                        y_col="feature",     # 特徴量名をY軸に
                                        orient="h",          # 横向き棒グラフが見やすい
                                        title="Feature Coefficients",
                                        # 係数には正負があるため、棒グラフの向きで表現されます
                                    )
                                else:
                                    st.info("特徴量の係数を取得できませんでした（モデルが訓練されていないなど）。")
                            except NotImplementedError as e:
                                 st.warning(f"特徴量の係数表示エラー: {e}")
                            except ValueError as e:
                                 st.warning(f"特徴量の係数取得に失敗しました: {e}")
                            except Exception as e:
                                 st.error(f"特徴量の係数描画中に予期しないエラーが発生しました: {e}")


                        except Exception as e:
                            st.error(f"Linear Regressionの学習または評価中にエラーが発生しました: {e}")
                            # より詳細なデバッグ情報のためにトレースバックを表示（開発時推奨）
                            import traceback
                            st.error(traceback.format_exc())


                with tab2:
                    st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                    cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="linreg_cv_folds_slider") # ユニークなキーを設定

                    if st.button("実行", key="linreg_cv_button"): # ユニークなキーを設定
                        try:
                            # --- Linear Regressionパイプラインの準備 ---
                            # タスクは固定 (REGRESSION)
                            # load_task() でデフォルトパラメータがロードされます
                            linreg.load_task(linreg_task)

                            # データをロード (特徴量/ターゲットの設定のために必要。CV自体はdf全体を使う)
                            # load_data の中でターゲットの数値型チェックも再度行われます
                            linreg.load_data(
                                df=eda.df,
                                features=linreg_features,
                                target=linreg_target,
                                random_state=42 # load_data に渡す random_state を固定
                            )

                            # --- 交差検証を実行 ---
                            with st.spinner(f"{cv_folds}分割交差検証を実行中..."):
                                # get_model() は cross_validate の内部で呼び出される
                                # その際、linreg.params に設定されているデフォルトパラメータが使用されます
                                cv_result = linreg.cross_validate(cv=cv_folds)

                            # --- 結果を表示 ---
                            st.subheader("交差検証結果")
                            st.write("デフォルトパラメータでの交差検証結果を確認できます。")
                            st.dataframe(cv_result.style.format("{:.3f}"))

                        except Exception as e:
                            st.error(f"Linear Regressionの交差検証中にエラーが発生しました: {e}")
                            # より詳細なデバッグ情報のためにトレースバックを表示
                            import traceback
                            st.error(traceback.format_exc())


        # --- 特徴量やターゲットが選択されていない場合の警告 ---
        elif not linreg_features:
            st.warning("Linear Regressionモデルの学習には特徴量カラムが必要です。")
        elif linreg_target is None:
            st.warning("Linear Regressionモデルの学習にはターゲットカラムが必要です。")
    else:
        # データセットがロードされていない場合の情報メッセージ
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。Linear Regressionモデルを使用するにはデータが必要です。")


def render_svm_tab(eda: EDADataFrame) -> None:
    """
    Support Vector Machine (SVM) モデルの学習、評価、係数/サポートベクター、交差検証のためのStreamlit UIを描画します。
    パラメータ設定UIは含まれず、デフォルトパラメータが使用されます。

    Parameters
    ----------
    eda : EDADataFrame
        処理対象のEDADataFrameインスタンス。
    """
    st.markdown("## Support Vector Machine (SVM)") # SVMタブのヘッダー
    st.warning("**重要:** SVMは特徴量のスケールに非常に敏感です。効果的なモデル構築のためには、**前処理でスケーリングを行うことを強く推奨します。**") # スケーリングの重要性を強調

    # データセットがロードされているかを確認
    if eda.df is not None and not eda.df.empty:
        # SVMPipeline インスタンスを作成
        # UIで選択されたタスク種別は後ほど load_task() で設定します
        svm = SVMPipeline()

        # --- タスク、特徴量、ターゲットのカラム選択UI ---
        # SVMTask Enum を使用
        svm_task = st.selectbox(
            "タスク種別",
            [
                SVMTask.CLASSIFICATION,
                SVMTask.REGRESSION,
            ],
            format_func=lambda x: {
                SVMTask.CLASSIFICATION: "分類",
                SVMTask.REGRESSION: "回帰"
            }[x],
            index=0,
            key="svm_task_select" # ユニークなキーを設定
        )

        # 特徴量カラムの選択
        available_features = eda.numeric_columns
        if not available_features:
             st.warning("SVMの特徴量として使用できる数値またはエンコード済みカラムがデータセットにありません。")
             svm_features = [] # 利用可能なカラムがない場合は特徴量を空リストにする
        else:
            svm_features = st.multiselect(
                "特徴量カラム",
                options=available_features,
                default=available_features, # デフォルトで全ての利用可能なカラムを選択
                key="svm_features_multiselect" # ユニークなキーを設定
            )

        # ターゲットカラムの選択
        available_targets = eda.columns
        default_target_index = 0 # デフォルトのインデックス

        # 分類タスクの場合、カテゴリカルカラムが存在すれば最初のカテゴリカルをデフォルトに試みる
        if svm_task == SVMTask.CLASSIFICATION and eda.categorical_columns:
             try:
                 default_target_index = available_targets.index(eda.categorical_columns[0])
             except ValueError:
                 pass # 見つからなければ0のまま
        # 回帰タスクの場合、数値カラムが存在すれば最初の数値をデフォルトに試みる
        elif svm_task == SVMTask.REGRESSION and eda.numeric_columns:
             try:
                  default_target_index = available_targets.index(eda.numeric_columns[0])
             except ValueError:
                  pass # 見つからなければ0のまま

        # 選択肢が空でないことを確認し、デフォルトインデックスを調整
        if not available_targets:
             svm_target = None
             st.warning("ターゲットカラムとして選択できるカラムがデータセットにありません。")
        else:
            default_target_index = min(default_target_index, len(available_targets) - 1) if available_targets else 0
            svm_target = st.selectbox(
                 "ターゲットカラム",
                 options=available_targets,
                 index=default_target_index,
                 key="svm_target_selectbox" # ユニークなキーを設定
             )

        # データ分割割合のスライダー
        test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.3, 0.05, key="svm_test_size_slider") # ユニークなキーを設定


        # --- 特徴量とターゲットが選択されているか確認 ---
        # タスクに応じてターゲットのデータ型が適切か確認
        is_target_type_ok = True
        if svm_target and svm_target in eda.df.columns:
             if svm_task == SVMTask.CLASSIFICATION:
                  # SVCは数値ラベルを期待
                  if not pd.api.types.is_numeric_dtype(eda.df[svm_target]):
                       st.warning(f"選択されたターゲットカラム '{svm_target}' は分類タスクに対して数値型ではありません（データ型: {eda.df[svm_target].dtype}）。SVCは数値ラベルを期待します。前処理でラベルエンコーディングを行ってください。")
                       is_target_type_ok = False
             elif svm_task == SVMTask.REGRESSION:
                  # SVRは数値ターゲットを期待
                  if not pd.api.types.is_numeric_dtype(eda.df[svm_target]):
                       st.warning(f"選択されたターゲットカラム '{svm_target}' は回帰タスクに対して数値型ではありません（データ型: {eda.df[svm_target].dtype}）。SVRは数値ターゲットを期待します。データ型を確認してください。")
                       is_target_type_ok = False


        if svm_target and svm_features and is_target_type_ok:
            # --- モデル学習/評価 と 交差検証 のためのタブ ---
            tab1, tab2 = st.tabs(['モデル学習 & 評価', '交差検証'])

            with tab1:
                st.write("SVMモデルをデフォルトパラメータで学習させ、テストデータで評価を行います。リニアカーネル使用時は特徴量の係数も確認できます。")
                if st.button("実行", key="svm_train_button"): # ユニークなキーを設定
                    try:
                        # --- SVMPipelineの準備 ---
                        # タスクを設定 (これにより、内部でデフォルトパラメータがロードされます)
                        svm.load_task(svm_task)

                        # モデルを初期化 (デフォルトパラメータが使われます)
                        svm.model = svm.get_model()
                        # 注意: デフォルトパラメータで probability=True の場合、AUC/LogLoss が計算可能になりますが、
                        # 訓練時間が長くなる可能性があります。

                        # データをロードし、学習用・テスト用に分割
                        # load_data の中でターゲットの数値型チェックも再度行われます
                        svm.load_data(
                            df=eda.df,
                            features=svm_features,
                            target=svm_target,
                            test_size=test_size,
                            random_state=svm._random_state # パイプラインで保持しているrandom_stateを使用
                        )

                        # --- モデルを学習 ---
                        with st.spinner("モデルを学習しています... これには時間がかかる場合があります。"):
                            svm.fit() # デフォルトパラメータで学習

                        # --- モデルを評価 ---
                        eval_df = svm.evaluate()

                        # --- 予測結果の取得と表示（テストデータに対するもの） ---
                        # テストデータに対する予測ラベルまたは回帰値を取得
                        y_pred = svm.model.predict(svm.X_test)

                        # テストデータの実測値を取得し、インデックスをリセットして予測値と揃える
                        actual_values = svm.y_test.reset_index(drop=True)

                        # 予測結果と実測値の比較DataFrameを作成
                        pred_df = pd.DataFrame({
                            "予測値": y_pred,
                            "実際の値": actual_values
                        })


                        # --- 結果を表示 ---
                        st.subheader("モデル評価結果")
                        st.write("テストデータに対するデフォルトパラメータでの評価指標を確認できます。")
                        st.dataframe(eval_df.style.format("{:.3f}"))

                        st.subheader("予測結果")
                        st.write("テストデータに対する予測値と実測値の比較です。")
                        # データが多い場合は先頭の一部のみ表示
                        if pred_df.shape[0] > 100:
                             st.dataframe(pred_df.head(100).style.format("{:.3f}"))
                             st.write(f"... 他 {pred_df.shape[0] - 100} 件を表示していません。")
                        else:
                            st.dataframe(pred_df.style.format("{:.3f}"))

                        # --- 特徴量の係数を表示 (リニアカーネルの場合のみ) ---
                        # カーネルが 'linear' の場合にのみ feature_coefficients を試みる
                        if svm.model.kernel == 'linear':
                             try:
                                 coefficients_df = svm.feature_coefficients()
                                 if coefficients_df is not None and not coefficients_df.empty:
                                     st.subheader("特徴量の係数 (Coefficients)")
                                     st.write("Linearカーネル使用時、各特徴量の予測への寄与度（正負含む）を示します。")
                                     # draw_barchart コンポーネントを使用して描画
                                     draw_barchart(
                                         df=coefficients_df,
                                         x_col="coefficient", # 係数値をX軸に
                                         y_col="feature",     # 特徴量名をY軸に
                                         orient="h",          # 横向き棒グラフが見やすい
                                         title="Feature Coefficients (Linear Kernel)",
                                         # 係数には正負があるため、棒グラフの向きで表現されます
                                     )
                                 else:
                                     st.info("特徴量の係数を取得できませんでした（モデルが訓練されていないなど）。")
                             except NotImplementedError as e:
                                  st.warning(f"特徴量の係数表示エラー: {e}")
                             except ValueError as e:
                                  st.warning(f"特徴量の係数取得に失敗しました: {e}")
                             except Exception as e:
                                  st.error(f"特徴量の係数描画中に予期しないエラーが発生しました: {e}")
                        else:
                             st.info(f"カーネル'{svm.model.kernel}'では特徴量の係数は直接利用できません。")


                        # TODO: サポートベクターの表示機能なども追加可能

                    except Exception as e:
                        st.error(f"SVMの学習または評価中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示（開発時推奨）
                        import traceback
                        st.error(traceback.format_exc())


            with tab2:
                st.write("K分割交差検証を用いて、より安定したモデル評価を行います。")
                cv_folds = st.slider("CV分割数", 2, 10, 5, 1, key="svm_cv_folds_slider") # ユニークなキーを設定

                if st.button("実行", key="svm_cv_button"): # ユニークなキーを設定
                    try:
                        # --- SVMPipelineの準備 ---
                        # タスクを設定 (これにより、内部でデフォルトパラメータがロードされます)
                        svm.load_task(svm_task)
                        # モデルを初期化 (デフォルトパラメータが使われます) - CV内部でも呼ばれるが、ここで設定は必要
                        svm.model = svm.get_model()
                        # 注意: probability=True の設定は CV 内部のモデルにも引き継がれますが、計算時間が長くなる可能性があります。

                        # データをロード (特徴量/ターゲットの設定のために必要。CV自体はdf全体を使う)
                        # load_data の中でターゲットの数値型チェックも再度行われます
                        svm.load_data(
                            df=eda.df,
                            features=svm_features,
                            target=svm_target,
                            random_state=svm._random_state # パイプラインで保持しているrandom_stateを使用
                        )

                        # --- 交差検証を実行 ---
                        with st.spinner(f"{cv_folds}分割交差検証を実行中..."):
                            # get_model() は cross_validate の内部で呼び出される
                            # その際、svm.params に設定されているデフォルトパラメータが使用されます
                            cv_result = svm.cross_validate(cv=cv_folds)

                        # --- 結果を表示 ---
                        st.subheader("交差検証結果")
                        st.write("デフォルトパラメータでの交差検証結果を確認できます。")
                        st.dataframe(cv_result.style.format("{:.3f}"))

                    except Exception as e:
                        st.error(f"SVMの交差検証中にエラーが発生しました: {e}")
                        # より詳細なデバッグ情報のためにトレースバックを表示
                        import traceback
                        st.error(traceback.format_exc())


        # --- 特徴量やターゲットが選択されていない場合、またはターゲットの型が不適切な場合の警告 ---
        elif not svm_features:
            st.warning("SVMモデルの学習には特徴量カラムが必要です。")
        elif svm_target is None:
            st.warning("SVMモデルの学習にはターゲットカラムが必要です。")
        elif not is_target_type_ok:
             # ターゲットの型が不適切な場合の警告は既に上で表示されています
             pass


    else:
        # データセットがロードされていない場合の情報メッセージ
        st.info("データセットがロードされていません。サイドバーからデータを選択またはアップロードしてください。SVMモデルを使用するにはデータが必要です。")



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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ['Dataset', 'LightGBM', 'MLP', 'RandomForest', 'LogisticRegression', 'LinearRegression', 'SVM']
        )
        with tab1:
            render_dataset_tab(eda)
        with tab2:
            render_lgbm_tab(eda)
        with tab3:
            render_mlp_tab(eda)
        with tab4:
            render_rf_tab(eda)
        with tab5:
            render_logireg_tab(eda)
        with tab6:
            render_linreg_tab(eda)
        with tab7:
            render_svm_tab(eda)
    else:
        st.info("EDAを開始するには、サイドバーからデータセットを選択するか、CSVファイルをアップロードしてください。")


if __name__ == '__main__':
    render_page()