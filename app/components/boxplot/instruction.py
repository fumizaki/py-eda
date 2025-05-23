def instruct_boxplot() -> str:
    lang = {
        'JA': {
            'instruction': """
        **目的:**
        - データの分布を要約して視覚化します。箱はデータの中心的な範囲（四分位数範囲 IQR）、線は中央値を示し、ひげは通常、データの大部分が含まれる範囲（例：IQRの1.5倍）を示します。ひげの外側の点は外れ値の候補です。カテゴリごとの分布比較によく用いられます。
                 
        **示唆:**
        - **中心傾向:** 各グループの中央値（箱の中の線）を比較できます。
        - **ばらつき:** 箱の高さ（IQR）やひげの長さで、データのばらつき具合を比較できます。箱が短いほどデータが中央値周りに集中しています。
        - **分布の対称性:** 箱の中の中央線の位置が箱の中心からずれている場合、データ分布の歪み（skewness）を示唆します。
        - **外れ値:** 箱やひげから離れた点がプロットされ、外れ値の存在を確認できます。
        - **グループ間比較:** X軸にカテゴリ変数を指定することで、カテゴリごとの量的変数の分布を簡単に比較できます。
        """
        }
    }

    return lang['JA']['instruction']