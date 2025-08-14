import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================
# ページ設定
# ============================================
st.set_page_config(
    page_title="ノーコードデータ整形",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('ノーコード データ整形アプリ')

# セッション状態の初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'column_renames' not in st.session_state:
    st.session_state.column_renames = {}
if 'value_changes' not in st.session_state:
    st.session_state.value_changes = {}

def get_dtype_name(dtype):
    """パンダスのデータ型を分かりやすい名前に変換"""
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    elif pd.api.types.is_float_dtype(dtype):
        return "float"
    elif pd.api.types.is_bool_dtype(dtype):
        return "bool"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    elif pd.api.types.is_categorical_dtype(dtype):
        return "factor"
    elif pd.api.types.is_string_dtype(dtype):
        return "string"
    else:
        return "object"

def get_unique_values(df, column):
    """指定した列のユニークな値を取得（上限100個）"""
    unique_vals = df[column].dropna().unique()
    if len(unique_vals) > 100:
        return list(unique_vals[:100])
    return list(unique_vals)

def convert_value_by_type(value, target_type):
    """指定された型に値を変換"""
    try:
        if target_type == "int":
            return int(float(value))  # floatを経由してintに変換（小数点がある場合も対応）
        elif target_type == "float":
            return float(value)
        elif target_type == "bool":
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        elif target_type == "factor":
            return str(value)  # factorは文字列として扱う（後で列全体をcategory型に変換）
        elif target_type == "date":
            if isinstance(value, str):
                # 一般的な日付フォーマットを試す
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return pd.to_datetime(value, format=fmt)
                    except:
                        continue
                # フォーマットが合わない場合はpandasの自動推定に任せる
                return pd.to_datetime(value)
            return pd.to_datetime(value)
        else:  # string
            return str(value)
    except Exception as e:
        st.warning(f"値 '{value}' を型 '{target_type}' に変換できませんでした: {e}")
        return value

def check_column_type_consistency(df, column_name):
    """列内のデータ型の一貫性をチェック"""
    series = df[column_name]
    
    # 欠損値を除外
    non_null_series = series.dropna()
    
    if len(non_null_series) == 0:
        return True, "列にデータがありません"
    
    # 各値の型を確認
    type_counts = {}
    for value in non_null_series:
        value_type = type(value).__name__
        type_counts[value_type] = type_counts.get(value_type, 0) + 1
    
    # 型が2つ以上ある場合は不整合
    if len(type_counts) > 1:
        return False, f"列内に複数の型が存在します: {dict(type_counts)}"
    
    return True, "型の一貫性に問題ありません"

def apply_column_type_conversion(df, value_changes):
    """列全体の型変換を適用"""
    result_df = df.copy()
    
    # 各列で指定された型を収集
    column_types = {}
    for column, changes in value_changes.items():
        if column in result_df.columns:
            # その列で使用されている型を取得（最後に指定された型を使用）
            for old_val, change_info in changes.items():
                if isinstance(change_info, dict):
                    column_types[column] = change_info['type']
    
    # 列全体の型変換を実行
    for column, target_type in column_types.items():
        try:
            if target_type == "int":
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce').astype('Int64')
            elif target_type == "float":
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce').astype('float64')
            elif target_type == "bool":
                # まず文字列に変換してからboolに
                result_df[column] = result_df[column].astype(str).str.lower().isin(['true', '1', 'yes', 'on', 'True'])
            elif target_type == "factor":
                # すべての値を文字列に変換してからcategory型に
                result_df[column] = result_df[column].astype(str).astype('category')
            elif target_type == "date":
                result_df[column] = pd.to_datetime(result_df[column], errors='coerce')
            elif target_type == "string":
                result_df[column] = result_df[column].astype('string')
        except Exception as e:
            st.warning(f"列 '{column}' を型 '{target_type}' に変換中にエラーが発生しました: {e}")
    
    return result_df

def create_conversion_template():
    """変換表のテンプレートExcelファイルを作成"""
    template_data = {
        'col_original': ['番号', '性別', '性別'],
        'col_new': ['ID', 'gender', 'gender'], 
        'data_type': ['string', 'factor', 'factor'],
        'original_value': ['', '1', '0'],
        'new_value': ['', 'male', 'female']
    }
    template_df = pd.DataFrame(template_data)
    return template_df

def parse_conversion_table(conversion_df):
    """変換表を解析してcolumn_renamesとvalue_changesに変換"""
    column_renames = {}
    value_changes = {}
    
    for _, row in conversion_df.iterrows():
        col_original = row['col_original']
        col_new = row['col_new']
        data_type = row['data_type']
        original_value = row['original_value']
        new_value = row['new_value']
        
        # 列名変更の処理
        if pd.notna(col_original) and pd.notna(col_new) and col_original != col_new:
            column_renames[col_original] = col_new
        
        # 値変換の処理
        if pd.notna(original_value) and pd.notna(new_value) and original_value != '' and new_value != '':
            target_column = col_new if pd.notna(col_new) else col_original
            
            if target_column not in value_changes:
                value_changes[target_column] = {}
            
            # 型に応じて値を変換
            converted_value = convert_value_by_type(new_value, data_type)
            
            value_changes[target_column][str(original_value)] = {
                'value': converted_value,
                'type': data_type
            }
    
    return column_renames, value_changes

# ============================================
# ファイルのアップロード
# ============================================
st.subheader("ファイルのアップロード")
uploaded_file = st.file_uploader("CSVデータの列名と値をノーコードで変換します", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df.copy()
        st.success(f"ファイルが正しくアップロードされました（{df.shape[0]}行 × {df.shape[1]}列）")
        
        # データの概要を表示
        with st.expander("データの概要を確認"):
            st.write("**データ型情報:**")
            dtype_info = pd.DataFrame({
                '列名': df.columns,
                'データ型': [get_dtype_name(df[col].dtype) for col in df.columns],
                'ユニーク数': [df[col].nunique() for col in df.columns],
                '欠損値数': [df[col].isnull().sum() for col in df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)
            
            st.write("**データのプレビュー:**")
            st.dataframe(df.head(10), use_container_width=True)
            
    except Exception as e:
        st.error(f"ファイルの読み込みでエラーが発生しました: {e}")

# ============================================
# 変換方法選択（タブ形式）
# ============================================
if st.session_state.df is not None:
    df = st.session_state.df
    
    # タブを作成
    tab1, tab2 = st.tabs(["🔧 入力して変換", "📊 変換表で変換"])
    
    with tab1:
        # ============================================
        # 列名の変更セクション
        # ============================================
        st.subheader("🏷️ 列名の変更")
        
        # 現在の列名リストを取得（既に変更があれば反映）
        current_columns = list(df.columns)
        for old_name, new_name in st.session_state.column_renames.items():
            if old_name in current_columns:
                idx = current_columns.index(old_name)
                current_columns[idx] = new_name
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            target_column = st.selectbox(
                "変更したい列名を選択", 
                options=list(df.columns),
                key="column_select"
            )
        
        with col2:
            new_column_name = st.text_input(
                "新しい列名を入力", 
                value="",
                key="new_column_name"
            )
        
        with col3:
            if st.button("列名を変更", key="add_column_rename"):
                if new_column_name and target_column:
                    st.session_state.column_renames[target_column] = new_column_name
                    st.success(f"'{target_column}' → '{new_column_name}' を追加しました")
                    st.rerun()
        
        # 現在の列名変更リストを表示
        if st.session_state.column_renames:
            st.write("**現在の列名変更予定:**")
            rename_df = pd.DataFrame([
                {"元の列名": old, "新しい列名": new, "削除": i} 
                for i, (old, new) in enumerate(st.session_state.column_renames.items())
            ])
            
            for idx, row in rename_df.iterrows():
                col_a, col_b = st.columns([3, 3])
                with col_a:
                    st.text(f"{row['元の列名']} → {row['新しい列名']}")

                with col_b:
                    if st.button("削除", key=f"del_col_{idx}"):
                        old_name = row["元の列名"]
                        del st.session_state.column_renames[old_name]
                        st.rerun()

        # ============================================
        # 値の変更セクション
        # ============================================
        st.subheader("🔄 値の変換")
        
        # 値変換の入力フォームと結果表示を2列レイアウトに
        input_col, stock_col = st.columns([3, 2])
        
        with input_col:
            st.write("**値変換の設定**")
            # 値変換の入力フォーム
            col1, col2, col3, col4 = st.columns([2, 0.3, 2, 1.5])
            
            with col1:
                target_value_column = st.selectbox(
                    "変換する列",
                    options=list(df.columns),
                    key="value_column_select"
                )

                # 元の値の選択肢を動的に生成
                if target_value_column:
                    unique_values = get_unique_values(df, target_value_column)
                    old_value = st.selectbox(
                        "変換前の値",
                        options=unique_values,
                        key="old_value_select"
                    )
                else: 
                    old_value = None
            
            with col2:
                st.markdown("<div style='text-align: center; padding-top: 95px;'><h2>→</h2></div>", 
                           unsafe_allow_html=True)
            
            with col3:
                # 列情報の表示
                if target_value_column:
                    current_dtype = get_dtype_name(df[target_value_column].dtype)
                    unique_count = df[target_value_column].nunique()
                    
                    # 型選択のセレクトボックス
                    target_type = st.selectbox(
                        "変換後の型",
                        options=["string", "int", "float", "bool", "factor", "date"],
                        index=0,  # デフォルトはstring
                        key="target_type_select"
                    )

                    # 新しい値の入力
                    new_value = st.text_input(
                        "変換後の値",
                        key="new_value_input"
                    )
                else:
                    new_value = ""
                    target_type = "string"
            
            with col4:
                # 値変換を追加するボタン
                st.markdown("<div style='text-align: center; padding-top: 25px;'><h2> </h2></div>", 
                           unsafe_allow_html=True)
                if st.button("追加", key="add_value_change"):
                    if target_value_column and old_value is not None and new_value:
                        # セッション状態に値変換を保存
                        if target_value_column not in st.session_state.value_changes:
                            st.session_state.value_changes[target_value_column] = {}
                        
                        # 選択された型に変換
                        converted_value = convert_value_by_type(new_value, target_type)
                        
                        # 型情報も一緒に保存
                        st.session_state.value_changes[target_value_column][str(old_value)] = {
                            'value': converted_value,
                            'type': target_type
                        }
                        st.success(f"追加: '{old_value}' → '{converted_value}' ({target_type})")
                        st.rerun()
        
        with stock_col:
            st.write("**変換予定一覧**")
            
            if st.session_state.value_changes:
                total_changes = sum(len(changes) for changes in st.session_state.value_changes.values())
                st.caption(f"全{len(st.session_state.value_changes)}列, {total_changes}件の変換予定")
                
                for column_name, changes in st.session_state.value_changes.items():
                    st.write(f"**{column_name}**")
                    
                    for i, (old_val, change_info) in enumerate(changes.items()):
                        change_col, del_col = st.columns([5, 2])
                        
                        with change_col:
                            # 型情報も表示
                            if isinstance(change_info, dict):
                                new_val = change_info['value']
                                val_type = change_info['type']
                                st.text(f"  {old_val} → {new_val} ({val_type})")
                            else:
                                # 旧形式との互換性
                                st.text(f"  {old_val} → {change_info}")
                        
                        with del_col:
                            if st.button("削除", key=f"del_val_{column_name}_{i}"):
                                del st.session_state.value_changes[column_name][old_val]
                                if not st.session_state.value_changes[column_name]:
                                    del st.session_state.value_changes[column_name]
                                st.rerun()
            else:
                st.info("変換予定はありません")
    
    with tab2:
        # ============================================
        # 変換表による変換セクション
        # ============================================
        st.subheader("📊 変換表による一括変換")
        
        # テンプレートダウンロード
        st.write("**1. 変換表テンプレートをダウンロード**")
        template_df = create_conversion_template()
        
        # テンプレートをExcelとしてダウンロード
        from io import BytesIO
        buffer = BytesIO()
        template_df.to_excel(buffer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 変換表テンプレート.xlsxをダウンロード",
            data=buffer.getvalue(),
            file_name="conversion_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        # テンプレートの説明
        with st.expander("変換表の使い方"):
            st.write("""
            
            **記入方法:**
            1. **列名変更**をしたいときは、**original_valueとnew_valueは空白にして、**col_newを入力します。
            2. **値の変換**をしたいときは、**col_originalとcol_newを同じにして、**original_valueとnew_valueを入力します。その際、data_typeの型でnew_valueに変換されます。
            """)
            
            st.write("**テンプレート例:**")
            st.dataframe(template_df, use_container_width=True)
        
        # 変換表のアップロード
        st.write("**2. 変換表をアップロード**")
        uploaded_conversion_file = st.file_uploader(
            "変換表ファイルをアップロード", 
            type=["xlsx", "xls"],
            key="conversion_file"
        )
        
        if uploaded_conversion_file is not None:
            try:
                conversion_df = pd.read_excel(uploaded_conversion_file)
                
                # 必要な列があるかチェック
                required_columns = ['col_original', 'col_new', 'data_type', 'original_value', 'new_value']
                missing_columns = [col for col in required_columns if col not in conversion_df.columns]
                
                if missing_columns:
                    st.error(f"変換表に必要な列がありません: {missing_columns}")
                else:
                    st.success(f"変換表が正しくアップロードされました（{conversion_df.shape[0]}行の変換設定）")
                    
                    # 変換表の内容を表示
                    st.write("**アップロードされた変換表:**")
                    st.dataframe(conversion_df, use_container_width=True)
                    
                    # 変換表を解析
                    if st.button("変換表を適用", key="apply_conversion_table"):
                        try:
                            column_renames, value_changes = parse_conversion_table(conversion_df)
                            
                            # セッション状態を更新
                            st.session_state.column_renames = column_renames
                            st.session_state.value_changes = value_changes
                            
                            st.success("変換表が正常に適用されました！")
                            
                            # 適用された変換の概要を表示
                            if column_renames:
                                st.write("**列名変更:**")
                                for old, new in column_renames.items():
                                    st.write(f"- {old} → {new}")
                            
                            if value_changes:
                                st.write("**値変換:**")
                                for column, changes in value_changes.items():
                                    st.write(f"**{column}列:**")
                                    for old_val, change_info in changes.items():
                                        new_val = change_info['value']
                                        val_type = change_info['type']
                                        st.write(f"  - {old_val} → {new_val} ({val_type})")
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"変換表の適用でエラーが発生しました: {e}")
            
            except Exception as e:
                st.error(f"変換表の読み込みでエラーが発生しました: {e}")

    # ============================================
    # 変換実行と結果表示（タブの外で共通）
    # ============================================
    st.subheader("変換の実行と結果")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("変換を実行", type="primary"):
            try:
                # データフレームのコピーを作成
                result_df = df.copy()
                
                # 値の変換を実行（修正版）
                conversion_log = []  # 変換ログを記録
                
                for column, changes in st.session_state.value_changes.items():
                    if column in result_df.columns:
                        for old_val, change_info in changes.items():
                            # 新しい形式と旧形式の両方に対応
                            if isinstance(change_info, dict):
                                new_val = change_info['value']
                            else:
                                new_val = change_info
                            
                            # 変換前の値の数をカウント
                            before_count = (result_df[column] == old_val).sum()
                            
                            # 型を考慮した値の置換
                            original_series = result_df[column]
                            if pd.api.types.is_numeric_dtype(original_series):
                                try:
                                    # 数値型の場合、old_valを適切な型に変換
                                    if pd.api.types.is_integer_dtype(original_series):
                                        old_val_converted = int(float(old_val))
                                    else:
                                        old_val_converted = float(old_val)
                                    
                                    # 置換実行
                                    mask = result_df[column] == old_val_converted
                                    result_df.loc[mask, column] = new_val
                                    
                                    # 変換後の確認
                                    after_count = mask.sum()
                                    conversion_log.append({
                                        'column': column,
                                        'old_val': old_val,
                                        'new_val': new_val,
                                        'converted_count': after_count,
                                        'expected_count': before_count
                                    })
                                    
                                except (ValueError, TypeError):
                                    # 数値変換に失敗した場合は文字列として処理
                                    mask = result_df[column].astype(str) == str(old_val)
                                    result_df.loc[mask, column] = new_val
                                    after_count = mask.sum()
                                    conversion_log.append({
                                        'column': column,
                                        'old_val': old_val,
                                        'new_val': new_val,
                                        'converted_count': after_count,
                                        'expected_count': before_count
                                    })
                            else:
                                # 文字列型の場合
                                mask = result_df[column].astype(str) == str(old_val)
                                result_df.loc[mask, column] = new_val
                                after_count = mask.sum()
                                conversion_log.append({
                                    'column': column,
                                    'old_val': old_val,
                                    'new_val': new_val,
                                    'converted_count': after_count,
                                    'expected_count': before_count
                                })
                
                # 列全体の型変換を適用
                result_df = apply_column_type_conversion(result_df, st.session_state.value_changes)
                
                # 列名の変更を実行
                if st.session_state.column_renames:
                    result_df = result_df.rename(columns=st.session_state.column_renames)
                
                # データ型の整合性チェック
                type_errors = []
                for column in result_df.columns:
                    is_consistent, message = check_column_type_consistency(result_df, column)
                    if not is_consistent:
                        type_errors.append(f"列 '{column}': {message}")
                
                # 変換結果の検証
                conversion_errors = []
                for log in conversion_log:
                    if log['converted_count'] != log['expected_count']:
                        conversion_errors.append(
                            f"列 '{log['column']}': 値 '{log['old_val']}' の変換で不整合が発生しました "
                            f"(期待: {log['expected_count']}件, 実際: {log['converted_count']}件)"
                        )
                
                # エラーがある場合は警告を表示
                if type_errors or conversion_errors:
                    st.error("⚠️ 変換でエラーが発生しました:")
                    for error in type_errors + conversion_errors:
                        st.error(f"• {error}")
                    
                    # エラーがあっても結果は保存（ユーザーが確認できるように）
                    st.session_state.result_df = result_df
                    st.warning("変換結果は保存されましたが、上記のエラーを確認してください。")
                else:
                    st.session_state.result_df = result_df
                    st.success("変換が正常に完了しました！")
                
                # 変換ログの表示
                if conversion_log:
                    with st.expander("変換の詳細ログ"):
                        for log in conversion_log:
                            st.write(f"列 '{log['column']}': {log['old_val']} → {log['new_val']} ({log['converted_count']}件変換)")
                
            except Exception as e:
                st.error(f"変換中にエラーが発生しました: {e}")
                st.error("詳細なエラー情報を確認してください")
    
    with col2:
        if st.button("設定をリセット"):
            st.session_state.column_renames = {}
            st.session_state.value_changes = {}
            if 'result_df' in st.session_state:
                del st.session_state.result_df
            st.success("設定をリセットしました")
            st.rerun()
    
    # 結果の表示
    if 'result_df' in st.session_state:
        st.write("**変換結果のプレビュー:**")
        result_df = st.session_state.result_df
        
        # 変換前後の比較
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**変換前（最初の5行）:**")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.write("**変換後（最初の5行）:**")
            st.dataframe(result_df.head(), use_container_width=True)
        
        # ダウンロードボタン
        with col3:
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 CSVをダウンロード",
                data=csv,
                file_name="transformed_data.csv",
                mime="text/csv"
            )
        
        # 変換の詳細情報
        with st.expander("変換の詳細情報"):
            st.write("**実行された変換:**")
            
            if st.session_state.column_renames:
                st.write("*列名の変更:*")
                for old, new in st.session_state.column_renames.items():
                    st.write(f"- {old} → {new}")
            
            if st.session_state.value_changes:
                st.write("*値の変換:*")
                for column, changes in st.session_state.value_changes.items():
                    st.write(f"**{column}列:**")
                    for old_val, change_info in changes.items():
                        if isinstance(change_info, dict):
                            new_val = change_info['value']
                            val_type = change_info['type']
                            st.write(f"  - {old_val} → {new_val} ({val_type})")
                        else:
                            st.write(f"  - {old_val} → {change_info}")
            
            st.write("**変換後のデータ型:**")
            dtype_info_after = pd.DataFrame({
                '列名': result_df.columns,
                'データ型': [get_dtype_name(result_df[col].dtype) for col in result_df.columns],
                'ユニーク数': [result_df[col].nunique() for col in result_df.columns]
            })
            st.dataframe(dtype_info_after, use_container_width=True)

else:
    st.info("👆 CSVファイルをアップロードしてください")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
    🄫 2025 Kyohei Suzuki</a>
    </div>
    """,
    unsafe_allow_html=True
)