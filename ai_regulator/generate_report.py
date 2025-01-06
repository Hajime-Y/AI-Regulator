import json
import os
import os.path as osp
from datetime import datetime
import markdown
import pdfkit
import difflib

def load_json_file(file_path):
    """JSONファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_regulation_summary(regulations):
    """規定集リストの要約を作成"""
    summary = "## 2. 改定必要性の確認結果\n\n"
    summary += "| 規定名 | 改定理由(確認前想定) | 改定理由(確認後) | 改定対象 |\n"
    summary += "|--------|-------------------|----------------|----------|\n"
    
    for reg in regulations:
        path = reg.get('path', '')
        # 拡張子なしのファイル名を取得
        filename = os.path.splitext(os.path.basename(path))[0]
        initial_reason = reg.get('reason', '-').replace('\n', '<br>')
        confirmed_reason = reg.get('comment', '-').replace('\n', '<br>')
        needs_revision = '要改定' if reg.get('revision_needed', False) else '不要'
        
        summary += f"| {filename} | {initial_reason} | {confirmed_reason} | {needs_revision} |\n"

    summary += "\n"
    
    return summary

def create_revision_details(base_dir):
    """改定案の詳細を作成"""
    details = "## 3. 改定案の詳細\n\n"
    
    # base_dir内の各改定フォルダを処理
    for folder in sorted(os.listdir(base_dir)):
        if folder.startswith('.'):  # ドットで始まるフォルダを無視
            continue
        folder_path = osp.join(base_dir, folder)
        if not osp.isdir(folder_path):
            continue
            
        revision_file = osp.join(folder_path, "revision.json")
        review_file = osp.join(folder_path, "review.json")
        review_improved_file = osp.join(folder_path, "review_improved.json")
        
        # 改定内容の読み込み
        revision_data = load_json_file(revision_file)
        if not revision_data:
            continue
            
        # レビュー内容の読み込み（改善後のレビューを優先）
        review_data = load_json_file(review_improved_file) or load_json_file(review_file)
        
        details += f"### {folder}\n\n"
        
        # レビュー評価の追加
        if review_data:
            details += "#### レビュー評価\n\n"
            review_items = {
                'format_check': 'フォーマット評価',
                'removal_check': '削除チェック',
                'consistency_check': '一貫性評価',
                'completeness_check': '完全性評価',
                'documentation_check': '記載原則評価',
                'necessity_check': '改定必要性評価',
                'overall': '総合評価',
                'confidence': '信頼度',
                'comment': 'コメント'
            }
            for key, label in review_items.items():
                if key in review_data:
                    details += f"- **{label}**: {review_data[key]}\n"
            details += "\n"
        
        # 新旧対比表の作成
        details += "#### 新旧対比表\n\n"
        details += "| 改定前 | 改定後 |\n"
        details += "|--------|--------|\n"
        
        # revision_dataがリストの場合の処理
        if isinstance(revision_data, list):
            for item in revision_data:
                section = item.get('section_name', '')
                original_text = item.get('original_text', '-')
                revised_text = item.get('revised_text', '-')

                # 差分を強調
                highlighted_old, highlighted_new = highlight_differences(original_text, revised_text)

                # セクション名を上に表示
                if section:
                    highlighted_old = f"{section}<br><br>{highlighted_old}"
                    highlighted_new = f"{section}<br><br>{highlighted_new}"
                
                # 改行を <br> タグに置き換える
                highlighted_old = highlighted_old.replace('\n', '<br>')
                highlighted_new = highlighted_new.replace('\n', '<br>')

                details += f"| {highlighted_old} | {highlighted_new} |\n"
        else:
            # 単一のrevision_dataの場合の処理
            section = revision_data.get('section_name', '')
            original_text = revision_data.get('original_text', '-')
            revised_text = revision_data.get('revised_text', '-')

            # 差分を強調
            highlighted_old, highlighted_new = highlight_differences(original_text, revised_text)

            if section:
                highlighted_old = f"{section}<br><br>{highlighted_old}"
                highlighted_new = f"{section}<br><br>{highlighted_new}"

            # 改行を <br> タグに置き換える
            highlighted_old = highlighted_old.replace('\n', '<br>')
            highlighted_new = highlighted_new.replace('\n', '<br>')

            details += f"| {highlighted_old} | {highlighted_new} |\n"
        
        details += "\n"
    
    return details

def highlight_differences(old_text, new_text):
    """
    old_text と new_text の差分を比較し、差分があった箇所を <u> タグで強調して返す
    """
    # 改定後テキストが"-"の場合は、改定前テキストはそのまま返す
    if new_text == '-':
        return old_text, "（改定作業においてやはり改定は不要だと判断しました。）"

    # SequenceMatcher で差分を解析
    d = difflib.SequenceMatcher(None, old_text, new_text)
    result_old = ""
    result_new = ""

    # get_opcodes() は差分の種類と、その差分が生じた index 範囲を返す
    # 例: [('equal', 0, 5, 0, 5), ('replace', 5, 6, 5, 6), ...] のようなイメージ
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        old_segment = old_text[i1:i2]
        new_segment = new_text[j1:j2]

        if tag == 'replace':
            # old_text と new_text の両方が書き換わった
            result_old += f'<u>{old_segment}</u>'
            result_new += f'<u>{new_segment}</u>'
        elif tag == 'delete':
            # old_text 側で削除された文字列
            result_old += f'<u>{old_segment}</u>'
        elif tag == 'insert':
            # new_text 側で追加された文字列
            result_new += f'<u>{new_segment}</u>'
        elif tag == 'equal':
            # 変更がない部分はそのまま連結
            result_old += old_segment
            result_new += new_segment

    return result_old, result_new


def generate_report(base_dir, regulations_file, update_info_file, md_report, pdf_report):
    """改定レポートを生成"""
    # 1. 改定内容（update_info.txt）
    try:
        with open(update_info_file, 'r', encoding='utf-8') as f:
            update_info_raw = f.read()
            update_info_formatted = update_info_raw.replace('\\n', '\n').replace('\n', '<br>')

            # 特殊文字を置換
            escape_chars = ['*', '_', '`']
            for char in escape_chars:
                update_info_formatted = update_info_formatted.replace(char, f"\\{char}")

            update_info = f"## 1. 変更内容\n\n{update_info_formatted}\n\n"
    except Exception as e:
        update_info = "## 1. 変更内容\n\n*変更情報なし*\n\n"
    
    # 2. 規定集リストの要約
    regulations = load_json_file(regulations_file) or []
    regulations_summary = create_regulation_summary(regulations)
    
    # 3. 改定内容の詳細
    revision_details = create_revision_details(base_dir)
    
    # マークダウンの作成
    timestamp = datetime.now().strftime("%Y年%m月%d日")
    markdown_content = f"# 規定改定レポート（{timestamp}）\n\n"
    markdown_content += update_info
    markdown_content += regulations_summary
    markdown_content += revision_details
    
    # マークダウンファイルの保存
    with open(md_report, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # PDFの生成
    try:
        # マークダウンをHTMLに変換
        html_body = markdown.markdown(
            markdown_content,
            extensions=['tables'],
            output_format='html5'
        )
        
        # テーブルスタイル・フォント用のCSSを追加
        css_styles = """
        <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        table, th, td {
            border: 1px solid lightgrey;
        }
        th {
            background-color: #f2f2f2;
        }
        /* フォント指定を追加 */
        body {
            font-family: "Hiragino Kaku Gothic Pro", "ヒラギノ角ゴ Pro W3", Meiryo, "メイリオ", sans-serif;
        }
        </style>
        """
        
        # 完全なHTMLドキュメントを作成
        html_text = f"""
        <html>
        <head>
        {css_styles}
        </head>
        <body>
        {html_body}
        </body>
        </html>
        """
        
        # PDF生成オプションの設定
        options = {
            'page-size': 'A4',
            'margin-top': '0.4in',
            'margin-right': '0.4in',
            'margin-bottom': '0.4in',
            'margin-left': '0.4in',
            'encoding': "UTF-8",
            'no-outline': None,
            'disable-smart-shrinking': '',
        }
        
        pdfkit.from_string(html_text, pdf_report, options=options)
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("PDF generation skipped. Please install wkhtmltopdf if you want to generate PDFs.")
    
    return markdown_content

if __name__ == "__main__":
    import argparse
    import os.path as osp

    print("[*] Generating final report...")
    
    parser = argparse.ArgumentParser(description='規定改定レポートを生成します')
    parser.add_argument('--base-dir', required=True, help='改定フォルダが含まれるベースディレクトリ')
    
    args = parser.parse_args()
    
    generate_report(
        base_dir=args.base_dir,
        regulations_file=osp.join(args.base_dir, "target_regulations.json"),
        update_info_file=osp.join(args.base_dir, "update_info.txt"),
        md_report=osp.join(args.base_dir, "revision_report.md"),
        pdf_report=osp.join(args.base_dir, "revision_report.pdf")
    )

    print("[+] Report generation completed.")
    print(f"    - Markdown report: {osp.join(args.base_dir, 'revision_report.md')}")
    print(f"    - PDF report: {osp.join(args.base_dir, 'revision_report.pdf')}")