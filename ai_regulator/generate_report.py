import json
import os
import os.path as osp
from datetime import datetime
import markdown
import pdfkit
from pypdf import PdfReader

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
    """改定内容の詳細を作成"""
    details = "## 3. 改定内容の詳細\n\n"
    
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
                # 改行を<br>タグに置換
                original = f"{section}<br><br>{item.get('original_text', '-')}" if section else item.get('original_text', '-')
                revised = f"{section}<br><br>{item.get('revised_text', '-')}" if section else item.get('revised_text', '-')
                # 文字列内の改行を<br>タグに置換
                original = original.replace('\n', '<br>')
                revised = revised.replace('\n', '<br>')
                details += f"| {original} | {revised} |\n"
        else:
            # 単一のrevision_dataの場合の処理
            section = revision_data.get('section_name', '')
            original = f"{section}<br><br>{revision_data.get('original_text', '-')}" if section else revision_data.get('original_text', '-')
            revised = f"{section}<br><br>{revision_data.get('revised_text', '-')}" if section else revision_data.get('revised_text', '-')
            original = original.replace('\n', '<br>')
            revised = revised.replace('\n', '<br>')
            details += f"| {original} | {revised} |\n"
        
        details += "\n"
    
    return details

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
        update_info = "## 1. 変更内容\n\n*更新情報なし*\n\n"
    
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