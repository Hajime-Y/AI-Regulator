import json
import os
import os.path as osp
from datetime import datetime
import markdown
import pdfkit

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
    summary = "## 2. 規定集確認結果\n\n"
    summary += "| 規定名 | 改訂理由(確認前想定) | 改訂理由(確認後) | 改訂対象 |\n"
    summary += "|--------|-------------------|----------------|----------|\n"
    
    for reg in regulations:
        path = reg.get('path', '')
        initial_reason = reg.get('initial_reason', '-')
        confirmed_reason = reg.get('confirmed_reason', '-')
        needs_revision = '要改訂' if reg.get('revision_needed', False) else '不要'
        
        summary += f"| {path} | {initial_reason} | {confirmed_reason} | {needs_revision} |\n"
    
    return summary

def create_revision_details(base_dir):
    """改訂内容の詳細を作成"""
    details = "## 3. 改訂内容の詳細\n\n"
    
    # base_dir内の各改訂フォルダを処理
    for folder in sorted(os.listdir(base_dir)):
        folder_path = osp.join(base_dir, folder)
        if not osp.isdir(folder_path):
            continue
            
        revision_file = osp.join(folder_path, "revision.json")
        review_file = osp.join(folder_path, "review.json")
        review_improved_file = osp.join(folder_path, "review_improved.json")
        
        # 改訂内容の読み込み
        revision_data = load_json_file(revision_file)
        if not revision_data:
            continue
            
        # レビュー内容の読み込み（改善後のレビューを優先）
        review_data = load_json_file(review_improved_file) or load_json_file(review_file)
        
        details += f"### {folder}\n\n"
        
        # レビュー評価の追加
        if review_data:
            details += "#### レビュー評価\n\n"
            for key, value in review_data.items():
                if key not in ['original_text', 'revised_text']:
                    details += f"- **{key}**: {value}\n"
            details += "\n"
        
        # 新旧対比表の作成
        details += "#### 新旧対比表\n\n"
        details += "| 改定前 | 改訂後 |\n"
        details += "|--------|--------|\n"
        
        # revision_dataがリストの場合の処理
        if isinstance(revision_data, list):
            for item in revision_data:
                section = item.get('section', '')
                original = f"{section}\n\n{item.get('original_text', '-')}" if section else item.get('original_text', '-')
                revised = f"{section}\n\n{item.get('revised_text', '-')}" if section else item.get('revised_text', '-')
                details += f"| {original} | {revised} |\n"
        else:
            # 単一のrevision_dataの場合の処理
            section = revision_data.get('section', '')
            original = f"{section}\n\n{revision_data.get('original_text', '-')}" if section else revision_data.get('original_text', '-')
            revised = f"{section}\n\n{revision_data.get('revised_text', '-')}" if section else revision_data.get('revised_text', '-')
            details += f"| {original} | {revised} |\n"
        
        details += "\n"
    
    return details

def generate_report(base_dir, regulations_file, update_info_file, md_report, pdf_report):
    """改訂レポートを生成"""
    # 1. 改訂内容（update_info.txt）
    try:
        with open(update_info_file, 'r', encoding='utf-8') as f:
            update_info = f"## 1. 改訂内容\n\n{f.read()}\n\n"
    except Exception as e:
        update_info = "## 1. 改訂内容\n\n*更新情報なし*\n\n"
    
    # 2. 規定集リストの要約
    regulations = load_json_file(regulations_file) or []
    regulations_summary = create_regulation_summary(regulations)
    
    # 3. 改訂内容の詳細
    revision_details = create_revision_details(base_dir)
    
    # マークダウンの作成
    timestamp = datetime.now().strftime("%Y年%m月%d日")
    markdown_content = f"# 規定改訂レポート（{timestamp}）\n\n"
    markdown_content += update_info
    markdown_content += regulations_summary
    markdown_content += revision_details
    
    # マークダウンファイルの保存
    with open(md_report, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # PDFの生成
    try:
        pdfkit.from_string(markdown.markdown(markdown_content), pdf_report)
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("PDF generation skipped. Please install wkhtmltopdf if you want to generate PDFs.")
    
    return markdown_content