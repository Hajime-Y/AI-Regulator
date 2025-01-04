import os
import os.path as osp
import json
from typing import Dict, Any, List, Optional
from pypdf import PdfReader

from aider.coders import Coder
from aider.models import Model

# --------------------------------------------
#  perform_revision.py
# --------------------------------------------

DRAFT_REVISION_SYSTEM_PROMPT = """あなたは銀行規定の改定を行うAIアシスタントです。
以下に示す「改定に関する銀行規定等の変更情報(update_info)」「規定集の内容(regulation_content)」と別の人が作成した「改定が必要だと考えられる理由・箇所(reason_and_comment)」に基づき、
(1) セクション名
(2) 改定前の文面 (original_text) 
(3) 改定後の文面 (revised_text)
のペアを複数リスト形式で生成してください。
"""

DRAFT_REVISION_USER_PROMPT = """プロジェクトに `revision.json` ファイルを用意しました。

以下の情報をもとに、セクション名、改定前の文面、改定後の文面 をjsonのリスト形式で改定案を作成してください。
なお、文面は省略や「...」などを使わずに全文を記載してください。

<update_info>
{update_info}
</update_info>

<regulation_content>
{regulation_content}
</regulation_content>

<reason_and_comment>
{comment}
</reason_and_comment>

ファイルは必ず以下のようなjson形式である必要があります。。
```json
[
  {{
    "section_name": "...",
    "original_text": "...",
    "revised_text": "..."
  }},
  ...
]
```

ファイルは、以下のフィールドを含むJSONフォーマットで確認の必要性を提供してください：
 - section_name: セクション名はregulation_contentのセクション名を必ずそのまま引用してください（改変しない）。
 - original_text: 改定前の文面はregulation_contentの文章を改行も含めて必ずそのまま引用してください（改変しない）。
 - revised_text: 改定後の文面は省略せず、提案する改定案を正確に書いてください。

改定箇所は可能な限り必要最低限とし、現状更新情報に関連のない箇所に不必要に情報を追加しようとはしないでください。
改定箇所がない場合は、空のリストをファイルに出力してください。
 
このJSONは自動的に解析されるため、フォーマットは正確である必要があります。

必ずファイル名を最初に指定し、これらの編集を行うために *SEARCH/REPLACE* ブロックを使用してください。
"""

JSON_FORMAT_FIX_PROMPT = """{error_text}
ファイルは必ず以下のようなjson形式である必要があります。。
```json
[
  {{
    "section_name": "...",
    "original_text": "...",
    "revised_text": "..."
  }},
  ...
]
```
"""

# ==============================
# 本来はoriginal_textが元の文面と正しかったかをチェックしたかった。
# pypdfでPDFから文章を取得した場合誤字が発生し、それをLLMが勝手に修正してoriginal_textとしてします。
# これによって処理が失敗となってしまうため、現状不使用とする。
# ==============================
# def _check_revision(
#         draft: List[Dict[str, str]], 
#         regulation_content: str
# ) -> List[str]:
#     """
#     機械的に検証する関数。
#     draft に含まれる "original_text" が必ず regulation_content に そのまま（完全一致で）含まれているかをチェックする。
#     完全一致とならなかった original_text のリストを返す。
#     すべて含まれていれば空のリストを返す。
#     """
#     not_found = []
#     for item in draft:
#         original_text = item.get("original_text", "")
#         if original_text not in regulation_content.replace("\n", ""):
#             not_found.append(original_text)
#     return not_found


def draft_revision(
        regulation: Dict[str, Any],
        regulations_dir: str,
        base_dir: str,
        coder: Coder,
        revision_file: str,
) -> List[Dict[str, str]]:
    """
    1. regulation(dict) は target_regulations.json から取得した一要素:
        {
            "path": "xxx/yyy.pdf",
            "reason": "...",
            "revision_needed": bool,
            "comment": "..."
        }
    2. pathを元に規定ファイルを読み込み、LLM(Coder)に
        (1)改定前文面(original_text) と
        (2)改定後文面(revised_text) を複数ペア生成させる。
    3. 最終的な生成結果に対してファイルがjson形式となっているか確認。エラーとなる場合には再生成（最大3回まで再試行）。
    4. 成功したら最終リストを返す。
    
    戻り値:
    [
        {
            "original_text": "...",
            "revised_text": "..."
        },
        ...
    ]
    """
    # update_info.txtの読み込み
    update_info_path = osp.join(base_dir, "update_info.txt")
    if not osp.exists(update_info_path):
        print("[draft_revision] update_info.txt not found.")
        return []
    
    with open(update_info_path, "r", encoding="utf-8") as f:
        update_info = f.read()

    # 規定ファイルの読み込み
    rel_path = regulation.get("path")
    full_path = osp.join(regulations_dir, rel_path)
    if not osp.exists(full_path):
        print(f"[draft_revision] Regulation file not found: {full_path}")
        return []

    try:
        if rel_path.lower().endswith('.pdf'):
            # PDFファイルの場合
            reader = PdfReader(full_path)
            regulation_content = ''
            for page in reader.pages:
                regulation_content += page.extract_text() + '\n'
        else:
            # 通常のテキストファイルの場合
            with open(full_path, "r", encoding="utf-8") as f:
                regulation_content = f.read()
    except Exception as e:
        print(f"[draft_revision] Error reading file {rel_path}: {str(e)}")
        return []

    # 改定コメント
    comment = regulation.get('comment', '')

    # --- Step 1: 初回生成 ---
    system_msg = DRAFT_REVISION_SYSTEM_PROMPT
    user_prompt = DRAFT_REVISION_USER_PROMPT.format(
        update_info=update_info,
        regulation_content=regulation_content,
        comment=comment,
    ).replace(r"{{", "{").replace(r"}}", "}")

    # Coderを用いてプロンプトを実行 (LLM呼び出し)
    print("[draft_revision] Generating draft revision...")
    coder_out = coder.run(
        f"{system_msg}\n\n{user_prompt}"
    )

    # --- Step 2: 最終チェック ---
    max_tries = 3
    json_check_success = False
    final_checked_data = None

    # JSON形式のチェック
    for attempt in range(max_tries):
        fix_prompt = ""
        try:
            with open(revision_file, "r", encoding="utf-8") as f:
                final_checked_data = json.load(f)
            json_check_success = True
            break
        except FileNotFoundError:
            print(f"[draft_revision] Revision file not found on attempt {attempt+1}")
            break
        except json.JSONDecodeError as e:
            print(f"[draft_revision] JSON parse error on attempt {attempt+1}: {e}")
            error_text = f"JSONパースエラーが発生しました。修正してください。\n{e}\n"
            fix_prompt += JSON_FORMAT_FIX_PROMPT.format(
                error_text=error_text
            ).replace(r"{{", "{").replace(r"}}", "}")
        except Exception as e:
            print(f"[draft_revision] Unexpected error on attempt {attempt+1}: {e}")
            error_text = f"予期せぬエラーが発生しました。修正してください。\n{e}\n"
            fix_prompt += JSON_FORMAT_FIX_PROMPT.format(
                error_text=error_text
            ).replace(r"{{", "{").replace(r"}}", "}")

        # 修正を行う
        coder_out = coder.run(fix_prompt)

    if not json_check_success:
        print("[draft_revision] Final json format check failed after 3 attempts.")
        return []

    print(f"[draft_revision] Successfully wrote draft revision to {revision_file}")

    return final_checked_data

if __name__ == "__main__":
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generate revision proposals for regulations")
    parser.add_argument("--regulations-dir", type=str, required=True, help="Directory containing regulation files")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to the base directory")
    parser.add_argument("--target-file", type=str, required=False, help="JSON file containing target regulations for revision")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13", help="LLM model to use")
    args = parser.parse_args()

    # target_fileのパスを決定
    target_file = args.target_file if args.target_file else osp.join(args.base_dir, "target_regulations.json")

    # 対象規定の読み込み
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            target_regulations = json.load(f)
    except Exception as e:
        print(f"対象規定ファイルの読み込みに失敗しました: {e}")
        exit(1)

    # 各規定について改定案を生成
    for regulation in target_regulations:
        if regulation.get("revision_needed", False):
            # regulation_nameを生成（pathから拡張子なしのファイル名を取得）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            regulation_name = os.path.splitext(os.path.basename(regulation["path"]))[0] + "_" + timestamp
            folder_name = osp.join(args.base_dir, regulation_name)
            
            # フォルダ作成とファイルパス設定
            os.makedirs(folder_name, exist_ok=True)
            revision_file = osp.join(folder_name, "revision.json")

            # 入出力の設定
            io = InputOutput(yes=True, chat_history_file=f"{folder_name}/revision_history.txt")
            main_model = Model(args.model)
            
            # Coderの初期化
            coder = Coder.create(
                main_model=main_model,
                fnames=[revision_file],
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )

            # 改定案生成
            try:
                draft_res = draft_revision(
                    regulation=regulation,
                    regulations_dir=args.regulations_dir,
                    base_dir=args.base_dir,
                    coder=coder,
                    revision_file=revision_file,
                )
            except Exception as e:
                print(f"規定 {regulation.get('path', '不明')} の改定案生成に失敗しました: {e}")
                continue

            if not draft_res:
                print(f"[draft_revision] 規定 {regulation.get('path', '不明')} の改定案生成に失敗しました。")
            else:
                print(f"[draft_revision] 規定 {regulation.get('path', '不明')} の改定案を生成しました。")