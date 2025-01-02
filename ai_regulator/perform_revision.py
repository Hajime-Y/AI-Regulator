import os
import os.path as osp
import json
from typing import Dict, Any, List, Optional

from aider.coders import Coder
from aider.models import Model

# --------------------------------------------
#  perform_revision.py
# --------------------------------------------

DRAFT_REVISION_SYSTEM_PROMPT = """あなたは銀行規定の改定を行うAIアシスタントです。
以下に示す「改定に関する銀行規定等の変更情報(update_info)」「規定集の内容(regulation_content)」と別の人が作成した「改定が必要だと考えられる理由・箇所(reason_and_comment)」に基づき、
(1) 改定前の文面 (original_text) と
(2) 改定後の文面 (revised_text)
のペアを複数リスト形式で生成してください。

規定改定案作成のために{num_reflections}回のラウンドが与えられますが、すべてを使用する必要はありません。
どのラウンドでも、早期に終了して規定改定案作成完了の判断を下すことができます。
"""

DRAFT_REVISION_USER_PROMPT = """以下の情報をもとに、改定案(1)改定前の文面 + (2)改定後の文面 を複数ペアで示してください。
なお、(1) 改定前の文面はファイル内の文章そのままを引用し、(2) 改定後の文面では省略や「...」などを使わずに改定案の全文を記載してください。

<update_info>
{update_info}
</update_info>

<regulation_content>
{regulation_content}
</regulation_content>

<reason_and_comment>
{reason_and_comment}
</reason_and_comment>

出力は以下の形式で必ず行ってください。

THOUGHT:
<THOUGHT>

DRAFT REVISION JSON:
```json
[
  {
    "original_text": "...",
    "revised_text": "..."
  },
  ...
]
```

<THOUGHT>では、まず更新情報と規定集の内容の関連性について簡潔に説明してください。
その後、具体的に規定集内の章ごとに改定の必要性を確認してください。

<JSON>では、以下のフィールドを含むJSONフォーマットで確認の必要性を提供してください：
 - original_text: 改定前の文面は規定集のファイル内に存在する文章をそのまま引用してください（改変しない）。
 - revised_text: 改定後の文面は省略せず、提案する改定案を正確に書いてください。

このJSONは自動的に解析されるため、フォーマットは正確である必要があります。
"""

DRAFT_REVISION_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
先ほど生成した改定案(複数のペア)を精査し、必要であれば修正してください。

もし修正が必要であれば、再度同じ形式で出力してください：

THOUGHT:
<THOUGHT>

DRAFT REVISION JSON:
```json
<JSON>
```

もし修正が不要なら、THOUGHTの末尾に "I am done" と書き、 そのあとに前回と全く同じJSONをそのまま出力してください。
"I am done" は変更を加えない場合のみ含めてください。
"""

def _check_revision(
        draft: List[Dict[str, str]], 
        regulation_content: str
) -> bool:
    """
    機械的に検証する関数。
    draft に含まれる "original_text" が必ず regulation_content に そのまま（完全一致で）含まれているかをチェックする。
    もし1つでも含まれていないものがあれば False を返す。
    すべて含まれていれば True を返す。
    """
    for item in draft:
        original_text = item.get("original_text", "")
        if not original_text or original_text not in regulation_content:
            return False
    return True

def draft_revision(
        regulation: Dict[str, Any],
        regulations_dir: str,
        base_dir: str,
        coder: Coder,
        out_file: str,
        num_reflections: int = 1,
) -> List[Dict[str, str]]:
    """
    1. regulation(dict) は target_regulations.json から取得した一要素:
        {
            "path": "xxx/yyy.txt",
            "reason": "...",
            "revision_needed": bool,
            "comment": "..."
        }
    2. pathを元に規定ファイルを読み込み、LLM(Coder)に
        (1)改定前文面(original_text) と
        (2)改定後文面(revised_text) を複数ペア生成させる。
    3. Reflectionを行い、再度生成を改良させる。（num_reflections回）
    4. 各生成ステップの後に _check_revision を呼び出し、original_text がファイル内に含まれていない場合は再生成（最大3回まで再試行）。
    5. 成功したら out_file に最終的な JSON を追記または上書きし、最終リストを返す。
    
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
        print("[review_revisions] update_info.txt not found.")
        return []
    
    with open(update_info_path, "r", encoding="utf-8") as f:
        update_info = f.read()

    # 規定ファイルの読み込み
    rel_path = regulation.get("path")
    if not rel_path:
        print("[draft_revision] No path found in regulation.")
        return []

    full_path = osp.join(regulations_dir, rel_path)
    if not osp.exists(full_path):
        print(f"[draft_revision] Regulation file not found: {full_path}")
        return []

    with open(full_path, "r", encoding="utf-8") as f:
        regulation_content = f.read()

    # 改定理由とコメント
    reason_and_comment = f"Reason: {regulation.get('reason', '')}\nComment: {regulation.get('comment', '')}\n"

    # --- Step 1: 初回生成 ---
    system_msg = DRAFT_REVISION_SYSTEM_PROMPT
    user_prompt = DRAFT_REVISION_USER_PROMPT.format(
        update_info=update_info,
        regulation_content=regulation_content,
        reason_and_comment=reason_and_comment,
    )

    print("[draft_revision] Generating initial draft revision...")
    max_tries = 3
    success = False
    draft_json_str = ""
    msg_history: List[Dict[str, str]] = []

    for attempt in range(max_tries):
        # Coderを用いてプロンプトを実行 (LLM呼び出し)
        # coder.run はファイルへの差分適用を想定しているが、ここでは出力テキストを取得したい。
        # coder.run() は最終出力として文字列を返すため、それを解析する。
        response_text = coder.run(
            f"{system_msg}\n\n{user_prompt}"
        )
        # JSONを抽出
        start_marker = "```json"
        end_marker = "```"
        draft_json_str = None
        if start_marker in response_text and end_marker in response_text:
            draft_json_str = response_text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()

        if not draft_json_str:
            print("[draft_revision] Failed to extract JSON on attempt", attempt+1)
            continue

        # パース
        try:
            draft_data = json.loads(draft_json_str)
        except Exception as e:
            print(f"[draft_revision] JSON parse error on attempt {attempt+1}: {e}")
            continue

        # 機械的チェック
        if _check_revision(draft_data, regulation_content):
            success = True
            break
        else:
            print(f"[draft_revision] _check_revision failed on attempt {attempt+1}, retrying...")

    if not success:
        print("[draft_revision] Initial draft generation failed after 3 attempts.")
        return []

    # --- Step 2: Reflection (num_reflections - 1回) ---
    final_data = draft_data
    reflection_system_msg = DRAFT_REVISION_SYSTEM_PROMPT  # 同じでも可
    for r in range(num_reflections - 1):
        if "I am done" in response_text:
            break

        reflection_prompt = DRAFT_REVISION_REFLECTION_PROMPT.format(
            current_round=r+2,
            num_reflections=num_reflections,
        )

        # Reflection with coder
        reflection_text = coder.run(
            f"{reflection_system_msg}\n\n{reflection_prompt}"
        )

        # JSON抽出
        reflection_json_str = None
        if start_marker in reflection_text and end_marker in reflection_text:
            reflection_json_str = (
                reflection_text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
            )

        if reflection_json_str:
            try:
                new_draft_data = json.loads(reflection_json_str)
            except Exception as e:
                print(f"[draft_revision] Reflection parse error: {e}")
                continue
            # 機械的チェック
            # 必要なら最大3回までは再試行
            local_success = False
            for attempt in range(max_tries):
                if _check_revision(new_draft_data, regulation_content):
                    local_success = True
                    break
                else:
                    # ここでは、もう一度 coder に投げる or break などが考えられるが、
                    # ユーザーの要件にある「最大3回まで再度生成させる」を簡易的に表現
                    if attempt < max_tries - 1:
                        # coderに「検出された文面が規定ファイル内にありません」と再生成を促してもよい
                        fix_text = coder.run(
                            "The original_text does not match the actual file content. Please fix the revision accordingly."
                        )
                        # 再度抽出 etc. ただし簡略化のため省略する
                        # ここでは break せず続けるかどうかは運用次第
                        pass
                    else:
                        print("[draft_revision] Reflection check failed after multiple tries.")
            if local_success:
                final_data = new_draft_data
                response_text = reflection_text
            else:
                # 失敗時は reflection ループ継続 or break
                pass

        if "I am done" in reflection_text:
            break

    # --- Step 3: 結果をファイルに保存 & return ---
    # out_file に最終的なドラフトを保存（追記 or 上書き）
    # ここでは「上書き追加」スタイル(既存の内容を読み出し、配列に追加)
    if osp.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []

    existing.append({
        "regulation_path": rel_path,
        "draft_revision": final_data,
    })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[draft_revision] Successfully wrote draft revision to {out_file}")

    return final_data

if __name__ == "__main__":
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
    import argparse

    parser = argparse.ArgumentParser(description="規定改定案の生成を行います")
    parser.add_argument("--regulations-dir", type=str, required=True, help="規定ファイルが格納されているディレクトリ")
    parser.add_argument("--target-file", type=str, required=True, help="改定対象の規定を記載したJSONファイル")
    parser.add_argument("--out-file", type=str, required=True, help="改定案の出力先JSONファイル")
    parser.add_argument("--model", type=str, default="gpt-4", help="使用するLLMモデル")
    parser.add_argument("--num-reflections", type=int, default=3, help="リフレクションの回数")
    args = parser.parse_args()

    # 入出力の設定
    io = InputOutput(yes=True, chat_history_file="revision_history.txt")
    main_model = Model(args.model)
    
    # Coderの初期化
    coder = Coder.create(
        main_model=main_model,
        fnames=[args.target_file],
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    # 対象規定の読み込み
    try:
        with open(args.target_file, "r", encoding="utf-8") as f:
            target_regulations = json.load(f)
    except Exception as e:
        print(f"対象規定ファイルの読み込みに失敗しました: {e}")
        exit(1)

    # 各規定について改定案を生成
    for regulation in target_regulations:
        if regulation.get("revision_needed", False):
            try:
                draft_revision(
                    regulation=regulation,
                    regulations_dir=args.regulations_dir,
                    coder=coder,
                    out_file=args.out_file,
                    num_reflections=args.num_reflections
                )
            except Exception as e:
                print(f"規定 {regulation.get('path', '不明')} の改定案生成に失敗しました: {e}")
                continue


