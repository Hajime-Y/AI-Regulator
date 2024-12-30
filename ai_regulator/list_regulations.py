import os
import os.path as osp
import json
from typing import List, Dict, Any

# 以下、generate_ideas.py内で利用されている関数を想定。
# ここではご利用の実装に合わせて置き換えてください。
from ai_regulator.llm import get_response_from_llm, extract_json_between_markers

# ---------------------------------------------------
# 定数・システムプロンプトなど
# ---------------------------------------------------

# 規定リストアップ時のシステムプロンプト
LIST_REGULATIONS_SYSTEM_PROMPT = """あなたは銀行規定の改定を行うAIアシスタントです。
以下の目次（XML形式）と更新情報に基づき、改定が必要になりそうな規定を検討し、そのファイルパスと理由を出力してください。
ファイルパスは必ず toc.xml に記載された内容を参照し、相対パスで出力してください。
また、必ず実在するファイルのみを出力してください（この後、ファイル存在チェックを行います）。
出力はJSON形式で行い、必ず正しい形式にしてください。

規定リストアップのために{num_reflections}回のラウンドが与えられますが、すべてを使用する必要はありません。
どのラウンドでも、早期に終了して規定リストアップ完了の判断を下すことができます。
"""

# 規定リストアップ時のユーザープロンプト
LIST_REGULATIONS_USER_PROMPT = """以下のXML目次（toc）と更新情報（update_info）を参照して、
改定が必要になりそうな規定とその理由を出力してください。

<toc>
{toc_content}
</toc>

<update_info>
{update_info}
</update_info>

返答は以下のフォーマットでお願いします:

THOUGHT:
<THOUGHT>

TARGET REGULATIONS JSON:
```json
[
  {
    "path": "<相対パス>",
    "reason": "<なぜ改定が必要か>"
  },
  ...
]
```
THOUGHTでは、まず改定が必要だと考えられる業務や金融商品についてのアイデアを簡潔に説明してください。
その後、その業務や金融商品に関連する規定をリストアップします。

TARGET REGULATIONS JSONには、上記のようなフィールドを持つJSONフォーマットのリストを提供してください：
- "path": toc.xmlを参照して得た改定の必要がありそうな資料の相対パス。
- "reason": その資料を改定する必要があると想定した理由、想定される改定内容を簡潔に記載。後続の規定集確認者に伝えられる。

このJSONは自動的に解析されるため、フォーマットは正確である必要があります。
後に、このJSONを元に実際の資料を確認し、改定が必要かの判断が下されます。自信が無い資料は、改定が必要であると判断してください。
"""

# 規定リストアップ時の Reflection 用プロンプト
LIST_REGULATIONS_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
先ほど生成したJSONを精査し、見落としがあれば修正してください。自信が無い資料は、改定が必要であると判断してください。
もし修正が必要であれば、修正したJSONを形式を崩さずに出力してください。
修正が不要であれば、思考の末尾に "I am done" と書き、その後に前回と同じJSONをそのまま出力してください。

先ほどと同様のフォーマットを期待します:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
改善の余地がない場合は、THOUGHTの後に前回のJSONを完全に同じ形で繰り返し、JSONの前かつ思考の最後に "I am done" と記載してください。
"I am done" は変更を加えない場合にのみ含めてください。
"""

# 改定要否チェック時のシステムプロンプト
CHECK_REVISIONS_SYSTEM_PROMPT = """あなたは銀行規定の改定を行うAIアシスタントです。
与えられた規定集の内容と、その規定集の名前から想定された改定理由・改定内容を確認して、本当に改定が必要かどうかを判断します。

規定改定の必要性確認のために{num_reflections}回のラウンドが与えられますが、すべてを使用する必要はありません。
どのラウンドでも、早期に終了して規定改定の必要性確認完了の判断を下すことができます。
"""

# 改定要否チェック時のユーザープロンプト
CHECK_REVISIONS_USER_PROMPT = """以下の更新情報（update_info）と規定集の内容（regulation_content）、その規定集の名前から想定された改定理由・改定内容（reason）を確認して、本当に改定が必要かどうかを判断してください。
改定が必要となった場合、コメントに改定が必要な箇所を洗い出します。ただし、具体的な改定後の文面はここでは出力しません。

<update_info>
{update_info}
</update_info>

<regulation_content>
{regulation_content}
</regulation_content>

<reason>
{reason}
</reason>

返答は以下のフォーマットでお願いします:

THOUGHT:
<THOUGHT>

CHECK RESULT JSON:
```json
<JSON>
```

<THOUGHT>では、まず更新情報と規定集の内容の関連性について簡潔に説明してください。
その後、具体的に規定集内の章ごとに改定の必要性を確認してください。

<JSON>では、以下のフィールドを含むJSONフォーマットで確認の必要性を提供してください：
- "revision_needed": 改定が必要かどうか。（True or False）
- "comment": 改定の必要性に関するコメント。改定が必要となった場合、コメントに改定が必要な箇所を洗い出す。後続の改定実施者に伝えられる。

このJSONは自動的に解析されるため、フォーマットは正確である必要があります。
規定改定の必要性確認のために{num_reflections}回のラウンドがありますが、すべてを使用する必要はありません。
"""

# 改定要否チェック時の Reflection 用プロンプト
CHECK_REVISIONS_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
先ほど生成したJSONを精査し、見落としや間違えがあれば修正してください。
特に改定が必要と判断した場合、改定が必要な箇所を十全に洗い出してください。
もし修正が必要であれば、修正したJSONを形式を崩さずに出力してください。
修正が不要であれば、思考の末尾に "I am done" と書き、その後に前回と同じJSONをそのまま出力してください。

先ほどと同様のフォーマットを期待します:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
改善の余地がない場合は、THOUGHTの後に前回のJSONを完全に同じ形で繰り返し、JSONの前かつ思考の最後に "I am done" と記載してください。
"I am done" は変更を加えない場合にのみ含めてください。
"""

# ---------------------------------------------------
# 1. list_regulations
# ---------------------------------------------------
def list_regulations(
    regulations_dir: str,
    base_dir: str,
    client,
    model,
    num_reflections: int = 1,
):
    """
    1. regulations_dir下のtoc.xmlとbase_dir下のupdate_info.txtを読み込み、
       LLMに「改定が必要そうな規定集のpathとreason」を出力させる。
    2. Reflection（num_reflections回）を行い、最終的なJSONリストを確定する。
       各Reflectionラウンドでは、修正が不要なら "I am done" と出力して終了。
    3. 規定ファイルの存在チェックを行い、実在しないファイルはリストから除外する。
    4. 結果を target_regulations.json として保存し、リストを返す。
    """
    toc_path = osp.join(regulations_dir, "toc.xml")
    update_info_path = osp.join(base_dir, "update_info.txt")
    target_regulations_file = osp.join(base_dir, "target_regulations.json")

    if not osp.exists(toc_path):
        raise FileNotFoundError(f"toc.xml not found in {regulations_dir}")
    if not osp.exists(update_info_path):
        raise FileNotFoundError(f"update_info.txt not found in {base_dir}")

    with open(toc_path, "r", encoding="utf-8") as f:
        toc_content = f.read()
    with open(update_info_path, "r", encoding="utf-8") as f:
        update_info = f.read()

    # --- Step 1: 最初の呼び出し ---
    msg_history = []
    system_message = LIST_REGULATIONS_SYSTEM_PROMPT.format(num_reflections=num_reflections)
    user_message = LIST_REGULATIONS_USER_PROMPT.format(
        toc_content=toc_content,
        update_info=update_info,
    )

    print("[list_regulations] Generating target regulations (initial call)...")
    text, msg_history = get_response_from_llm(
        user_message,
        client=client,
        model=model,
        system_message=system_message,
        msg_history=msg_history,
    )

    # JSON抽出
    raw_json = extract_json_between_markers(text)
    if raw_json is None:
        print("[list_regulations] Failed to extract JSON on first attempt.")
        return []

    # --- Step 2: Reflection ---
    final_json_str = raw_json
    if num_reflections > 1:
        for j in range(num_reflections - 1):
            if "I am done" in text:
                break  # Reflection打ち切り

            # Reflection用プロンプト
            reflection_prompt = LIST_REGULATIONS_REFLECTION_PROMPT.format(
                current_round=j+2,
                num_reflections=num_reflections,
            )
            reflection_text, msg_history = get_response_from_llm(
                reflection_prompt,
                client=client,
                model=model,
                system_message=system_message,
                msg_history=msg_history,
            )
            new_json = extract_json_between_markers(reflection_text)
            if new_json is not None:
                final_json_str = new_json
            text = reflection_text
            if "I am done" in text:
                break

    # --- Step 3: JSONパース & ファイル存在チェック ---
    try:
        proposed_list = json.loads(final_json_str)
    except Exception as e:
        print("[list_regulations] Error parsing final JSON:", e)
        return []

    final_list = []
    for item in proposed_list:
        rel_path = item.get("path")
        reason = item.get("reason", "")
        if not rel_path:
            continue
        full_path = osp.join(regulations_dir, rel_path)
        if osp.exists(full_path):
            final_list.append({"path": rel_path, "reason": reason})
        else:
            print(f"[list_regulations] File does not exist: {rel_path} (skipped)")

    # --- Step 4: 保存 & 結果を返す ---
    with open(target_regulations_file, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"[list_regulations] Done. Saved {len(final_list)} targets to {target_regulations_file}")
    return final_list


# ---------------------------------------------------
# 2. check_revisions
# ---------------------------------------------------
def check_revisions(
    target_regulations: List[Dict[str, Any]],
    regulations_dir: str,
    base_dir: str,
    client,
    model,
    num_reflections: int = 1,
):
    """
    1. target_regulations（なければ base_dir下のtarget_regulations.json から取得）を読み込み、
       各規定ファイルの内容・更新情報(update_info.txt)・"reason"を LLM に渡して
       "revision_needed" (bool) と "comment" (str) を判断させる。
    2. Reflection（num_reflections回）を行い、最終的なJSONを確定する。
    3. 結果を各規定に付与してリストを返す。
    """
    if not target_regulations:
        target_regulations_file = osp.join(base_dir, "target_regulations.json")
        if not osp.exists(target_regulations_file):
            print("[check_revisions] No target_regulations provided and no file found.")
            return []
        with open(target_regulations_file, "r", encoding="utf-8") as f:
            target_regulations = json.load(f)

    update_info_path = osp.join(base_dir, "update_info.txt")
    if not osp.exists(update_info_path):
        print("[check_revisions] update_info.txt not found.")
        return []
    with open(update_info_path, "r", encoding="utf-8") as f:
        update_info = f.read()

    updated_list = []
    for reg in target_regulations:
        rel_path = reg.get("path")
        reason = reg.get("reason", "")
        full_path = osp.join(regulations_dir, rel_path) if rel_path else ""

        if not rel_path or not osp.exists(full_path):
            print(f"[check_revisions] File not found or path is empty: {rel_path}")
            reg["revision_needed"] = False
            reg["comment"] = "File not found or path is empty."
            updated_list.append(reg)
            continue

        print(f"[check_revisions] Starting revision check for: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            regulation_content = f.read()

        # --- Step 1: 初回呼び出し ---
        msg_history = []
        system_message = CHECK_REVISIONS_SYSTEM_PROMPT.format(num_reflections=num_reflections)
        user_message = CHECK_REVISIONS_USER_PROMPT.format(
            update_info=update_info,
            regulation_content=regulation_content,
            reason=reason,
            num_reflections=num_reflections,
        )

        print(f"[check_revisions] Generating check results (initial call) for: {rel_path}")
        text, msg_history = get_response_from_llm(
            user_message,
            client=client,
            model=model,
            system_message=system_message,
            msg_history=msg_history,
        )
        raw_json = extract_json_between_markers(text)
        if raw_json is None:
            print(f"[check_revisions] Failed to extract JSON for: {rel_path}")
            reg["revision_needed"] = False
            reg["comment"] = "Failed to parse JSON."
            updated_list.append(reg)
            continue

        # --- Step 2: Reflection ---
        final_json_str = raw_json
        if num_reflections > 1:
            for j in range(num_reflections - 1):
                if "I am done" in text:
                    print(f"[check_revisions] Reflection terminated early for: {rel_path}")
                    break  # Reflection termination

                print(f"[check_revisions] Performing reflection round {j+2} for: {rel_path}")
                reflection_prompt = CHECK_REVISIONS_REFLECTION_PROMPT.format(
                    current_round=j+2,
                    num_reflections=num_reflections,
                )
                reflection_text, msg_history = get_response_from_llm(
                    reflection_prompt,
                    client=client,
                    model=model,
                    system_message=system_message,
                    msg_history=msg_history,
                )
                new_json = extract_json_between_markers(reflection_text)
                if new_json is not None:
                    final_json_str = new_json
                text = reflection_text
                if "I am done" in text:
                    print(f"[check_revisions] Reflection completed for: {rel_path}")
                    break

        # --- Step 3: 最終JSONをパース & 書き込み ---
        try:
            check_result = json.loads(final_json_str)
            reg["revision_needed"] = check_result.get("revision_needed", False)
            reg["comment"] = check_result.get("comment", "")
            print(f"[check_revisions] Revision needed: {reg['revision_needed']} for: {rel_path}")
        except Exception as e:
            reg["revision_needed"] = False
            reg["comment"] = f"Error parsing final JSON: {e}"
            print(f"[check_revisions] Error parsing JSON for: {rel_path} - {e}")

        updated_list.append(reg)

    return updated_list