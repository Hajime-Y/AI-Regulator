import os
import os.path as osp
import json
import time
from typing import Dict, Any, List, Optional

from aider.coders import Coder
from ai_regulator.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

# --------------------------------------------
#  perform_review.py
# --------------------------------------------

REVIEW_REVISION_SYSTEM_PROMPT = """あなたは銀行規定の改定案をレビューするAI銀行員です。
判断には批判的かつ慎重であるべきです。改定案の質が低いか、信頼度(Confidence)が低い場合は、低い評価点をつけます。
「規定集の内容(regulation_content)」と「更新情報(update_info)」、「改定案(draft_revision)」が与えられるます。
それを踏まえて改定案のレビューを行います。
"""

REVIEW_REVISION_USER_PROMPT = """以下の情報を参照し、改定案に対するレビューを行ってください。

<regulation_content>
{regulation_content}
</regulation_content>

<update_info>
{update_info}
</update_info>

<draft_revision>
{draft_revision}
</draft_revision>

## レビューフォーム
規定改定案に対するレビュー時に考慮すべき指針を以下に示します。

1. フォーマット評価(Format Check)：改定案は規定集の形式を適切に維持していますか？
   - 文書構造は一貫していますか？
   - 用語の使用は統一されていますか？
   - 規定特有の表現や形式は保持されていますか？

2. 削除チェック(Removal Check)：不要な削除がないことを確認してください
   - 重要な条項や文言が欠落していませんか？
   - 必要な参照や関連規定への言及は維持されていますか？
   - 削除された部分は適切に代替されていますか？

3. 一貫性評価(Consistency Check)：規定集の役割・目的との整合性を評価してください
   - 規定の本来の意図は保持されていますか？

4. 完全性評価(Completeness Check)：改定理由と更新者のコメントの反映を確認してください
   - 改定理由に示された課題は適切に対処されていますか？
   - 更新者のコメントは十分に考慮されていますか？
   - 改定の意図が明確に反映されていますか？

5. 総合評価(Overall)：以下の基準に基づいて改定案の総合評価を提供してください：
   5: 完璧な改定案：改定は完璧で、規定の目的を完全に達成し、形式も内容も申し分ない
   4: 優れた改定案：改定は優れたものであり、規定の目的を十分に達成し、形式・内容ともに高水準
   3: 良好な改定案：改定は良好で、規定の目的を達成し、重大な問題がない
   2: 要改善の改定案：改定は問題や不明確な点があり、修正が必要
   1: 不適切な改定案：重大な問題があり、目的達成が困難

6. 信頼度(Confidence)：評価の確信度を1から5の尺度で示してください
   5: 評価について絶対的な確信がある。規定内容や改定に関わる変更情報に精通しており、詳細まで慎重に確認した。
   4: 評価についてかなりの確信はあるが、絶対的ではない。可能性は低いが、規定内容や改定に関わる変更情報の一部を理解していない可能性がある。
   3: 評価について確信がある。ただし、規定内容や改定に関わる変更情報の一部を理解していない可能性がある。
   2: 評価を擁護する意思はあるが、規定内容や改定に関わる変更情報の中心的な部分を理解していない。
   1: 評価は推測を含むものである。規定内容や改定に関わる変更情報が専門分野外であり、ほとんど理解できていない。

出力は以下の形式で必ず行ってください:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

<THOUGHT>では、まず評価に対するあなたの直感的な考えと推論について簡潔に説明してください。
レビューの高レベルな論点、必要な判断、望ましい結果について詳しく述べてください。
ここでは一般的なコメントは避け、現在の規定改定案に特化した具体的な内容を書いてください。

<JSON>では、以下のフィールドを含むJSONフォーマットでレビューを提供してください：
 - "format_check": 規定集の形式を適切に維持しているか
 - "removal_check": 不要な削除がないか
 - "consistency_check": 規定集の役割・目的との整合性
 - "completeness_check": 変更情報の内容を漏れなく改定に反映しているか
 - "overall": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）
 - "confidence": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）
 - "comment": 改定案に対するコメントや建設的なフィードバック

このJSONは自動的に解析されるため、フォーマットは正確である必要があります。
"""

REVIEW_REVISION_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
まず、あなたが作成したレビューの正確性と妥当性を慎重に検討してください。
銀行規定集の改定案を評価する上で重要だと考えられる他の要因も含めてください。
レビューが明確で簡潔であり、JSONが正しい形式であることを確認してください。
不必要に複雑にしないでください。
次の試みでは、レビューの改善と洗練を心がけてください。
明らかな問題がない限り、元のレビューの本質は維持してください。

以前と同じ形式で回答してください:
THOUGHT:
<THOUGHT>

REVIEW JSON:
<JSON>

もし修正が不要なら、THOUGHTの末尾に "I am done" と書き、そのあとに前回と全く同じJSONをそのまま出力してください。
"I am done" は変更を加えない場合のみ含めてください。
"""

IMPROVE_REVISION_SYSTEM_PROMPT = """あなたは銀行規定の改定を行うAIアシスタントです。
以下に示す「改定案（original_text → revised_text の複数ペア）」と「レビュー内容」を踏まえて、改定案を再編集してください。
特にレビューで指摘された問題点を改善し、より優れた改定案にしてください。なお、
(1)改定前文面(original_text) は必ず規定ファイルそのままを保持してください。
(2)改定後文面(revised_text) はレビューコメントの内容を適切に取り入れて修正してください。

出力形式は以下のとおりにしてください:
THOUGHT:
<THOUGHT>

IMPROVED REVISION JSON:
```json
[
  {
    "original_text": "...",
    "revised_text": "..."
  },
  ...
]
```
"""

IMPROVE_REVISION_USER_PROMPT = """以下の情報を基に、改定案の改善を行ってください。

<current_draft_revision>
{current_draft_revision}
</current_draft_revision>

<review_result>
{review_result}
</review_result>

改定案(1)original_textは規定ファイルの内容をそのまま、 (2)revised_textはレビュー指摘を反映した修正案を提示してください。
出力フォーマット:

THOUGHT:
<THOUGHT>

IMPROVED REVISION JSON:
```json
[
  {{
    "original_text": "...",
    "revised_text": "..."
  }},
  ...
]
```
"""

IMPROVE_REVISION_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
先ほど生成した改定案（修正後）を再度検討し、必要があれば修正してください。

修正が必要なら、以下の形式で出力してください:
THOUGHT:
<THOUGHT>

IMPROVED REVISION JSON:
```json
<JSON>
```

もし修正が不要なら、THOUGHTの末尾に "I am done" と書き、そのあとに前回と全く同じJSONをそのまま出力してください。
"I am done" は変更を加えない場合のみ含めてください。
"""


def review_revision(
        regulation: Dict[str, Any],
        draft_revision: List[Dict[str, str]], 
        regulations_dir: str, 
        base_dir: str,
        model: str,
        client: Any,
        num_reflections: int = 1,
        num_reviews_ensemble: int = 1,
        temperature: float = 0.7,
        msg_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    改定案のレビューを行う。
     - regulation(dict): target_regulations.json のエントリ
     - draft_revision: draft_revision() で生成された改定案のリスト
        [ { "original_text": "...", "revised_text": "..." }, ... ]
    
    手順:
        1. 規定ファイルを読み込み、reason/comment などをまとめる
        2. LLMにレビューさせる(Reflectionあり)
        3. 最終的なレビューJSONを返す
    """
    # 規定ファイルの読み込み
    rel_path = regulation.get("path")
    if not rel_path:
        print("[review_revision] No path found in regulation.")
        return {}

    full_path = os.path.join(regulations_dir, rel_path)
    if not os.path.exists(full_path):
        print(f"[review_revision] Regulation file not found: {full_path}")
        return {}

    with open(full_path, "r", encoding="utf-8") as f:
        regulation_content = f.read()

    # update_info.txtの読み込み
    update_info_path = osp.join(base_dir, "update_info.txt")
    if not osp.exists(update_info_path):
        print("[review_revisions] update_info.txt not found.")
        return []
    
    with open(update_info_path, "r", encoding="utf-8") as f:
        update_info = f.read()

    # JSONとして使いやすい形にしておく
    draft_json_str = json.dumps(draft_revision, ensure_ascii=False, indent=2)

    base_prompt = REVIEW_REVISION_USER_PROMPT.format(
        regulation_content=regulation_content,
        update_info=update_info,
        draft_revision=draft_json_str,
    )

    print("[review_revision] Generating initial review...")

    # --- Step 1: 初回呼び出し ---
    if num_reviews_ensemble > 1:
        # 複数のレビューを生成
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=REVIEW_REVISION_SYSTEM_PROMPT,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
            n_responses=num_reviews_ensemble,
        )
        
        # 各レビューからJSONを抽出
        parsed_reviews = []
        for idx, rev in enumerate(llm_reviews):
            try:
                review_json = extract_json_between_markers(rev)
                if review_json:
                    parsed_reviews.append(review_json)
            except Exception as e:
                print(f"[review_revision] Ensemble review {idx} failed: {e}")
        
        # レビューの集約
        final_review = aggregate_reviews(parsed_reviews)
        
        # メッセージ履歴の更新
        msg_history = msg_histories[0][:-1]
        msg_history += [{
            "role": "assistant",
            "content": f"""
THOUGHT:
{num_reviews_ensemble}人のレビュアーの意見を集約しました。

REVIEW JSON:
```json
{json.dumps(final_review, ensure_ascii=False, indent=2)}
```
"""
        }]
    else:
        # 単一レビューの生成
        response_text, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=REVIEW_REVISION_SYSTEM_PROMPT,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        final_review = extract_json_between_markers(response_text)
        if not final_review:
            print("[review_revision] Failed to extract JSON on initial review.")
            return {}

    # --- Step 2: Reflection ---
    for r in range(num_reflections - 1):
        if "I am done" in response_text:
            break

        reflection_prompt = REVIEW_REVISION_REFLECTION_PROMPT.format(
            current_round=r+2,
            num_reflections=num_reflections,
        )
        reflection_output, msg_history = get_response_from_llm(
            reflection_prompt,
            model=model,
            client=client,
            system_message=REVIEW_REVISION_SYSTEM_PROMPT,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        reflection_json = extract_json_between_markers(reflection_output)
        if reflection_json:
            final_review = reflection_json
            response_text = reflection_output

        if "I am done" in reflection_output:
            break

    return final_review

def aggregate_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """複数のレビューを1つに集約する"""
    if not reviews:
        return {}
    
    # 最初のレビューをベースとして使用
    aggregated = reviews[0].copy()
    
    # 数値フィールドの平均を計算
    numeric_fields = ["overall", "confidence"]
    for field in numeric_fields:
        values = [r[field] for r in reviews if field in r]
        if values:
            aggregated[field] = round(sum(values) / len(values))
    
    # テキストフィールドの結合
    text_fields = ["format_check", "removal_check", "consistency_check", "completeness_check", "comment"]
    for field in text_fields:
        if field in aggregated:
            values = [r[field] for r in reviews if field in r]
            aggregated[field] = "\n".join(f"レビュアー{i+1}: {v}" for i, v in enumerate(values))
    
    return aggregated


def improve_revision(
        current_draft: List[Dict[str, str]], 
        review_data: Dict[str, Any], 
        coder: Coder, 
        num_reflections: int = 1,
) -> List[Dict[str, str]]:
    """
    review_revision の結果 (review_data) を踏まえて改定案を再編集する。
    
    手順:
    1. LLM に「current_draft_revision」(オリジナルの改定案) と「review_result」(上記レビューJSON) を渡して
        改定案を修正させる(Reflectionあり)
    2. 最終的な改定案のリストを返す

    戻り値:
    [
        { "original_text": "...", "revised_text": "..." },
        ...
    ]
    """
    current_draft_str = json.dumps(current_draft, ensure_ascii=False, indent=2)
    review_str = json.dumps(review_data, ensure_ascii=False, indent=2)

    system_msg = IMPROVE_REVISION_SYSTEM_PROMPT
    user_msg = IMPROVE_REVISION_USER_PROMPT.format(
        current_draft_revision=current_draft_str,
        review_result=review_str,
    )

    print("[improve_revision] Generating improved revision...")

    # --- Step 1: 初回呼び出し ---
    response_text = coder.run(f"{system_msg}\n\n{user_msg}")
    improved_json = extract_json_between_markers(response_text)
    if not improved_json:
        print("[improve_revision] Failed to extract JSON on initial improvement call.")
        return current_draft  # fallback

    # --- Step 2: Reflection ---
    final_improved = improved_json
    for r in range(num_reflections - 1):
        if "I am done" in response_text:
            break

        reflection_prompt = IMPROVE_REVISION_REFLECTION_PROMPT.format(
            current_round=r+2,
            num_reflections=num_reflections,
        )
        reflection_text = coder.run(f"{system_msg}\n\n{reflection_prompt}")
        new_json = extract_json_between_markers(reflection_text)
        if new_json:
            final_improved = new_json
            response_text = reflection_text

        if "I am done" in reflection_text:
            break

    return final_improved