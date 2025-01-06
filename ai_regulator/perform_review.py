import os
import os.path as osp
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional
from pypdf import PdfReader

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
「規定集の内容(regulation_content)」と「変更情報(update_info)」、「改定案(draft_revision)」が与えられるます。
それを踏まえて改定案のレビューを行います。
"""

REVIEW_FORM = """

## レビューフォーム
規定改定案に対するレビュー時に考慮すべき指針を以下に示します。

1. フォーマット評価(Format Check)：改定案は規定集の形式を適切に維持しているか確認して建設的なフィードバックを提供してください
   - 文書構造は一貫していますか？
   - 用語の使用は統一されていますか？
   - 規定特有の表現や形式は保持されていますか？

2. 削除チェック(Removal Check)：不要な削除がないことを確認して建設的なフィードバックを提供してください
   - 重要な条項や文言が欠落していませんか？
   - 必要な参照や関連規定への言及は維持されていますか？
   - 削除された部分は適切に代替されていますか？

3. 一貫性評価(Consistency Check)：規定集の役割・目的との整合性を確認して建設的なフィードバックを提供してください
   - 規定の本来の意図は保持されていますか？
   - 元の規定文にない情報を不必要に追加していませんか？
   - 規定は現在の状態のみを記述し、変更過程や移行期間に関する説明を避けていますか？

4. 完全性評価(Completeness Check)：改定理由と更新者のコメントの反映を確認して建設的なフィードバックを提供してください
   - 改定理由に示された課題は適切に対処されていますか？
   - 更新者のコメントは十分に考慮されていますか？
   - 改定の意図が明確に反映されていますか？

5. 記載原則評価(Documentation Check)：規定文書の記載原則が遵守されているか確認して建設的なフィードバックを提供してください
   - 現在の状態のみを記述し、変更過程や一時的な状態の説明を避けていますか？
   - 「一部のみ変更」「当面の間」「なお～のままです」などの移行的な表現を使用していませんか？
   - 移行期間や変更に関する説明を規定本文に含めていませんか？

6. 改定必要性評価(Necessity Check)：改定の判断基準に照らして必要な改定のみが行われているか確認して建設的なフィードバックを提供してください
   - 情報の更新：既存の規定内に関連する記述がある場合のみ改定していますか？
     例：手数料率「年1.5%」→「年2.0%」への変更は、既存の手数料率記載箇所のみ更新
   - 情報の追加：既存の規定内に関連する記述が全くない場合のみ改定していますか？
     例：新設の「インターネットバンキング」手数料について、新規条項として追加
   - 情報の削除：変更情報に基づく既存ルールの廃止の場合のみ削除していますか？
     例：「FAXでの申込受付」廃止に伴い、FAX関連の記述のみ削除
   - 不要な改定の禁止：変更情報に直接関係のない箇所への改定は含まれていませんか？
     例：手数料改定時に申込方法の説明文を改善するのは不可

7. 総合評価(Overall)：以下の基準に基づいて改定案の総合評価を提供してください
   5: 完璧な改定案：改定は完璧で、規定の目的を完全に達成し、形式も内容も申し分ない
   4: 優れた改定案：改定は優れたものであり、規定の目的を十分に達成し、形式・内容ともに高水準
   3: 良好な改定案：改定は良好で、規定の目的を達成し、重大な問題がない
   2: 要改善の改定案：改定は問題や不明確な点があり、修正が必要
   1: 不適切な改定案：重大な問題があり、目的達成が困難

8. 信頼度(Confidence)：評価の確信度を1から5の尺度で示してください
   5: 評価について絶対的な確信がある。規定内容や改定に関わる変更情報に精通しており、詳細まで慎重に確認した。
   4: 評価についてかなりの確信はあるが、絶対的ではない。可能性は低いが、規定内容や改定に関わる変更情報の一部を理解していない可能性がある。
   3: 評価について確信がある。ただし、規定内容や改定に関わる変更情報の一部を理解していない可能性がある。
   2: 評価を擁護する意思はあるが、規定内容や改定に関わる変更情報の中心的な部分を理解していない。
   1: 評価は推測を含むものである。規定内容や改定に関わる変更情報が専門分野外であり、ほとんど理解できていない。
"""

REVIEW_REVISION_USER_PROMPT = REVIEW_FORM + """以下の情報を参照し、改定案に対するレビューを行ってください。

<regulation_content>
{regulation_content}
</regulation_content>

<update_info>
{update_info}
</update_info>

<draft_revision>
{draft_revision}
</draft_revision>

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
 - "format_check": 規定集の形式を適切に維持しているかの確認結果(string)
 - "removal_check": 不要な削除がないかの確認結果(string)
 - "consistency_check": 規定集の役割・目的との整合性の確認結果(string)
 - "completeness_check": 変更情報の内容を漏れなく改定に反映しているかの確認結果(string)
 - "documentation_check": 規定文書の記載原則が遵守されているかの確認結果(string)
 - "necessity_check": 改定の判断基準に照らして必要な改定のみが行われているかの確認結果(string)
 - "overall": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）(int)
 - "confidence": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）(int)
 - "comment": 改定案に対するコメントや建設的なフィードバック(string)

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
規定とは、銀行の取引やサービスに関する現在のルール・規則を定めた文書です。

以下に示す「改定に関する銀行規定等の変更情報(update_info)」「規定集の内容(regulation_content)」と別の人が作成した「改定が必要だと考えられる理由・箇所(reason_and_comment)」に基づき、
(1) セクション名
(2) 改定前の文面 (original_text) 
(3) 改定後の文面 (revised_text)
のペアを複数リスト形式で渡されます。レビュー結果を元にこれを改善してください。

重要な注意事項：
- 規定は現在の状態のみを記述してください。「一部のみ変更」「当面の間」「なお～のままです」など、変更過程や状態の一時性を説明する文言は含めないでください。
- 例：
  × 「なお、支店窓口での手続きは従来通りとなります。」
  ○ 「手続きは、インターネットバンキングまたは支店窓口で受け付けます。」
- 移行期間や変更に関する説明は、規定本文ではなく別途の通知文書で伝えるべき内容です。
"""

IMPROVE_REVISION_USER_PROMPT = """プロジェクトに `revision.json` ファイルを用意しました。

以下のレビュー結果をもとに、セクション名、改定前の文面、改定後の文面 をjsonのリスト形式で構成された改定案を改善してください。
なお、文面は省略や「...」などを使わずに全文を記載してください。

<review_result>
{review_result}
</review_result>

ファイルは必ず以下のようなjson形式である必要があります。
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
 - section_name: セクション名は元の改定案のセクション名をそのまま引用してください（改変しない）。
 - original_text: 改定前の文面は元の改定案のoriginal_textをそのまま引用してください（改変しない）。
 - revised_text: 改定後の文面は省略せず、レビュー結果を反映した改定案を正確に書いてください。

改定箇所は可能な限り必要最低限とし、レビュー結果に関連のない箇所に不必要に情報を追加しようとはしないでください。

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

META_REVIEWER_SYSTEM_PROMPT = """あなたは銀行規定の改定案をレビューする上級AIレビュアーです。
{reviewer_count}人のレビュアーからのフィードバックを分析し、より包括的で客観的なメタレビューを作成することが役割です。
あなたの仕事は、複数のレビューを同じフォーマットで1つのメタレビューにまとめることです。
判断は批判的かつ慎重であり、全レビュアーの意見を尊重しながらそれらを統合した最終的な評価を提供してください。
"""

META_REVIEWER_USER_PROMPT = REVIEW_FORM + """
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
 - "format_check": 規定集の形式を適切に維持しているかの確認結果(string)
 - "removal_check": 不要な削除がないかの確認結果(string)
 - "consistency_check": 規定集の役割・目的との整合性の確認結果(string)
 - "completeness_check": 変更情報の内容を漏れなく改定に反映しているかの確認結果(string)
 - "documentation_check": 規定文書の記載原則が遵守されているかの確認結果(string)
 - "necessity_check": 改定の判断基準に照らして必要な改定のみが行われているかの確認結果(string)
 - "overall": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）(int)
 - "confidence": 1から5の評価（低い、中程度、高い、非常に高い、絶対的）(int)
 - "comment": 改定案に対するコメントや建設的なフィードバック(string)

このJSONは自動的に解析されるため、フォーマットは正確である必要があります。
"""

def get_meta_review(
    model: str,
    client: Any,
    temperature: float,
    reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    複数のレビューを分析し、統合されたメタレビューを生成する

    Args:
        model: 使用するLLMモデル
        client: LLMクライアント
        temperature: 生成時の温度パラメータ
        reviews: レビューのリスト

    Returns:
        統合されたメタレビュー
    """
    # レビューをテキスト形式に変換
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
レビュー {i + 1}/{len(reviews)}:
```json
{json.dumps(r, ensure_ascii=False, indent=2)}
```
"""

    # プロンプトの構築
    system_prompt = META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews))
    base_prompt = META_REVIEWER_USER_PROMPT + review_text

    # メタレビューの生成
    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )

    # JSONの抽出
    meta_review = extract_json_between_markers(llm_review)

    return meta_review

def review_revision(
        regulation: Dict[str, Any],
        draft_revision: List[Dict[str, str]], 
        regulations_dir: str, 
        base_dir: str,
        model: str,
        client: Any,
        num_reflections: int = 1,
        num_reviews_ensemble: int = 1,
        temperature: float = 0.75,
        msg_history: Optional[List[Dict[str, str]]] = None,
        return_msg_history=False,
) -> Dict[str, Any]:
    """
    銀行規定の改定案に対してAIレビューを実施する

    手順:
    1. 規定ファイルと変更情報を読み込み、レビュー用の情報を準備
    2. LLMを用いて初回レビューを生成
        - 単一レビュー: 1回のレビュー生成
        - アンサンブル: 複数のレビューを生成し、メタレビューで統合
    3. レビュー結果に対して指定回数のリフレクション（再検討）を実施
    4. 最終的なレビュー結果を返す
    """
    # --- Step 1: 規定ファイルと変更情報の読み込み ---
    rel_path = regulation.get("path")
    if not rel_path:
        print("[review_revision] No path found in regulation.")
        return {}

    full_path = os.path.join(regulations_dir, rel_path)
    if not os.path.exists(full_path):
        print(f"[review_revision] Regulation file not found: {full_path}")
        return {}

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
        print(f"[review_revision] Error reading file {rel_path}: {str(e)}")
        return {}

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

    # --- Step 2: 初回レビューの生成 ---
    if num_reviews_ensemble > 1:
        # 複数のレビューを生成してメタレビューに統合
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=REVIEW_REVISION_SYSTEM_PROMPT,
            print_debug=False,
            msg_history=msg_history,
            # Higher temperature to encourage diversity.
            temperature=0.75,
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

        # 各レビューの内容を出力
        for idx, review in enumerate(parsed_reviews):
            print(f"\n[review_revision] Review {idx + 1}/{len(parsed_reviews)}:")
            print(json.dumps(review, ensure_ascii=False, indent=2))
        
        # メタレビューの生成
        review = get_meta_review(model, client, temperature, parsed_reviews)

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # スコアの制限と平均値の計算
        for score, limits in [
            ("overall", (1, 5)),
            ("confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))
        
        # メタレビューの内容を出力
        print(f"[review_revision] Meta-review:")
        print(json.dumps(review, ensure_ascii=False, indent=2))

        # メッセージ履歴の更新
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
THOUGHT:
以前に取得した{num_reviews_ensemble}人のレビュアーの意見を集約していきます。

REVIEW JSON:
```json
{json.dumps(review, ensure_ascii=False, indent=2)}
```
""",
            }
        ]
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
        review = extract_json_between_markers(response_text)

    # --- Step 3: Reflectionの実施 ---
    for r in range(num_reflections - 1):
        reflection_prompt = REVIEW_REVISION_REFLECTION_PROMPT.format(
            current_round=r+2,
            num_reflections=num_reflections,
        )
        reflection_output, msg_history = get_response_from_llm(
            reflection_prompt,
            client=client,
            model=model,
            system_message=REVIEW_REVISION_SYSTEM_PROMPT,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(reflection_output)

        # リフレクションの内容を出力
        print(f"[review_revision] Reflection {r+2}/{num_reflections}:")
        print(json.dumps(review, ensure_ascii=False, indent=2))

        if "I am done" in reflection_output:
            break

    # --- Step 4: 最終結果の返却 ---
    if not review:
        print("[review_revision] Failed to extract JSON on initial review.")
        return {}
    
    print(f"[review_revision] Successfully generated review")

    if return_msg_history:
        return review, msg_history
    else:
        return review


def format_review_result(review_data: Dict[str, Any]) -> str:
    """レビュー結果をテキスト形式にフォーマットする"""
    result = []
    result.append("## レビュー結果")
    
    # 各チェック項目
    checks = {
        "フォーマットチェック": "format_check",
        "削除チェック": "removal_check",
        "一貫性チェック": "consistency_check",
        "完全性チェック": "completeness_check",
        "記載原則チェック": "documentation_check",
        "改定必要性チェック": "necessity_check"
    }
    
    for title, key in checks.items():
        if key in review_data:
            result.append(f"\n### {title}")
            result.append(review_data[key])
    
    # 評価とコメント
    result.append("\n### 総合評価")
    result.append(f"評価: {review_data.get('overall', '不明')}/5")
    result.append(f"信頼度: {review_data.get('confidence', '不明')}/5")
    
    if "comment" in review_data:
        result.append("\n### コメント")
        result.append(review_data["comment"])
    
    return "\n".join(result)

def improve_revision(
        current_draft: List[Dict[str, str]], 
        review_data: Dict[str, Any], 
        coder: Coder,
        revision_file: str,
) -> List[Dict[str, str]]:
    """
    review_revision の結果 (review_data) を踏まえて改定案を再編集する。
    
    手順:
    1. LLMに改定案の修正を指示
    2. 最終的な生成結果に対してファイルがjson形式となっているか確認
    3. JSON形式エラーの場合は修正を試行（最大3回）
    4. 成功したら最終リストを返す(失敗したらcurrent_draftを返す)
    """
    # --- Step 1: 初回生成 ---
    # レビュー結果をテキスト形式に変換
    review_result_text = format_review_result(review_data)
    
    system_prompt = IMPROVE_REVISION_SYSTEM_PROMPT
    user_prompt = IMPROVE_REVISION_USER_PROMPT.format(
        review_result=review_result_text,
    ).replace(r"{{", "{").replace(r"}}", "}")

    # Coderを用いてプロンプトを実行
    print("[improve_revision] Generating improved revision...")
    coder_out = coder.run(
        f"{system_prompt}\n\n{user_prompt}"
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
            print(f"[improve_revision] Revision file not found on attempt {attempt+1}")
            break
        except json.JSONDecodeError as e:
            print(f"[improve_revision] JSON parse error on attempt {attempt+1}: {e}")
            error_text = f"JSONパースエラーが発生しました。修正してください。\n{e}\n"
            fix_prompt += JSON_FORMAT_FIX_PROMPT.format(
                error_text=error_text
            ).replace(r"{{", "{").replace(r"}}", "}")
        except Exception as e:
            print(f"[improve_revision] Unexpected error on attempt {attempt+1}: {e}")
            error_text = f"予期せぬエラーが発生しました。修正してください。\n{e}\n"
            fix_prompt += JSON_FORMAT_FIX_PROMPT.format(
                error_text=error_text
            ).replace(r"{{", "{").replace(r"}}", "}")

        if fix_prompt:
            # 修正を行う
            coder_out = coder.run(fix_prompt)

    if not json_check_success:
        print("[improve_revision] Final json format check failed after 3 attempts.")
        return current_draft  # fallback

    print(f"[improve_revision] Successfully wrote improved revision to {revision_file}")

    return final_checked_data