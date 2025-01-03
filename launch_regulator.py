import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from datetime import datetime
from typing import Dict, Any

# aider関連のライブラリは必要に応じてインストールしてください
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

# AI Regulator 用モジュール
from ai_regulator.list_regulations import list_regulations, check_revisions
from ai_regulator.perform_revision import draft_revision
from ai_regulator.perform_review import review_revision, improve_revision
from ai_regulator.generate_report import generate_report
from ai_regulator.llm import create_client, AVAILABLE_LLMS

def print_time():
    """現在時刻を表示するヘルパー関数"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def parse_arguments():
    """AI Regulator用の引数をパースする関数"""
    parser = argparse.ArgumentParser(description="Run AI regulator")

    parser.add_argument(
        "--skip_list_regulations",
        action="store_true",
        help="Skip regulation listing and use existing target_regulations.json.",
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip revision check and only list target regulations.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--regulations_dir",
        type=str,
        required=True,
        help="Path to the regulations directory.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Regulator.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Create draft revisions.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Perform reviews on drafts.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=3,
        help="Number of reflection iterations. Default is 3.",
    )

    return parser.parse_args()

def get_available_gpus(gpu_ids=None):
    """
    使用可能なGPUリストを返すユーティリティ関数
    """
    import torch
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))

def do_regulation(
    regulation: Dict[str, Any],
    regulations_dir: str,
    base_dir: str,
    num_reflections: int,
    model: str,
    client: Any,
    client_model: str,
    draft: bool = True,
    review: bool = False,
    improvement: bool = False,
    log_file: bool = False,
) -> bool:
    """
    1つの規定を改定するための処理をまとめた関数。
    フラグに応じて以下の手順を実行する:
      1. 改定案の作成 (draft_revision) - draft=True
      2. 改定案のレビュー (review_revision) - review=True
      3. 改定案の改善 (improve_revision) - improvement=True
    結果は revision_file (revision.json) と review_file (review.json) に出力。
    """
    # ログファイル用の標準出力・標準エラー出力の退避
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log

    try:
        # regulation_nameを生成（pathから拡張子なしのファイル名を取得）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        regulation_name = os.path.splitext(os.path.basename(regulation["path"]))[0] + "_" + timestamp
        folder_name = osp.join(base_dir, regulation_name)
        
        # フォルダ作成とファイルパス設定
        os.makedirs(folder_name, exist_ok=True)
        revision_file = osp.join(folder_name, "revision.json")
        review_file = osp.join(folder_name, "review.json")

        print_time()
        print(f"[*] Start revising: {regulation['path']}")

        # Coderの初期化
        io = InputOutput(yes=True, chat_history_file=f"{folder_name}/revision_history.txt")
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=[revision_file],
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        draft_res = None
        review_res = None

        # 改定案の作成
        if draft:
            draft_res = draft_revision(
                regulation=regulation,
                regulations_dir=regulations_dir,
                base_dir=folder_name,
                coder=coder,
                revision_file=revision_file,
                num_reflections=num_reflections,
            )
            if not draft_res:
                print(f"[draft_revision] 規定 {regulation['path']} の改定案生成に失敗しました。")
                return False
            
            # draft_revision の結果(バージョン1)を revision.json に保存
            _append_revision(revision_file, draft_res)
        elif osp.exists(revision_file):
            # 既存の改定案を読み込む
            with open(revision_file, "r", encoding="utf-8") as f:
                revisions = json.load(f)
                draft_res = revisions[-1] if isinstance(revisions, list) else revisions

        # レビュー
        if review and draft_res:
            review_res = review_revision(
                regulation=regulation,
                draft_revision=draft_res,
                regulations_dir=regulations_dir,
                base_dir=folder_name,
                model=client_model,
                client=client,
                num_reflections=num_reflections,
                num_reviews_ensemble=5,
                temperature=0.1,
            )
            if not review_res:
                print(f"[review_revision] 規定 {regulation['path']} のレビューに失敗しました。")
                return False
            
            _append_review(review_file, review_res)
        elif osp.exists(review_file):
            # 既存のレビューを読み込む
            with open(review_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
                review_res = reviews[-1] if isinstance(reviews, list) else reviews

        # 改善
        if improvement and draft_res and review_res:
            improved_res = improve_revision(
                current_draft=draft_res,
                review_data=review_res,
                coder=coder,
                num_reflections=num_reflections,
            )
            if not improved_res:
                print(f"[improve_revision] 規定 {regulation['path']} の改善に失敗しました。")
                return False

            # improve_revision の結果(バージョン2)を revision.json に再度上書き追記
            _append_revision(revision_file, improved_res)

        print_time()
        print(f"[+] Finished revising: {regulation['path']}")
        return True

    except Exception as e:
        print(f"[!] Error in do_regulation for {regulation.get('path', 'Unknown')}: {str(e)}")
        return False

    finally:
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


def _append_revision(revision_file, revision_data):
    """
    revision.json に複数バージョンの改定内容を上書き追加するためのヘルパー関数。
    """
    if not osp.exists(revision_file):
        with open(revision_file, "w", encoding="utf-8") as f:
            json.dump([revision_data], f, ensure_ascii=False, indent=2)
    else:
        with open(revision_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

        existing_data.append(revision_data)
        with open(revision_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

def _append_review(review_file, review_data):
    """
    review.json にレビュー結果を上書き追加するためのヘルパー関数。
    """
    if not osp.exists(review_file):
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump([review_data], f, ensure_ascii=False, indent=2)
    else:
        with open(review_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

        existing_data.append(review_data)
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

def worker(
        queue,
        regulations_dir: str,
        base_dir: str,
        num_reflections: int,
        model: str,
        client: Any,
        client_model: str,
        draft: bool,
        review: bool,
        improvement: bool,
        gpu_id: int,
):
    """
    並列実行時に呼ばれるワーカー関数。
    queue から規定を取り出して revise する。
    """
    # 並列実行用にGPU指定を行う
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")

    while True:
        regulation = queue.get()
        if regulation is None:
            break

        success = do_regulation(
            regulation=regulation,
            regulations_dir=regulations_dir,
            base_dir=base_dir,
            num_reflections=num_reflections,
            model=model,
            client=client,
            client_model=client_model,
            draft=draft,
            review=review,
            improvement=improvement,
            log_file=True,
        )
        print(f"Completed regulation: {regulation['path']}, Success: {success}")
    
    print(f"Worker {gpu_id} finished.")

def main():
    args = parse_arguments()

    # GPUのチェックと並列実行数の調整
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print_time()
    print(f"Using GPUs: {available_gpus}")

    # client初期化
    client, client_model = create_client(args.model)

    regulations_dir = args.regulations_dir
    base_dir = args.base_dir

    # 規定集をリストアップ
    regulations = list_regulations(
        regulations_dir=regulations_dir,
        base_dir=base_dir,
        client=client,
        model=client_model,
        num_reflections=args.num_reflections,
        skip_list_regulations=args.skip_list_regulations,
    )
    # 規定ファイルを保存（ターゲットとなる規定の一覧）
    with open(os.path.join(base_dir, "target_regulations.json"), "w", encoding="utf-8") as f:
        json.dump(regulations, f, ensure_ascii=False, indent=4)

    # 改定要否の確認（改定が必要なものをフィルタリングするなど）
    regs_to_revise = check_revisions(
        target_regulations=regulations,
        regulations_dir=regulations_dir,
        base_dir=base_dir,
        client=client,
        model=client_model,
        num_reflections=args.num_reflections,
    )

    # 並列実行モードとシーケンシャルモードの分岐
    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()

        # Queueに規定を投入
        for reg in regs_to_revise:
            queue.put(reg)

        # ワーカーを起動
        processes = []
        for i in range(args.parallel):
            # GPU が足りなければ round-robinで使い回す
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue, 
                    regulations_dir, 
                    base_dir, 
                    args.num_reflections,
                    args.model, 
                    client, 
                    client_model,
                    args.draft,
                    args.review,
                    args.improvement,
                    gpu_id, 
                ),
            )
            p.start()
            time.sleep(150)  # ワーカー間の起動間隔
            processes.append(p)

        # 終了シグナルを送信
        for _ in range(args.parallel):
            queue.put(None)

        # 全プロセスの完了を待機
        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        # シーケンシャル実行
        for reg in regs_to_revise:
            success = do_regulation(
                regulation=reg,
                regulations_dir=regulations_dir,
                base_dir=base_dir,
                num_reflections=args.num_reflections,
                model=args.model,
                client=client,
                client_model=client_model,
                draft=args.draft,
                review=args.review,
                improvement=args.improvement,
                log_file=True,
            )
            if not success:
                print(f"Failed to revise regulation {reg['path']}")

    # すべての改定提案が完了した後にレポートを生成
    # revision.json と review.json の内容を踏まえてレポート作成
    # generate_report(
    #     # revision_file=revision_file,
    #     # review_file=review_file,
    #     md_report="revision_report.md",
    #     pdf_report="revision_report.pdf"
    # )

    print("All regulations revised. Final report generated.")

if __name__ == "__main__":
    main()
