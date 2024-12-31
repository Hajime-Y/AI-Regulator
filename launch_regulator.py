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

NUM_REFLECTIONS = 3

def print_time():
    """現在時刻を表示するヘルパー関数"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def parse_arguments():
    """AI Regulator用の引数をパースする関数"""
    parser = argparse.ArgumentParser(description="Run AI regulator")

    # 必要に応じてコマンドライン引数を追加
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
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
        default="gpt-4",
        help="使用するLLMモデル",
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

def do_regulation(regulation, revision_file, review_file, regulations_dir, coder, num_reflections):
    """
    1つの規定を改定するための処理をまとめた関数。
    以下の手順を実行する:
      1. 改定案の作成 (draft_revision)
      2. 改定案のレビュー (review_revision)
      3. 改定案の改善 (improve_revision)
    結果は revision_file (revision.json) に上書きで追記し、
    review_file (review.json) にも必要に応じて追記する。
    """
    print_time()
    print(f"[*] Start revising: {regulation['title']}")

    # 改定案の作成
    draft_res = draft_revision(
        regulation=regulation,
        regulations_dir=regulations_dir,
        coder=coder,
        out_file=revision_file,
        num_reflections=num_reflections
    )
    if not draft_res:
        print(f"[draft_revision] 規定 {regulation['title']} の改定案生成に失敗しました。")
        return

    # draft_revision の結果(バージョン1)を revision.json に保存
    _append_revision(revision_file, draft_res)

    # レビュー
    review_res = review_revision(draft_res)
    _append_review(review_file, review_res)

    # 改善
    improved_res = improve_revision(draft_res, review_res)
    # improve_revision の結果(バージョン2)を revision.json に再度上書き追記
    _append_revision(revision_file, improved_res)

    print_time()
    print(f"[+] Finished revising: {regulation['title']}")
    return

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

def worker(queue, revision_file, review_file, gpu_id, regulations_dir, base_dir, client, model, coder, num_reflections):
    """
    並列実行時に呼ばれるワーカー関数。
    queue から規定を取り出して revise する。
    """
    # 並列実行用にGPU指定を行う
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker started on GPU {gpu_id}.")

    while True:
        regulation = queue.get()
        if regulation is None:
            break
        try:
            do_regulation(regulation, revision_file, review_file, regulations_dir, coder, num_reflections)
        except Exception as e:
            print(f"Failed to revise regulation {regulation['title']}: {str(e)}")

    print(f"Worker on GPU {gpu_id} finished.")

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
        num_reflections=NUM_REFLECTIONS,
    )
    # 規定ファイルを保存（ターゲットとなる規定の一覧）
    with open(os.path.join(base_dir, "target_regulations.json"), "w", encoding="utf-8") as f:
        json.dump(regulations, f, ensure_ascii=False, indent=2)

    # 改定要否の確認（改定が必要なものをフィルタリングするなど）
    regs_to_revise = check_revisions(
        target_regulations=regulations,
        regulations_dir=regulations_dir,
        base_dir=base_dir,
        client=client,
        model=client_model,
        num_reflections=NUM_REFLECTIONS,
    )

    # revision.json と review.json の初期化（あれば流用でもよい）
    revision_file = os.path.join(base_dir, "revision.json")
    review_file = os.path.join(base_dir, "review.json")

    # すでに存在していれば（前回処理の続きなどを想定し）追記可能だが、
    # 必要に応じて初期化する場合はコメントアウトを外してください。
    # if osp.exists(revision_file):
    #     os.remove(revision_file)
    # if osp.exists(review_file):
    #     os.remove(review_file)

    # Coderの初期化（launch_scientist.pyを参照）
    io = InputOutput(yes=True, chat_history_file="revision_history.txt")
    main_model = Model(args.model)
    coder = Coder.create(
        main_model=main_model,
        fnames=[revision_file, review_file],
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
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
                args=(queue, revision_file, review_file, gpu_id, regulations_dir, base_dir, client, client_model, coder, NUM_REFLECTIONS),
            )
            p.start()
            # 必要に応じてワーカー間の起動間隔を入れる (例: time.sleep(2))
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
            try:
                do_regulation(reg, revision_file, review_file, regulations_dir, coder, NUM_REFLECTIONS)
            except Exception as e:
                print(f"Failed to revise regulation {reg['title']}: {str(e)}")

    # すべての改定提案が完了した後にレポートを生成
    # revision.json と review.json の内容を踏まえてレポート作成
    generate_report(
        revision_file=revision_file,
        review_file=review_file,
        md_report="revision_report.md",
        pdf_report="revision_report.pdf"
    )

    print("All regulations revised. Final report generated.")

if __name__ == "__main__":
    main()
