import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# csv記録用
import time
import csv

# 日時設定
from datetime import datetime, timezone, timedelta


# シミュレータ上でのダミー行動（最初の安定化待ちに使用。6軸移動は0、グリッパーは開く[-1]）
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# 学習データに使用されているレンダリング解像度
LIBERO_ENV_RESOLUTION = 256 


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # モデルサーバーの設定パラメータ
    #################################################################################################################
    host: str = "0.0.0.0"  # サーバーのホスト名
    port: int = 8000       # サーバーのポート番号
    resize_size: int = 224 # モデル入力時のリサイズ後の解像度
    replan_steps: int = 5  # 何ステップごとに計画（推論）を更新するか
    JST = timezone(timedelta(hours=+9))
    timestamp = datetime.now(JST).strftime('%Y-%m-%d_%H-%M-%S')# 評価結果の動画やCSVのファイル名にタイムスタンプを付与して管理しやすくする

    #################################################################################################################
    # LIBERO環境固有の設定パラメータ
    #################################################################################################################
    task_suite_name: str = ( # タスクの種類: libero_spatialは位置推定タスク、libero_objectは物体操作タスク、libero_goalは目標達成タスク、libero_10は全10タスク、libero_90は難易度の高い90タスク
        #"libero_spatial",
        "libero_object"
        #"libero_goal",
        #"libero_10",
        #"libero_90"
    )
    num_steps_wait: int = 10  # シミュレーション開始時、オブジェクトが静止するまで待機するステップ数
    num_trials_per_task: int = 5  # 1タスクあたりに実行する試行回数（エピソード数）

    #################################################################################################################
    # ユーティリティ設定
    #################################################################################################################
    video_out_path: str = f"videos/test"  # 評価結果の動画を保存するパス
    csv_out_path: str = f"csv/libero/test"  # 評価結果のCSVを保存するパス

    seed: int = 7  # 再現性のための乱数シード


def eval_libero(args: Args) -> None:
    # csvファイルに記録するための準備
    output_dir = pathlib.Path(args.csv_out_path) / args.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 試行ごとの詳細ログ (Raw Data)
    raw_csv_path = output_dir / "episode_log.csv"
    raw_headers = ["task_id", "episode_idx", "success", "steps", "avg_latency_ms"]
    with open(raw_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(raw_headers)

    # 2. タスクごとの統計ログ (Summary)
    summary_csv_path = output_dir / "task_log.csv"
    summary_headers = [
        "task_id", "task_description", "success_rate", 
        "steps_mean", "steps_std", 
        "latency_mean_ms", "latency_std_ms"
    ]
    with open(summary_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(summary_headers)
            
    # 乱数シードの設定
    np.random.seed(args.seed)

    # LIBEROタスクスイートの初期化
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # 動画保存用ディレクトリの作成
    video_dir = pathlib.Path(args.video_out_path) / args.task_suite_name / args.timestamp
    video_dir.mkdir(parents=True, exist_ok=True)

    # 各タスクスイートごとの最大ステップ数を設定（学習データの最長デモに基づき余裕を持たせる）
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 推論サーバーと通信するクライアントの初期化
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 評価ループの開始
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # csv記録用の変数
        task_latencies = []  
        task_steps = []
        task_success_flags = [] 
        
        # タスク情報の取得
        task = task_suite.get_task(task_id)

        # LIBEROのデフォルト初期状態（姿勢等）を取得
        initial_states = task_suite.get_task_init_states(task_id)

        # LIBERO環境とタスク説明（言語指示）の初期化
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # エピソード（試行）ループの開始
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            episode_latencies = []  # このエピソード内の各推論のレイテンシを記録
            logging.info(f"\nTask: {task_description}")

            # 環境のリセット
            env.reset()
            # 実行予定のアクション列を格納するデック
            action_plan = collections.deque()

            # 初期状態の設定
            obs = env.set_init_state(initial_states[episode_idx])

            # 各種変数のセットアップ
            t = 0
            done = False
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # 【重要】シミュレータがオブジェクトを配置する際に落下・振動するため、
                    # 最初の数ステップは何もしないで静止を待つ
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # 前処理済みの画像を取得
                    # 【重要】学習時の前処理に合わせて180度回転（上下左右反転）させる
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # 画像のリサイズとパディングを行い、8bit形式に変換
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # 再生動画用に前処理済み画像を保存
                    replay_images.append(img)

                    if not action_plan:
                        # 以前のアクション計画を全て実行し終えた場合、新しいチャンクを計算する
                        
                        # 観測データ（Observations）辞書の作成
                        # image: 第三者視点カメラ画像
                        # wrist_image: ロボット手元カメラ画像
                        # state: ロボットの内部状態（手先位置[3] + 軸角表現の姿勢[3] + グリッパー状態[1]）
                        # prompt: タスクの内容を示す自然言語テキスト
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        start_time = time.perf_counter()
                        # サーバーへ問い合わせてアクション列を取得
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"再計画の間隔を {args.replan_steps} ステップに設定していますが、モデルは {len(action_chunk)} ステップ分しか予測していません。"
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                        episode_latencies.append(latency_ms)
                        task_latencies.append(latency_ms)
                        
                        # アクションチャンクのうち、再計画ステップまでの分を計画に加える
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # 計画から次のアクションを取り出す
                    action = action_plan.popleft()

                    # 環境内でアクションを実行
                    obs, reward, done, info = env.step(action.tolist())
                    # 成功判定（LIBERO環境側でdoneが返れば成功）
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"例外を検知しました: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # エピソード単位の統計を更新
            episode_steps = t + 1 if done else t
            avg_latency_ms = float(np.mean(episode_latencies)) if episode_latencies else 0.0
            task_steps.append(episode_steps)
            task_success_flags.append(done)

            # 試行ごとの詳細ログをCSVに追記
            with open(raw_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    task_id,
                    episode_idx,
                    done,
                    episode_steps,
                    avg_latency_ms,
                ])

            # エピソードの再生動画を保存
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_dir / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # 現在の結果をログ出力
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # タスク単位および全体の成功率をログ出力
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # タスクごとの統計をCSVに追記
        success_rate = float(np.mean(task_success_flags)) if task_success_flags else 0.0
        steps_mean = float(np.mean(task_steps)) if task_steps else 0.0
        steps_std = float(np.std(task_steps)) if task_steps else 0.0
        lat_mean = float(np.mean(task_latencies)) if task_latencies else 0.0
        lat_std = float(np.std(task_latencies)) if task_latencies else 0.0

        with open(summary_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                task_id,
                task_description,
                success_rate,
                steps_mean,
                steps_std,
                lat_mean,
                lat_std,
            ])

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """LIBERO環境を初期化し、環境オブジェクトとタスク説明（言語）を返します。"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "control_freq": 20,
        }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # 重要: 固定の初期状態を使用する場合でも、シード値が物体の配置に影響することがあります
    return env, task_description


def _quat2axisangle(quat):
    """
    クォータニオンを軸角表現に変換します。
    robosuiteのtransform_utilsからコピー:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # クォータニオンの値をクリップして範囲内に収める
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 回転角が0度に近い場合、ゼロベクトルを返す
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    # ログレベルをINFOに設定
    logging.basicConfig(level=logging.INFO)
    # CLI引数を解析してeval_libero関数を実行
    tyro.cli(eval_libero)