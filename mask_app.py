import torch
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

import colorsys
import datetime
import os
import subprocess

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np

from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor



class PromptGUI(object):
    """
    SAM2モデルの初期化と管理
    init_sam_model で内部のSegmentationモデル（SAM2）をロードし、self.sam_model として保持

    ユーザー入力の管理
    selected_points や selected_labels によって、ユーザーがクリック・指定した箇所（正例／負例など）を登録

    マスクの生成・保存
    ユーザーの入力に基づいて get_sam_mask でマスクを生成し、run_tracker で動画全体にマスクを伝播
    最後に save_masks_to_dir でまとめて保存

    動画・フレームの切り替え
    set_img_dir で画像フォルダを指定し、set_input_image で特定フレームをロード。ユーザーが任意のフレームを選んでマスクを作成
    """
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir # SAM2のチェックポイントファイルが置いてあるディレクトリ
        self.model_cfg = model_cfg  #  モデルの設定
        self.sam_model = None
        self.tracker = None

        self.selected_points = [] # ユーザーがクリックで選んだ点
        self.selected_labels = [] # ユーザーがクリックで選んだ点のラベル
        self.cur_label_val = 1.0  # 新しく追加する点が正例か負例かを示すフラグ（1.0 or 0.0）

        self.frame_index = 0  # 現在のフレーム（画像）インデックス
        self.image = None  # 現在ロード中の画像 (np.ndarray)
        self.cur_mask_idx = 0  # 現在生成しているマスクのID
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}  # 現在のフレームで生成したマスクを、マスクID→2値マスク(np.ndarray)で保持
        self.cur_logits = {}  # 現在のフレームで生成したマスクのロジット(モデル出力)
        self.index_masks_all = []  # 全フレームに対する「インデックスマスク」を順番に格納
        self.color_masks_all = []  #  全フレームに対する「カラー化したマスク」を順番に格納

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

    def init_sam_model(self): # SAM2のモデルを初期化
        if self.sam_model is None:
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
            print("checking sam_model")
            print(next(self.sam_model.parameters()).device)  # cuda:0
            guru.info(f"loaded model checkpoint {self.checkpoint_dir}")


    def clear_points(self) -> tuple[None, None, str]:  # 現在選択している点（正例／負例）をすべてクリア
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def add_new_mask(self):  # 新しいマスクを作成開始
        self.cur_mask_idx += 1
        self.clear_points()  # 選択ポイントをクリア。
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    # 1フレーム分の複数マスク（IDごとに分かれている）を1枚の「インデックスマスク」（画素値にマスクIDが入った画像）にまとめる
    def make_index_mask(self, masks):  # masks は {マスクID: 2値マスク} の辞書になっている想定
        assert len(masks) > 0
        idcs = list(masks.keys())
        # 一つ目のマスクを idx_mask としてコピーし、その後、各マスク部分に対応する画素を i+1 などで書き込む。
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask  # 最終的に、「背景=0, マスク1の画素=1, マスク2の画素=2, …」のようなインデックス形式のマスクを返す

    def _clear_image(self):  # 現在の画像とマスクをクリア
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def reset(self):  # 画像やマスクなど、クラス内部の状態をリセットし、SAMモデルの状態も初期化し直す
        self._clear_image()
        self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir: str) -> int:  # 指定したディレクトリ内の画像を取得して配列に格納
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:  # self.img_paths からインデックス i の画像を読み込み、self.image にセットして返す
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image

        return image

    # SAMモデルに対し、フォルダ内の画像群を「動画」として扱うための初期化 (init_state) を行う
    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        # sam2/sam2/utils/misc.pyの246行目にに".png", ".PNG"がないの意味わからん
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir) # SAM2モデル側に「このフォルダが動画ソースですよ」と教える
        # self.inference_state = self.sam_model.init_state(video_path=os.path.abspath(self.img_dir))
        self.sam_model.reset_state(self.inference_state) # SAM2モデルの状態をリセット
        msg = (
            "SAM features extracted. "
            "Click points to update mask, and submit when ready to start tracking"
        )
        return msg, self.image

    def set_positive(self) -> str:  # 新たに追加する点を「正例（1.0）」として扱うよう設定
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:  # 新たに追加する点を「負例（0.0）」として扱うよう設定
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, frame_idx, i, j):  # GUIなどでユーザーがクリックした座標 (i, j) を正例 or 負例として登録し、SAM2モデルに入力してマスクを更新
        """
        get the index mask of the objects
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )
        mask = self.make_index_mask(masks)  # 1フレーム分の複数マスク（IDごとに分かれている）を1枚の「インデックスマスク」（画素値にマスクIDが入った画像）にまとめる

        return mask
    
    # 与えられた点やラベルから、指定フレームのマスクを作成(SAM2モデルへの実際の推論リクエスト)
    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_model is not None
        
        # GPU最適化
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )  # オブジェクトIDとロジットを受け取る

        return  {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }  # ロジットから (logits > 0.0) の箇所をTrueとする2値マスクを作り、IDごとの辞書として返す

    # すでに設定・追加した点情報をもとに、動画（フォルダ内の画像全フレーム）に対してマスクを「伝播」し、全フレームのマスクを生成
    def run_tracker(self, fps: int = 30) -> tuple[str, str]:

        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]  # 画像を読み込み(alphaチャンネルを除去)
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 全フレームについて(マスクID→2値マスク)の辞書を得る。
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
            # index_masks_all.append(self.make_index_mask(masks))

        self.index_masks_all = [self.make_index_mask(v) for k, v in video_segments.items()]

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all) # インデックスマスクをカラー表示用に変換(green background)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames, fps=fps, macro_block_size=1)  # 生成したカラー付きの画像を動画（MP4）として書き出す
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        assert self.color_masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask, id_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, clr_mask)
            np_out_path = f"{output_dir}/{name[:-4]}.npy" # ファイル名末尾4字（拡張子 .png や .jpg を想定）を切り落とし代わりに ".npy" を付与
            np.save(np_out_path, id_mask)
        
        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        return message


def isimage(p): # 与えられたファイルパスがpng, jpg, jpegのいずれかの拡張子を持つかどうかを調べる
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):  # img に対して、points の座標にラベルに応じた円を描画
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (255, 0, 0) if label == 1.0 else (0, 0, 255)
        out = cv2.circle(out, (x, y), 10, color, -1)  # 座標 (x, y) に半径 10 ピクセルの円を描画(-1 は塗りつぶし)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    最初の一色は黒、それ以降 n_colors - 1 の色相 (Hue) を均等に分割して、HLS 空間から RGB 空間に変換したカラーパレットを NumPy 配列として返す関数
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")  # 返り値は (n_colors, 3) の形状で、各色を 8bit (0〜255) の RGB で表現


# def colorize_masks(images, index_masks, fac: float = 0.5):  # 元画像と色マスク（例えばヒートマップなど）を合成する関数
#     max_idx = max([m.max() for m in index_masks])
#     guru.debug(f"{max_idx=}")
#     palette = get_hls_palette(max_idx + 1)
#     color_masks = []
#     out_frames = []
#     for img, mask in zip(images, index_masks):
#         clr_mask = palette[mask.astype("int")]
#         color_masks.append(clr_mask)
#         out_u = compose_img_mask(img, clr_mask, fac)
#         out_frames.append(out_u)
#     return out_frames, color_masks
def colorize_masks(images, index_masks):
    """
    index_masks[i] を使って前景と背景を分離し、
    マスク部分は元画像、背景は緑色にしたフレームを返す。
    """
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        # 変更後の compose_img_mask を使う
        out_u = compose_img_mask(img, mask)
        out_frames.append(out_u)
        # color_masks には合成後の画像をそのまま保持するでもOK
        color_masks.append(out_u)
    return out_frames, color_masks


# def compose_img_mask(img, color_mask, fac: float = 0.5):
#     out_f = fac * img / 255 + (1 - fac) * color_mask / 255
#     out_u = (255 * out_f).astype("uint8")
#     return out_u
def compose_img_mask(img, index_mask):
    """
    マスク領域(index_mask > 0)は元のimgを使用し、
    それ以外(==0)の領域を青色に塗りつぶす。
    (コードの実装的には青色の長方形を作り出し、マスク>0の部分を元画像で上書きしている)
    """
    # 例えば OpenCV(BGR) で純粋な青色にするなら (255, 0, 0)
    # RGB で青にしたい場合は (0, 0, 255) 等に変えてください
    blue_bgr = (0, 255, 0)

    # 画像と同じサイズで全ピクセルを青色に初期化
    out = np.full_like(img, blue_bgr, dtype=img.dtype)  # 画像と同じサイズ・型の青画像

    # マスク領域(>0)だけを元の画像に置き換える
    mask = (index_mask > 0)  # マスク領域を True にする
    out[mask] = img[mask]  # 前景部分だけを元画像からコピー

    return out


def listdir(vid_dir):  # 指定されたフォルダパス vid_dir が存在する場合、その中のファイル名をソートして返す
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(
    checkpoint_dir,
    model_cfg,
    root_dir,
    vid_name: str = "videos", # 動画フォルダ名
    img_name: str = "images", # 画像（フレーム）フォルダ名
    mask_name: str = "masks", # マスク保存先フォルダ名
):
    prompts = PromptGUI(checkpoint_dir, model_cfg)  # SAM関連のモデルや選択点などを管理するクラス PromptGUI を初期化。

    start_instructions = (
        "Select a video file to extract frames from, "
        "or select an image directory with frames already extracted."
    )
    vid_root, img_root = (f"{root_dir}/{vid_name}", f"{root_dir}/{img_name}")  # 動画フォルダと画像フォルダのパスを作成
    with gr.Blocks() as demo:  # Gradio UI の構築
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )
        with gr.Row():  # データセットのルート、サブフォルダ名、シーケンス名を設定。(Gradio UIの最上段)
            root_dir_field = gr.Text(root_dir, label="Dataset root directory")
            vid_name_field = gr.Text(vid_name, label="Video subdirectory name")
            img_name_field = gr.Text(img_name, label="Image subdirectory name")
            mask_name_field = gr.Text(mask_name, label="Mask subdirectory name")
            seq_name_field = gr.Text(None, label="Sequence name", interactive=False)

        with gr.Row():
            with gr.Column():  # 左列：動画選択とフレーム抽出
                vid_files = listdir(vid_root)
                vid_files_field = gr.Dropdown(label="Video files", choices=vid_files)
                input_video_field = gr.Video(label="Input Video")

                with gr.Row():
                    start_time = gr.Number(0, label="Start time (s)")
                    end_time = gr.Number(0, label="End time (s)")
                    sel_fps = gr.Number(30, label="FPS")
                    sel_height = gr.Number(1080, label="Height")
                    extract_button = gr.Button("Extract frames")

            with gr.Column():  # 中央列：画像選択とポイント操作
                img_dirs = listdir(img_root)
                img_dirs_field = gr.Dropdown(
                    label="Image directories", choices=img_dirs
                )
                img_dir_field = gr.Text(
                    None, label="Input directory", interactive=False
                )
                frame_index = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=len(prompts.img_paths) - 1,
                    value=0,
                    step=1,
                )
                sam_button = gr.Button("Get SAM features")
                reset_button = gr.Button("Reset")
                input_image = gr.Image(
                    prompts.set_input_image(0),
                    label="Input Frame",
                    every=1,
                )
                with gr.Row():
                    pos_button = gr.Button("Toggle positive")
                    neg_button = gr.Button("Toggle negative")
                clear_button = gr.Button("Clear points")

            with gr.Column():  # 右列：マスク処理と保存
                output_img = gr.Image(label="Current selection")
                add_button = gr.Button("Add new mask")
                submit_button = gr.Button("Submit mask for tracking")
                final_video = gr.Video(label="Masked video")
                mask_dir_field = gr.Text(
                    None, label="Path to save masks", interactive=False
                )
                save_button = gr.Button("Save masks")

        def update_vid_root(root_dir, vid_name):  # ルートディレクトリ更新に応じてファイル一覧更新
            vid_root = f"{root_dir}/{vid_name}"
            vid_paths = listdir(vid_root)
            guru.debug(f"Updating video paths: {vid_paths=}")
            return vid_paths

        def update_img_root(root_dir, img_name):  # ルートディレクトリ更新に応じて画像フォルダのファイル一覧更新
            img_root = f"{root_dir}/{img_name}"
            img_dirs = listdir(img_root)
            guru.debug(f"Updating img dirs: {img_dirs=}")
            return img_root, img_dirs

        def update_mask_dir(root_dir, mask_name, seq_name):  # ルートディレクトリ更新に応じてマスクフォルダのパスを更新
            return f"{root_dir}/{mask_name}/{seq_name}"

        def update_root_paths(root_dir, vid_name, img_name, mask_name, seq_name):  # ルートディレクトリ更新に応じて動画、画像、マスクフォルダのパスを更新
            return (
                update_vid_root(root_dir, vid_name),
                update_img_root(root_dir, img_name),
                update_mask_dir(root_dir, mask_name, seq_name),
            )

        def select_video(root_dir, vid_name, seq_file):  # 動画選択時にファイルパスとシーケンス名を決定
            seq_name = os.path.splitext(seq_file)[0]
            guru.debug(f"Selected video: {seq_file=}")
            vid_path = f"{root_dir}/{vid_name}/{seq_file}"
            return seq_name, vid_path

        def extract_frames(  # ffmpegでフレーム抽出し保存
            root_dir, vid_name, img_name, vid_file, start, end, fps, height, ext="png"
        ):
            seq_name = os.path.splitext(vid_file)[0]
            vid_path = f"{root_dir}/{vid_name}/{vid_file}"
            out_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Extracting frames to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            def make_time(seconds):
                return datetime.time(
                    seconds // 3600, (seconds % 3600) // 60, seconds % 60
                )

            start_time = make_time(start).strftime("%H:%M:%S")
            end_time = make_time(end).strftime("%H:%M:%S")
            # cmd = (
            #     f"ffmpeg -ss {start_time} -to {end_time} -i {vid_path} "
            #     f"-vf 'scale=-1:{height},fps={fps}' {out_dir}/%05d.{ext}"
            # )
            cmd = (
                f"ffmpeg -ss {start_time} -to {end_time} -i {vid_path} "
                # f"-vf \"scale=-1:{height},fps={fps}\" {out_dir}/%05d.{ext}"
                f"-vf \"scale=-2:{height},fps={fps}\" {out_dir}/%05d.{ext}"
            )
            print(cmd)
            subprocess.call(cmd, shell=True)
            img_root = f"{root_dir}/{img_name}"
            img_dirs = listdir(img_root)
            return out_dir, img_dirs

        def select_image_dir(root_dir, img_name, seq_name):  # 画像ディレクトリの選択
            img_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Selected image dir: {img_dir}")
            return seq_name, img_dir

        def update_image_dir(root_dir, img_name, seq_name):  # スライダー更新
            img_dir = f"{root_dir}/{img_name}/{seq_name}"
            num_imgs = prompts.set_img_dir(img_dir)
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            message = (
                f"Loaded {num_imgs} images from {img_dir}. Choose a frame to run SAM!"
            )
            return slider, message

        # def get_select_coords(frame_idx, img, evt: gr.SelectData):
        #     i = evt.index[1]  # type: ignore
        #     j = evt.index[0]  # type: ignore
        #     index_mask = prompts.add_point(frame_idx, i, j)
        #     guru.debug(f"{index_mask.shape=}")
        #     palette = get_hls_palette(index_mask.max() + 1)
        #     color_mask = palette[index_mask]
        #     out_u = compose_img_mask(img, color_mask)
        #     out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
        #     return out

        def get_select_coords(frame_idx, img, evt: gr.SelectData):  #  画像クリック時の座標を取得し、マスク生成と可視化
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            index_mask = prompts.add_point(frame_idx, i, j)
            guru.debug(f"{index_mask.shape=}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, index_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        # update the root directory
        # and associated video, image, and mask root directories
        # Gradio イベントの結びつけ
        root_dir_field.submit(
            update_root_paths,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                mask_name_field,
                seq_name_field,
            ],
            outputs=[vid_files_field, img_dirs_field, mask_dir_field],
        )
        vid_name_field.submit(
            update_vid_root,
            [root_dir_field, vid_name_field],
            outputs=[vid_files_field],
        )
        img_name_field.submit(
            update_img_root,
            [root_dir_field, img_name_field],
            outputs=[img_dirs_field],
        )
        mask_name_field.submit(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting a video file
        vid_files_field.select(
            select_video,
            [root_dir_field, vid_name_field, vid_files_field],
            outputs=[seq_name_field, input_video_field],
        )

        # when the img_dir_field changes
        img_dir_field.change(
            update_image_dir,
            [root_dir_field, img_name_field, seq_name_field],
            [frame_index, instruction],
        )
        seq_name_field.change(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting an image directory
        img_dirs_field.select(
            select_image_dir,
            [root_dir_field, img_name_field, img_dirs_field],
            [seq_name_field, img_dir_field],
        )

        # extracting frames from video
        extract_button.click(
            extract_frames,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                vid_files_field,
                start_time,
                end_time,
                sel_fps,
                sel_height,
            ],
            outputs=[img_dir_field, img_dirs_field],
        )

        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [frame_index, input_image], [output_img])

        sam_button.click(prompts.get_sam_features, outputs=[instruction, input_image])
        reset_button.click(prompts.reset)
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])

        add_button.click(prompts.add_new_mask, outputs=[output_img, instruction])
        submit_button.click(prompts.run_tracker, inputs=[sel_fps], outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [mask_dir_field], outputs=[instruction]
        )

    return demo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--vid_name", type=str, default="videos")
    parser.add_argument("--img_name", type=str, default="images")
    parser.add_argument("--mask_name", type=str, default="masks")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    demo = make_demo(
        args.checkpoint_dir,
        args.model_cfg,
        args.root_dir,
        args.vid_name,
        args.img_name,
        args.mask_name
    )
    demo.launch(server_port=args.port, share=True, allowed_paths = ["/content/drive/MyDrive/data_sam2/"])