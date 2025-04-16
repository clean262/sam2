import yt_dlp


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # 高画質かつMP4形式でダウンロード
        'outtmpl': '%(title)s.%(ext)s',  # ダウンロードしたファイルの名前
        'merge_output_format': 'mp4',  # 映像と音声のフォーマットをmp4で統合
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ダウンロードしたいYouTube動画のURLを入力
video_url = 'https://www.youtube.com/watch?v=M2cckDmNLMI'
download_video(video_url)