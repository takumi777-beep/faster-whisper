import os
# DLLの読み込み許可（拓海くんが見つけた解決策をそのまま採用！）
os.add_dll_directory(os.getcwd())

from faster_whisper import WhisperModel

# 1. SRT用の時間フォーマット (00:00:00,000) に変換する関数
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def wav_to_srt(wav_file, srt_file):
    # モデルの読み込み (CPU処理、int8で軽量化)
    # 5060の力を使わなくても、これなら確実に動くよ
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    print(f"--- {wav_file} の解析を開始 ---")
    segments, info = model.transcribe(wav_file, beam_size=5)

    print(f"検知された言語: {info.language}")

    # SRTファイルとして書き込み
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_time(segment.start)
            end = format_time(segment.end)
            text = segment.text.strip()

            # SRTの基本フォーマット：番号 -> 時間 -> テキスト -> 空行
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

            # 進捗を画面にも出す
            print(f"[{start} --> {end}] {text}")

    print(f"--- 完了！ {srt_file} を保存したよ ---")

if __name__ == "__main__":
    # ここにファイル名を入れて実行！
    wav_to_srt("kamo.wav", "kamo.srt")