def find_invalid_utf8_positions(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    try:
        data.decode('utf-8')
        print("✅ 文件是有效的 UTF-8")
    except UnicodeDecodeError as e:
        start = max(0, e.start - 20)
        end = min(len(data), e.end + 20)
        print("❌ 非法字节位置:", e.start, "-", e.end)
        print("上下文字节:", data[start:end])
        print("上下文字符（Latin-1 解码）:", data[start:end].decode('latin-1'))

# 用法
find_invalid_utf8_positions('/home/server/science/zry/BadFreq/poison_dataset/SST-2/positive/LongBD/z_remote_gpt/train.json')