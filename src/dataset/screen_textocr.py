import json

def extract_charset_from_textocr(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    charset = set()

    for img_info in data['anns'].values():
        transcription = img_info.get('utf8_string', '')
        charset.update(transcription)

    charset = sorted(charset)
    print("Unique characters (charset):")
    print("".join(charset))
    print(f"\nTotal unique characters: {len(charset)}")

    return charset

# Example usage:
# charset = extract_charset_from_textocr('TextOCR_0.1_train.json')

extract_charset_from_textocr('dataset/TextOCR/TextOCR_0.1_train.json')