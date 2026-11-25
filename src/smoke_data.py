"""Dữ liệu GIẢ chỉ để kiểm tra đường ống chạy được (python src/prepare_data.py --smoke).
KHÔNG dùng cho kết quả thật."""
import random


def make_corpus(n=4000, seed=0):
    rng = random.Random(seed)
    subj = ["học sinh", "sinh viên", "giáo viên", "giảng viên", "bác sĩ", "kỹ sư", "công nhân", "nông dân"]
    verb = ["đọc", "viết", "học", "dạy", "khám", "sửa", "làm", "trồng"]
    obj = ["sách", "bài tập", "thuốc", "máy tính", "xe máy", "lúa", "thí nghiệm", "chương trình"]
    place = ["ở trường", "ở bệnh viện", "ở nhà máy", "ở ruộng", "ở thư viện", "ở Hà Nội"]
    return [f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)} {rng.choice(place)} mỗi ngày ."
            for _ in range(n)]


def make_vsfc(n_train=600, n_dev=150, n_test=150, seed=0):
    rng = random.Random(seed)
    pos = ["thầy dạy rất hay và nhiệt tình", "giảng viên truyền đạt dễ hiểu", "phòng học sạch sẽ thoáng mát",
           "chương trình học bổ ích", "cô giáo tận tâm với sinh viên", "thư viện đầy đủ sách"]
    neg = ["thầy dạy quá nhanh khó hiểu", "wifi yếu không vào được", "phòng học nóng và bẩn",
           "chương trình học nặng nề", "giảng viên hay đến muộn", "căng tin đắt và dở"]
    neu = ["em không có ý kiến gì", "bình thường", "tạm được"]
    extra = ["ạ", "nhé", "thật sự", "theo em", "năm nay"]

    def one():
        s = rng.choice([0, 0, 1, 2, 2])  # 0 neg, 1 neu, 2 pos
        base = {0: neg, 1: neu, 2: pos}[s]
        return {"sentence": f"{rng.choice(base)} {rng.choice(extra)}", "sentiment": s,
                "topic": rng.randint(0, 3)}
    return {"train": [one() for _ in range(n_train)], "dev": [one() for _ in range(n_dev)],
            "test": [one() for _ in range(n_test)]}
