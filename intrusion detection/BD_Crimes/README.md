# BD_Crimes_K

CCTV ও অনলাইন ভিডিও থেকে অপরাধ ও intrusion শনাক্তকরণ ডেটাসেট (২৫টি ক্লাস)। Kinetics ধাঁচে সাজানো।
মডেল: YOLO26n-pose + ByteTrack + GRU (PTG-Net), Raspberry Pi 5-এ চালানোর জন্য।

## কাঠামো

```
BD_Crimes_K/
├── raw/youtube/          ইউটিউব থেকে নামানো পুরো ভিডিও (<youtube_id>.mp4) — কখনো বদলাবেন না, প্রকাশ করবেন না
├── annotations/          train.csv, val.csv, test.csv (প্রতি ক্লিপে এক সারি), classes.csv
├── videos/               train/ val/ test/ → ২৫টি ক্লাস ফোল্ডার → কাটা ক্লিপ
├── detection/            YOLO ফরম্যাটে bounding box (YOLO26n detect ট্রেনিং)
├── pose/                 keypoint ক্রম .npy [32, 2, 17, 3] (GRU ট্রেনিং)
├── models/               Raspberry Pi 5-এর জন্য export করা মডেল
└── README.md
```

## নামকরণ

| ধরন | নাম | উদাহরণ |
| --- | --- | --- |
| মূল ভিডিও | `<youtube_id>.mp4` | `abc123.mp4` |
| কাটা ক্লিপ | `<youtube_id>_<শুরু ৬ অঙ্ক>_<শেষ ৬ অঙ্ক>.mp4` | `abc123_000720_000780.mp4` |
| ফ্রেম | `<ক্লিপের নাম>_f<ফ্রেম ৫ অঙ্ক>.jpg` | `abc123_000720_000780_f00625.jpg` |
| বক্স লেবেল | ছবির নামেই `.txt` | `abc123_000720_000780_f00625.txt` |

## annotations/*.csv কলাম

| কলাম | অর্থ |
| --- | --- |
| label | ক্লাসের নাম (ফোল্ডারের নামের মতো, যেমন snatching) |
| youtube_id | মূল ভিডিওর ID |
| time_start, time_end | মূল ভিডিওর কোন অংশ কাটা হয়েছে (সেকেন্ড) |
| split | train / val / test |
| event_start, event_end | **ক্লিপের ভেতরে** ঘটনার সময় (সেকেন্ড); normal-এ ফাঁকা |
| fps | ক্লিপের fps |
| channel | ইউটিউব চ্যানেল |
| footage_type | cctv / phone / dashcam / broadcast |
| lighting | day / night / low_light |
| scene | street / shop / market / home / office / transport |
| annotator, verified | কে লেবেল করেছে, দ্বিতীয়জন যাচাই করেছে কি না (yes/no) |

উদাহরণ:
```
snatching,abc123,720,780,train,25,33,25,Somoy TV,cctv,night,street,A1,yes
```

## নিয়ম

1. একই ইউটিউব ভিডিওর সব ক্লিপ একই split-এ থাকবে (data leakage এড়াতে)।
2. detection/ আর pose/-এর split মূল ক্লিপের split অনুসরণ করবে (videos/val → detection/valid)।
3. ঘটনার আগে ১৫–২০ সে. আর পরে ১০–২০ সে. রেখে ক্লিপ কাটুন (মোট ~৬০ সে.)।
4. class_id ০ থেকে শুরু (PyTorch-এর জন্য)।

## কমান্ড

```bash
# ডাউনলোড
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" --merge-output-format mp4 \
       -o "raw/youtube/%(id)s.mp4" "https://www.youtube.com/watch?v=abc123"

# কাটা (720 সেকেন্ড থেকে 60 সেকেন্ড)
ffmpeg -ss 720 -i raw/youtube/abc123.mp4 -t 60 -c:v libx264 -crf 20 -preset fast -c:a aac \
       videos/train/snatching/abc123_000720_000780.mp4
```

## ক্লাস

| id | ফোল্ডার | বাংলা অর্থ |
| --- | --- | --- |
| 0 | normal | স্বাভাবিক কাজকর্ম |
| 1 | stealing | চুরি (জিনিস তুলে নেওয়া) |
| 2 | burglary_breakin | দরজা বা জানালা ভেঙে প্রবেশ |
| 3 | robbery | অস্ত্রের ভয় দেখিয়ে ডাকাতি |
| 4 | snatching | ছিনতাই (টান দিয়ে নিয়ে দৌড়) |
| 5 | gun_threat | বন্দুক তাক করা বা গুলি করা |
| 6 | knife_attack | ছুরি দিয়ে আক্রমণ |
| 7 | stick_attack | লাঠি বা রড দিয়ে আঘাত |
| 8 | fighting | মারামারি |
| 9 | assault_harassment | আক্রমণ বা হয়রানি |
| 10 | kidnapping | অপহরণ |
| 11 | hanging | ফাঁসি বা আত্মহত্যার চেষ্টা |
| 12 | falling | পড়ে যাওয়া |
| 13 | fainting_collapse | অজ্ঞান হয়ে পড়া |
| 14 | vandalism | ভাঙচুর |
| 15 | arson_fire | আগুন লাগানো |
| 16 | wall_climbing | দেয়াল বা গেট টপকানো |
| 17 | loitering | সন্দেহজনকভাবে ঘোরাফেরা |
| 18 | throwing_object | ইট বা বস্তু ছোঁড়া |
| 19 | camera_tampering | CCTV ঢেকে দেওয়া বা ঘোরানো |
| 20 | abuse | শিশু, বৃদ্ধ বা প্রাণীর প্রতি নিষ্ঠুর আচরণ |
| 21 | arrest | পুলিশ কর্তৃক গ্রেপ্তার |
| 22 | explosion | বিস্ফোরণ |
| 23 | road_accident | সড়ক দুর্ঘটনা |
| 24 | shoplifting | দোকান থেকে পণ্য চুরি |

## লাইসেন্স ও নৈতিকতা

ইউটিউব ভিডিওর কপিরাইট চ্যানেলের। প্রকাশ করবেন শুধু annotations/ (ইউটিউব ID + সময়), ভিডিও নয়।
