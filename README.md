# Онцгой цэгт суурилсан авто замын тэмдэг таних систем

## Танилцуулга

Энэхүү төсөл нь “Онцгой цэгт суурилсан авто замын тэмдэг таних программ” сэдэвт бакалаврын дипломын ажлын хэрэгжүүлэлтийн хэсэг юм. :contentReference[oaicite:0]{index=0}

Системийн үндсэн зорилго нь видео дүрсээс авто замын тэмдгийг бодит цагийн горимд илрүүлж, жолоочид анхааруулга өгөхөд оршино. :contentReference[oaicite:1]{index=1}

Энэхүү систем нь deep learning загвар ашиглахын оронд SIFT feature detection, FLANN matcher, RANSAC homography, object tracking зэрэг computer vision аргуудыг ашиглан хэрэгжүүлсэн. :contentReference[oaicite:2]{index=2}

Систем нь:

- Замын тэмдэг илрүүлэх
- Олон тэмдгийг зэрэг таних
- Тэмдэг хүртэлх ойролцоох зай тооцоолох
- Жолоочид анхааруулга харуулах
- Видео болон камерын дүрс боловсруулах боломжуудтай.

---

## Ашигласан технологи

- Python
- OpenCV
- Flask
- NumPy
- Pillow (PIL)

---

## Ажиллуулах заавар

### 1. Virtual environment үүсгэх
python -m venv venv
### 2. Virtual environment идэвхжүүлэх 
Windows дээр:

venv\Scripts\activate

Linux / Mac дээр:

source venv/bin/activate

### 3. Шаардлагатай сангууд суулгах
pip install -r requirements.txt
Хэрэв requirements.txt байхгүй бол дараах сангуудыг суулгана:
pip install flask opencv-python opencv-contrib-python numpy pillow

### 4. Программыг ажиллуулах
python web_server.py

### 5. Browser дээр нээх
http://localhost:5000

## Харагдах үр дүн
Систем ажиллах үед хэрэглэгч видео оруулж боловсруулах боломжтой. Программ видео кадр бүрийг боловсруулж, замын тэмдэг илэрвэл тухайн тэмдгийг хүрээгээр тэмдэглэнэ.

Мөн тэмдэг хүртэлх ойролцоох зайг тооцоолж, дэлгэцийн дээд хэсэгт анхааруулга харуулна.

