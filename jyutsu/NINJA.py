import cv2
from ultralytics import YOLO
import numpy as np 
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image


####学習済みモデルの読み込み
model = YOLO("models/NINJA.pt")

####カメラのデバイス番号を指定 (0はデフォルトの内蔵カメラ)
cap = cv2.VideoCapture(0)

###印検出用のキュー(キューの中にリアルタイムに認識した印の情報を入れていく)
sign_display_queue = deque(maxlen=15) ###最大長15のキュー
chattering_check_queue = deque(maxlen=3)#印検出用のキューの大きさを３にする。
for index in range(-1, -4, -1):         #-1から-4まで1ずつ小さくしながら、印検出用のキューに格納していく。[-1,-2,-3]となる。
    chattering_check_queue.append(index)

###時刻の変数初期化##########################################　
sign_interval_start =0
jyutsu_interval_start = 0

##術のエフェクト画像のサイズを調整する関数（入力した画像のアスペクト比を維持したまま任意の高さに修正）
def scale_to_height(img,height):
    h,w = img.shape[:2]
    width = round(w * (height/h))
    dst = cv2.resize(img, dsize=(width, height))
    return dst

####術のエフェクト画像#######################################
Katon_Goukakyuu= cv2.imread("jyutsu/Katon_Goukakyuu.webp")
Chidori = cv2.imread("jyutsu/Chidori.webp")
Bunnshinn = cv2.imread('jyutsu/Bunnshinn.JPG')
Kuchiyose = cv2.imread('jyutsu/Kuchiyose.JPG')

###クラス名を取得する関数#####################################
def get_cls(number):
    mapping = {0: " 未 ", 1: " 亥 ", 2:" 戌 ", 3: " 巳 ", 4:" 子 ", 5:' 申 ', 6:' 辰 ', 7:' 寅 ', 8:' 酉 ',9:' 卯 ',10:' 午 ',11:' 丑 '}
    return mapping.get(number)

####cv2で日本語表示をするメソッド##############################
class CvPutJaText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls, cv_image, text, point, font_path, font_size, color=(0,0,0)):
        font = ImageFont.truetype(font_path, font_size)
        
        cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)
        
        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)
        
        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

        return cv_bgr_result_image

####術を判定する関数(術の文字列、エフェクト画像を返す）############
    
def check_jyutsu(sign_display_queue):
    if list(sign_display_queue)[-6:] == [3,0,5,1,10,7]:
        jyutsu = " 火遁・豪火球の術!!"
        jyutsu_img = Katon_Goukakyuu
    elif list(sign_display_queue)[-3:] == [11,9,5]:
        jyutsu = " 千鳥!!"
        jyutsu_img = Chidori
    elif list(sign_display_queue)[-3:] == [0,3,7]:
        jyutsu = " 分身の術"
        jyutsu_img = Bunnshinn
    elif list(sign_display_queue)[-5:] == [1,2,8,5,0]:
        jyutsu = " 口寄せの術！"
        jyutsu_img = Kuchiyose
    else:
        jyutsu = None
        jyutsu_img = None
    return jyutsu,jyutsu_img

###表示する画像を生成する関数#######################################
###yoloで推論した画像を土台に、フッター画像の作成とエフェクトの描画を行う##

def draw_results_image(results_img,sign_display_queue):
    font = "font/GenShinGothic-Bold.ttf"

    #推論後の画像の高さ、幅を取得
    frame_height, frame_width, _  = results_img.shape 

    #フッター作成
    footer_image = np.ones((int(frame_height / 10), frame_width, 3), np.uint8)*255
    
    #印の履歴文字列作成
    sign_display = ''   
    if len(sign_display_queue)>0:
        for sign_id in sign_display_queue:
            sign_display = sign_display + get_cls(sign_id)

    #フッターに術または印の履歴を表示
    jyutsu,jyutsu_img= check_jyutsu(sign_display_queue)
    if jyutsu is not None:
        footer_image = CvPutJaText.puttext(footer_image,jyutsu,(5,3),font,40,(0,0,0))
    else:
        footer_image = CvPutJaText.puttext(footer_image, sign_display, (5, 3),font, 40, (0, 0, 0))
    
    # 術の画像をメイン画面に表示
    if jyutsu_img is not None:
        #エフェクト画像のサイズを修正して高さ、幅を取得
        frame_height, frame_width = results_img.shape[:2]
        jyutsu_img = scale_to_height(jyutsu_img,frame_height)
        height, width = jyutsu_img.shape[:2]

        # 描画位置を計算（中心に描画）
        start_y = int((frame_height - height) / 2)
        start_x = int((frame_width - width) / 2)
        # 画像を重ねて描画
        results_img[start_y:start_y+height, start_x:start_x+width] = jyutsu_img

    # フッターを結合する
    display_img = np.vstack((results_img, footer_image))

    return display_img


###カメラスタート############################################
while True:
    # フレームの読み込み
    ret, frame = cap.read()

    # フレームが正常に読み込めなければ終了
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # フレームを左右反転させる
    flipped_frame = cv2.flip(frame, 1)  # 1は水平方向に反転

    # YOLOで物体検出を行う
    results = model(flipped_frame)#学習の実行
    results_img = results[0].plot()###推論結果の画像

    #検出された時だけ、クラスidを保存する。
    if len(results[0].boxes.cls)>0:    
        class_id = int(results[0].boxes.cls[0])###クラスid番号(cls[0]だけ取得する。一度に複数はなし。
    
        #検出内容の履歴追加#############################
        # 3回数以上、同じ印が続いた場合に、印検出とみなす
        chattering_check_queue.append(class_id)#長さ3のキューに検出したclass_idを入れる。
        if len(set(chattering_check_queue)) == 1:
           #set関数でキューの中にある集合にまとめる（同じ要素は一つにまとめられる）。
           #lenで長さにする。lenが2以上の場合、直近3回の検出の中に異なる印が含まれていたことになる

        # 前回と異なる印の場合のみ表示用のキューに登録（max15個）
            if time.time()-jyutsu_interval_start >5 and (len(sign_display_queue
                   ) == 0 or sign_display_queue[-1] != class_id):#術検出から5秒以上経過かつ、キュー中身が空または右端のidが現在のidと同じではない時に追加
                sign_display_queue.append(class_id)
                sign_interval_start = time.time()  # 印の最終検出時間

    # 前回の印検出から指定時間が経過した場合、履歴を消去 ####################
    if (time.time() - sign_interval_start) > 10: #最後の印検出から10秒経過したらキューをクリアする
        sign_display_queue.clear()

    ###術の検出時刻
    jyutsu, jyutsu_img = check_jyutsu(sign_display_queue)
    if jyutsu is not None and jyutsu_interval_start ==0:
        jyutsu_interval_start = time.time() #術検出の初回の時刻を記録しておく。
    elif jyutsu is None:
        jyutsu_interval_start = 0           #術が検出されなくなったら0に戻す

    #術の検出４秒経過後に、キューが術のままの場合はクリア。※術検出後５秒はキューが変わらないようにしてるので確実に消える。
    if jyutsu is not None and time.time()-jyutsu_interval_start >4:
        sign_display_queue.clear()

    ###画像の表示
    display_img = draw_results_image(results_img,sign_display_queue)
    cv2.imshow('NINJA_IN', display_img)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放
cap.release()
cv2.destroyAllWindows()

