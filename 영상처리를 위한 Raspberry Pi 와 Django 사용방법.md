## 영상처리를 위한 Raspberry Pi 와 Django 사용방법

### Raspberry Pi

- Raspberry Pi  4 를 통해 프로젝트를 진행
- opencv 와 python 가 dependence 하게 묶여 있기 때문에 가급적 opencv 와 관련된 라이브러리는 건들이지 않는 것을 추천함



#### Raspberry Pi 실행방법

1. 카메라, 랜선, 키보드등 주변기기장치를 라즈베리파이에 붙임

2. 라즈베리파이를 실행 
   - 라즈베리파이는 전원버튼이 없으며 전원포트에 입력만하면 바로 실행됨

3. 라즈베리파이에 입력
   - sudo service ssh restart 
   - (ssh 재시작) 명령어 입력

4. putty를 통해 원격 연결
   - ip address : 164.125.154.221
   - port : 1022
   - id : pi
   - passworad : locslab

> **참고** : 현재 와이파이에서 포트포워딩을 통해 아이피를 하나 할당 받은 것이므로 혹시 접속이 문제가되면 iptime 접속을 통해 꼭 확인해볼것 `(잘 이해가 안되거나 혹시 아이피를 확인할일이 있으면 기영씨의 도움을 받는것을 추천드림)`

5. cd /home/pi/Workspace/mjpg-streamer (디렉토리 이동)

6. mjpg_streamer -i "input_uvc.so" -o "output_http.so -p 8090 -w /usr/local/share/mjpg-streamer/www/"
   (실시간 영상처리)

7. http://164.125.154.221:8090/에 접속

   - 실시간으로 화면이 보이는지 반드시 확인
   - 해당 사이트는 단순히 스트리밍을 웹상에 보여주는 형태
   - 따라서 실시간으로 django에서 받아와 deep learning 모델과 연동해 얼굴을 인식 및 나이/연령 추정 알고리즘 실행

   



### Django

- 아직 프론트엔드랑 연동되지않음
- 레거시 faceyolo 로 mysql 에 실시간으로 예측된 값을 저장하고 있음 (수정 필요)



#### Django 실행방법

1. *Face_Detection_Web* 에서 *backend* 로 이동 (django 세팅 완료)

> **참고** : django에 대한 내용을 학습하기 어려울 수도 있을 꺼라 판단하여 관련된 ppt를 첨부함 
>
> - 먼저 django 공식 홈페이지에 나온 짧은 예문을 보고 스터디를 진행
> - *Introdution to Django.pdf*, *How does a backend word.pdf*, *Django ORM.pdf*, *Part of Django-Settings, Urls, Apps.pdf*
> - ppt 위치는 nas에 /데이터백업/개발/이철희/Nomadclone PDFs 를 들어가보면 됨
> - 어디까지나 참고사항이니 각자 원하는데로 스터디를 진행하는 것도 좋은 방법이라고 생각됨

2. *face* 안에 얼굴인식과 관련된 코드가 있음

   - veiws.py 를 보면

   ```python
   def get_model():
       store_path = os.path.join('.', 'face', 'detection', 'store')
       model_name = 'model_fr_net-batch_size-400_lr-0.01_004965.pth'
       model_path = os.path.join(store_path, model_name)
       checkpoint = torch.load(model_path, map_location='cpu')
       model_type = checkpoint['model_type']
       model_parameter = checkpoint['model_parameter']
   
       m = None
       if model_type == 'vgg':
           from face.detection.models.vgg import VGG
           m = VGG(**model_parameter)
       elif model_type == 'fr_net':
           from face.detection.models.face_recognition import FRNet
           m = FRNet()
       m.load_state_dict(checkpoint['model_state_dict'])
       m.eval()
       return m
   
   
   video_info = {}
   model = get_model()
   
   ```

   - 모델을 불어오는 코드

   - django 는 웹서버이기 이고 접속자가 들어올때마다 모델을 loading 하면 안된다는 것을 반드시 상기할 필요가 있음
   - 따라서 모델이 실행시 한번 로딩하고 여러API에서 사용할 수 있도록 전역변수로 지정

   

   ```python
   class VideoStreamingView(APIView):
       def get(self, request, device_id):
           try:
               device = Device.objects.get(id=device_id)
           except Device.DoesNotExist:
               return Response(status=status.HTTP_404_NOT_FOUND)
   
           if device.src_url not in video_info.keys():
               try:
                   video_info[device.src_url] = VideoCamera(url=device.src_url)
               except ModuleNotFoundError:
                   return Response(status=status.HTTP_400_BAD_REQUEST)
           cam = video_info[device.src_url]
   
           try:
               return StreamingHttpResponse(
                   video_generator(cam),
                   content_type="multipart/x-mixed-replace;boundary=frame",
                   status=status.HTTP_200_OK
               )
           except AttributeError:
               pass
           except Exception as e:
               return Response(status=status.HTTP_400_BAD_REQUEST)
   ```

   - 직접적인 VideoStreaming API View 

   

   ```python
   class VideoCamera(object):
       def __init__(self, url):
           try:
               self.device = Device.objects.get(src_url=url)
           except Device.DoesNotExist:
               raise ModuleNotFoundError('Device not found...')
   
           self.video = cv2.VideoCapture(url + '/?action=stream')
           self.global_faces_objs = []
           (self.grabbed, self.frame) = self.video.read()
           threading.Thread(target=self.update, args=()).start()
   
       def __del__(self):
           self.video.release()
   
       def get_frame(self):
           image = self.frame
           image = self.face_detection(image)
           ret, jpeg = cv2.imencode('.jpg', image)
   
           return jpeg.tobytes()
   
       @staticmethod
       def image_loader(img_paths):
           from PIL import Image
           from torchvision import transforms
   
           img_size = 64
           image_set = []
           for img_path in img_paths:
               img = Image.open(img_path)
               preprocess = transforms.Compose([transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
   
               img = preprocess(img)
               image_set.append(img)
           img = torch.stack(image_set, dim=0)
           return img
   
       def face_detection(self, image):
           # saving database
           second = timezone.localtime().now().second
           if second % 10 == 0 and len(self.global_faces_objs) >= 40:
               Face.objects.bulk_create(self.global_faces_objs)
               self.global_faces_objs = []
   
           faces = face_recognition.api.face_locations(image, model='cnn')
           if len(faces) == 0:
               return image
   
           face_objs = OrderedDict()
           for i, (t, r, b, l) in enumerate(faces):
               roi_color = image[t:b, l:r]
               # roi_color = image[b:r+l, t:b+t]
   
               # Face image saving
               suffix = str(uuid.uuid4())
               img_path = os.path.join(MEDIA_ROOT, suffix + '.jpg')
   
               cv2.imwrite(img_path, roi_color)
   
               face_objs[img_path] = {
                   'device': self.device,
                   'date': timezone.localtime(),
               }
               cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
   
   ```

   - 실시간으로 스트리밍되는 영상을 처리하는 부분

   - http://164.125.154.221:8090/?action=stream 에서 실시간으로 스트리밍되는 영상을 처리하는 부분

   - 스트리밍되는 영상을 사진형태로 바꾼후 `faces = face_recognition.api.face_locations(image, model='cnn')` 를 통해 얼굴 인식

     - 스트리밍되는 영상을 중간에 처리를 하기 때문에 조금 버벅될수 있지만 크게 문제될 정도는 아님
     - yoloface 사용

   - 얼굴인식되면 얼굴부분을 짤라서 나이 및 성별 추정

     - classification 코드와 모델은 *face/detection* 에 있음
     -  https://github.com/dlcjfgmlnasa/Detection_Face_Information 에 detection 파일 참고

     

3. django 실행