from ast import Global
from urllib import response
from flask import request,Flask, Response, json, jsonify, make_response
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model 
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from itertools import count
import gc
from datetime import date, datetime



app = Flask(__name__)

model_path = "./weights/weights.h5"
model = load_model(model_path)
    
fps = 30
sec = 4

def save_shorts(video_path, shorts_unique, start_time,upload_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3    

    segRange = [(shorts_idx[0], shorts_idx[0] + fps*sec*2) for shorts_idx in shorts_unique]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    mk_file = []
    time_list = []
    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(f'{upload_path[:-4]}_{idx}.mp4',fourcc,fps,(w, h))
        mk_file.append(f'{upload_path[:-4]}_{idx}.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        time_ = 0
        
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_ = np.datetime64(start_time) + np.timedelta64(int(1000 * frame_number / fps), 'ms')
            cv2.putText(frame, str(time_)[:-3], org=(10, 80), fontFace=1, fontScale=5, color=(0, 0, 0), thickness=4)

            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        time_list.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
        writer.release()
    return mk_file,time_list

def shorts_intersect(shorts_idx):
    shorts_unique = [shorts_idx[0]]
    for i in range(1, len(shorts_idx)):
        intersect = np.intersect1d(shorts_unique[-1], shorts_idx[i])
        if len(intersect) >= 1:
            continue
        else:
            shorts_unique.append(shorts_idx[i])

    return shorts_unique

def get_shorts(pred_lst,fps):

    shorts_len = fps * sec
    shorts_idx = []
    for idx in range(len(pred_lst) - shorts_len):
        shorts = pred_lst[idx:idx+shorts_len].tolist()
        
        if shorts.count(1) >= 30:
            shorts_idx.append(range(idx, idx+shorts_len*2))
    
    if len(shorts_idx):
        shorts_unique = shorts_intersect(shorts_idx)
        #print(shorts_unique)
        return shorts_unique
    else:
        return None

def video_show(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c = 3
    
    threshold=0.95
    pred_lst = []
    tmp = 0
    batch = np.empty((1, 512, 512, 3))
    batch_size = 4

    while True:
        ret, frame = cap.read()
        #print("Predicting")
        if ret:
            img = frame.copy()
            img = cv2.resize(img, (512, 512))
            img = img.astype(np.float32)
            img /= 255.
            img = np.expand_dims(img, axis=0)
            y_pred = model.predict(img, verbose=0)
            y_pred = (np.array(y_pred)[:, 1] > threshold)            
            pred_lst.append(y_pred)
        else:
            break

        #cv2.putText(frame, violation[y_pred], org=(10, 80), fontFace=1, fontScale=5*(y_pred+1), color=color_list[y_pred], thickness=4)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(1)
    pred_lst = np.reshape(pred_lst, (-1, ))

    return pred_lst,int(fps)

def predict_violation(frame, model):
    img = frame.copy()
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)
    img /= 255.
    img = np.expand_dims(img, axis=0)

    y_pred = model.predict(img, verbose=0)
    
    return np.argmax(y_pred)      # 0: normal, 1: violation

def binary_search(arr,target,low=None,high=None):
    low, high = low or 0, high or len(arr) - 1
    if low > high:
        return low

    mid = (low + high) // 2
    if arr[mid] > target:
        return binary_search(arr, target, low, mid)
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search(arr, target, mid + 1, high)

@app.route('/been',methods=['POST'])
def predict_play():
    print("start",flush=True)
    data = {'path':'https://aizostorage.blob.core.windows.net/aizo-cropped/kakaotalk'
            ,'lat':'80.12'
            ,'lon':'190.1'
            ,'time':'2022-02-12 20:20:20'}
    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response



@app.route('/play',methods=['POST'])
def temp():
    temp = request.args.get("path")
    print(temp)
    params = request.get_json()
    params = json.loads(params)
    # params = request.get_json('')
    # params = json.loads()
    #print(params,flush=True)   
    #print(type(params))
    #print(params['path'])
    #gps = params['gps']
    #data = params['data']
    #times = []
    # for i in data:
    #     times.append(float(i['time']))
    # # 본 서버용

    download_file_path = './temp/'+str(params["path"])
    connect_str = os.getenv("STORAGE_CONNECTION_STRING")    
    #print(connect_str)
    file_name = params['path']
    download_container = os.getenv('STORAGE_AZURE_CONTAINER')
    upload_container = os.getenv('STORAGE_CROPPED_CONTAINER')

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(download_container)
    result = {'upload': 'fail'}
    try:
        with open(download_file_path, "wb") as download_file:
            download_file.write(container_client.download_blob(file_name).readall())
            download_file.close()
            result['upload']='succeced'
    except:
        print("다운로드 실패!")
        return make_response(jsonify(result), 503)
    
    # start = time.time()
    upload_file_path = './media/'+params['path']
    video_path = download_file_path
    
    print("start",flush=True)
    pred_lst, fps = video_show(video_path, model)
    shorts_unique = get_shorts(pred_lst,fps)
    print("detected succ",flush=True)
    mk_file_list = []
    gps_time_list =[]
    result_time = []
    gps_list = []
    if shorts_unique is not None:
        mk_file_list, gps_time_list = save_shorts(video_path, shorts_unique,params['time'],upload_file_path)
    else:
        result =  {
        "path" : [" "]
        ,"gps" : [" "]
        ,"date" : [" "]
        }
        result_dump = json.dumps(result)
        return make_response(result_dump, 204)
    print(mk_file_list)
    print("upload_end",flush=True)
    for j in gps_time_list:
        result_time.append(str(np.datetime64(params['time'])+np.timedelta64(int(1000*j))))
    # for i in gps_time_list:
    #     gps_list.append(binary_search(times,i))

    for file_path in mk_file_list:
        file_name = file_path.split('/')
        #print(file_name[-1])
        temp = file_name[-1]
        blob_client = blob_service_client.get_blob_client(container=upload_container,blob=temp)
        with open(file_path, 'rb') as data:
            blob_client.upload_blob(data)
            data.close()

    for file_path in mk_file_list:
        os.remove(file_path)
    for i in range(0,len(mk_file_list)):
        mk_file_list[i] = "https://aizostorage.blob.core.windows.net/aizo-cropped/"+ mk_file_list[i].split('/')[-1]
        gps_list.append({"lat":"37.5664","lon":"126.9851"})

   ## GPS, time 시간 처리 미구현
#    다운로드 링크 받아서 처리해줘야함
#    with open()
    os.remove(download_file_path)
    result = {
        "path" : mk_file_list
        ,"gps" : gps_list
        ,"date" : result_time
    }
    return make_response(jsonify(result), 201)

if __name__== '__main__':
    app.run(debug = True, port=8080)
