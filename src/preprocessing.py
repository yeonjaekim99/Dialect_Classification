import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import os
import pickle
import wave
import numpy as np
import sklearn

class MFCC():
    def __init__(self, label_dir, wav_dir, save_dir, img_dir):
        ## json file names
        json_file_names = os.listdir(label_dir)
        json_file_names = [j for j in json_file_names if j.endswith('.json')]
        
        ## wav file names
        wav_file_names = os.listdir(wav_dir)
        wav_file_names = [w for w in wav_file_names if w.endswith('.wav')]

        ## wav label dict
        wav_label_dict = dict()
        for json_file_name in json_file_names:
            matched_wav_file = list(filter(lambda x: x[:-3] == json_file_name[:-4], wav_file_names))
            if len(matched_wav_file) == 0: 
                continue
            wav_label_dict[json_file_name] = matched_wav_file[0]
        
        ## self.variable
        self.label_dir = label_dir
        self.wav_dir = wav_dir
        self.save_dir = save_dir
        self.img_dir = img_dir
        self.wav_label_dict = wav_label_dict

    def run(self):
        wav_label_dict = self.wav_label_dict
        save_dir = self.save_dir
        if not os.path.isdir(save_dir): 
            os.mkdir(save_dir)
        img_dir = self.img_dir
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        vis = True

        for i, (label_file, wav_file) in enumerate(wav_label_dict.items(), 1):
            ## set path
            label_path = os.path.join(self.label_dir, label_file)
            wav_path = os.path.join(self.wav_dir, wav_file)
            save_path = os.path.join(save_dir, str(i).zfill(3))
            save_file = os.path.join(save_path, str(i).zfill(3)+"_mfcc.pickle")
            img_path = os.path.join(img_dir, str(i).zfill(3))
            if not os.path.isdir(save_path): 
                os.mkdir(save_path)
            if not os.path.isdir(img_path) and vis:
                os.mkdir(img_path)
            
            ## parshing time info
            time_info = self.parshing_time_info(label_path)

            ## preprocessing
            if i != 1:
                vis = False
            data = self.mfcc_func(wav_path, time_info, img_path, vis)
            self.save_data(data, save_file)

    def parshing_time_info(self, label_path):
        with open(label_path, "r") as f:
            json_data = json.load(f)
        dialog = json_data['utterance']
        time_info = [(d['start'], d['end']) for d in dialog]
        
        return time_info

    def mfcc_func(self, wav_path, time_info, img_path, vis):
        data = []
        i = 0
        for (start, end) in time_info:
            y, sr = librosa.load(wav_path, sr=16000, offset=start, duration=end-start)
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
            # mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            ## modify padding length ##
            pad_mfcc = self.padding(mfcc, 40)
            data.append(pad_mfcc)

            ## save visualize file as jpg ##
            if vis:
                file_name = os.path.join(img_path, str(i)+".jpg")
                self.vis_func(pad_mfcc, file_name)
                i += 1

        return np.array(data)

    def padding(self, data, size):
        if data.shape[1] > size:
            return data[:, 0:size]
        else:
            return librosa.util.pad_center(data, size=size, axis=1)

    def vis_func(self, data, file_name):
        fig = plt.figure()
        img = librosa.display.specshow(data, sr=16000, x_axis='time')
        fig.savefig(file_name)

    def save_data(self, data, path):
        with open(path,"wb") as fw:
            pickle.dump(data, fw)

def main():

    ## data dir path ##
    ### modify this ###
    label_dir = './dataset/gangwon_label'
    wav_dir = './dataset/gangwon_1'
    save_dir = './gangwon'
    img_dir = './gangwon_img'

    ## preprocessing ## 
    mfcc = MFCC(label_dir, wav_dir, save_dir, img_dir)
    mfcc.run()

if __name__ == '__main__':
    main()
