from scipy import signal
from scipy.io import wavfile
import numpy as np
import glob

def read_spect_matrix(audio_file_list):

    data = []

    for i,filename in enumerate(audio_file_list):
        
        print ("Done",i,"/",len(audio_file_list))

        sample_rate, samples = wavfile.read(filename)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        
        data.append(spectrogram.T)

    return data

def get_file_list():

    meta_data = np.load('meta_data.npy')
    male_usa_idx  = np.logical_and(meta_data[:, 2]=='m', meta_data[:, 3]=='USA')
    male_usa_list = meta_data[male_usa_idx, 0]
    female_usa_idx  = np.logical_and(meta_data[:, 2]=='f', meta_data[:, 3]=='USA')
    female_usa_list = meta_data[female_usa_idx, 0]
    male_file_list  = []
    female_file_list = []
    male_id = []
    female_id = []
    
    for id in male_usa_list:
        for filename in glob.iglob('vox1_dev/wav/' + id + '/**/*.wav', recursive=True):
            male_file_list.append(filename)
            male_id.append(id)
           

            
    for id in female_usa_list:
        for filename in glob.iglob('vox1_dev/wav/' + id + '/**/*.wav', recursive=True):
            female_file_list.append(filename)
            female_id.append(id)
            
    return male_file_list, female_file_list, male_id, female_id
    
    

if __name__ == '__main__':

    male_file_list, female_file_list,male_id, female_id = get_file_list()
    male_spect   = read_spect_matrix(male_file_list[:10000])
    female_spect = read_spect_matrix(female_file_list[:10000])
    np.save('male_spect.npy', np.array(male_spect))
    
    np.save('female_spect.npy', np.array(female_spect))
    np.save('male_id.npy', np.array(male_id))
    np.save('female_id.npy', np.array(female_id))
    
    
    
    print('done!!!')    
    print(len(male_spect), len(female_spect))
