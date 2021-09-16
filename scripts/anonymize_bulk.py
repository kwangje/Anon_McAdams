import argparse
from optimize import anonymize
import librosa
import json
import soundfile as sf
import os
import shutil
from tqdm import tqdm


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def parse_args():
    parser = argparse.ArgumentParser(description="lightweight speaker anonymization")
    parser.add_argument("--model", type=str, help="model parameters (*.json)")
    parser.add_argument("--data_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":

    data_path = "/Users/kwang/Desktop/lightweight_spkr_anon/"
    fs = 16000  # sampling frequency
    fn_model = parse_args().model  # model parameters

    src = os.path.join(data_path, "data_16k")
    dst = os.path.join(data_path, "data_anonymized")
    if not os.path.exists(dst):
        os.mkdir(dst)

    copytree(src, dst)
    print("Anonymization: {}---{}-->{}".format(src, fn_model, dst))

    for fol in os.listdir(dst):
        for f in tqdm(os.listdir(os.path.join(dst, fol))):
            file = os.path.join(dst, fol, f)
            print(file)

            # load wav
            # y, sr = librosa.load(file)
            x = librosa.load(file, fs)[0]

            # load model parameters
            params = json.load(open(fn_model, "r"))

            # anonymize
            y = anonymize(x, fs, **params)

            # save wav
            sf.write(file, y, fs, "PCM_16")

            # data = librosa.resample(y, sr, fs)
            # sf.write(file, data, fs)

    # print("Final Path of 16K Dataset = ", dst)
    print("Final Path of Anonymized Dataset = ", dst)

    # fn_wav, fn_wav_out = "data/vctk/p227_001.wav", "anonymized.wav"
    # fn_model = parse_args().model  # model parameters
    # print("Anonymization: {}---{}-->{}".format(fn_wav, fn_model, fn_wav_out))

    # # load wav
    # x = librosa.load(fn_wav, fs)[0]

    # # load model parameters
    # params = json.load(open(fn_model, "r"))

    # # anonymize
    # y = anonymize(x, fs, **params)

    # # save wav
    # sf.write(fn_wav_out, y, fs, "PCM_16")

