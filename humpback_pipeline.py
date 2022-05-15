import csv
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime, timedelta
from redun import task, File
from redun.functools import map_
from typing import List


redun_namespace = "humpback_pipeline"


# Pull down model into global namespace to avoid multiple downloads
MODEL_URL = './humpback_model_cache'
model = hub.load(MODEL_URL)


def get_date_file(
        date_string: str,  # format YYYY-MM-DD
        bucket: str = "pacific-sound-16khz",
        prefix: str = "MARS-",
        suffix: str = "T000000Z-16kHz.wav"
) -> File:
    """
    Gets the .wav file name given date
    """
    date_value = datetime.strptime(date_string, '%Y-%m-%d')
    year = date_value.year
    month = date_value.month
    day = date_value.day
    filename = f"{prefix}{year}{month:02d}{day:02d}{suffix}"
    obj_path = f"{year}/{month:02d}/{filename}"
    return File(f"s3://{bucket}/{obj_path}")


def resample(
        chunk_array: np.array,
        orig_sr: int = 16000,
        target_sr: int = 10000
) -> np.array:
    """Resamples the input array to the desired sampling rate"""
    return librosa.resample(
        chunk_array,
        orig_sr=orig_sr,
        target_sr=target_sr,
        scale=True
    )


def predict(
        resampled_chunk: np.array
) -> np.array:
    waveform = np.expand_dims(resampled_chunk, axis=1)
    waveform_exp = tf.expand_dims(waveform, 0)
    pcen_spectrogram = model.front_end(waveform_exp)

    def fit_window(i):
        start = int((119997 / 3600) * i)
        end = start + 128
        if end > pcen_spectrogram.shape[1]:
            # repeat the last window
            context_window = pcen_spectrogram[:, -128:, :]
        else:
            context_window = pcen_spectrogram[:, start:end, :]
        logits = model.logits(context_window)
        probabilities = tf.nn.sigmoid(logits)
        return probabilities.numpy()[0][0]

    score_values = list(map(fit_window, range(3600)))
    return score_values


def add_time(
        date_value: datetime,
        hour: int,
        predictions: List[float]
) -> List[dict]:
    return [
        {"timestamp": date_value + +timedelta(hours=hour) + timedelta(seconds=i),
         "raw_score": x
         } for i, x in enumerate(predictions)
    ]


@task(executor="batch", limits=["batch"])
def process_chunk(
        chunk_number: int,
        wav_file: File,
        date: datetime,
        output_dir: str
) -> File:
    """Process a chunk of audio file from end to end and write out predictions"""
    with wav_file.open('rb') as fp:
        wav = sf.SoundFile(fp)
        frames = wav.frames
        chunk_size = int(frames / 24)
        wav.seek(chunk_number * chunk_size)
        wav_chunk = wav.read(frames=chunk_size, dtype='float32')
    resampled = resample(wav_chunk)
    predictions = predict(resampled)
    predictions_ts = add_time(date, chunk_number, predictions)
    output_file = File(f"{output_dir}/{date.strftime('%Y%m%d')}_{chunk_number}.csv")
    output = write_output(output_file, predictions_ts)
    return output


def write_output(
        output_file: File,
        predictions: List[dict]
) -> File:
    with output_file.open('w') as fp:
        writer = csv.DictWriter(fp, fieldnames=["timestamp", "raw_score"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
    return output_file


@task()
def run_for_date(
        date: str,
        output_dir: str
) -> List[File]:
    date_file = get_date_file(date_string=date)
    date_value = datetime.strptime(date, '%Y-%m-%d')
    chunks = map_(process_chunk.partial(date=date_value,
                                        wav_file=date_file,
                                        output_dir=output_dir,
                                        ), range(24))
    return chunks
