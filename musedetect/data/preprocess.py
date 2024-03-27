import shutil
import warnings
from pathlib import Path

import medleydb_instruments as mdb
import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class Preprocessor:
    def __init__(
        self,
        transform,
        serializer=torch.save,
        serialized_extension=".pt",
    ):
        self.transform = transform
        self.serializer = serializer
        self.serialized_extension = serialized_extension

    @staticmethod
    def _check_overwrite(dest_path, overwrite):
        if dest_path.exists():
            if overwrite:
                shutil.rmtree(dest_path)
            else:
                raise FileExistsError(f"{dest_path} already exists, and overwrite=False.")


class MedleyDBPreprocessor(Preprocessor):
    def _apply_file(self, track, dest_path, data_path):
        activations_df = track.activations
        df = pd.DataFrame(columns=["start_time", "end_time", "instrument"])
        instruments = track.instruments
        # for instrument in instruments:  # Avoid sections, as they aren't individual instruments
        #     if "section" in instrument:
        #         return None
        df_infos = pd.DataFrame(columns=["track_id_frame", "labels"])
        for i in range(len(instruments)):
            time_array = np.array(track.activations["time"])
            # Indice activ pour 1e colonne
            list_activations = np.array(activations_df.values)[:, i + 1] > 0.5
            # indice du temps de début et fin de la présence de l'instrument dans l'extrait
            list_intervals = np.flatnonzero(np.diff(np.r_[0, list_activations, 0]) != 0).reshape(-1, 2) - [0, 1]
            # récuperer un array même format list_intervals mais avec temps plutot que indice :
            list_intervals_temps = time_array[list_intervals]
            # Multiplier par sampling rate pour avoir l'indice dans notre array de musique
            list_indice_extraits = (list_intervals_temps * 44100).astype(int)
            df_to_concat = pd.DataFrame(list_indice_extraits, columns=["start_time", "end_time"])
            df_to_concat["instrument"] = track.instruments[i]
            df = pd.concat([df, df_to_concat], axis=0, ignore_index=True)
        try:
            waveform, _ = torchaudio.load(
                Path(str(data_path) + "/" + str(track.track_name) + "/" + str(track.mix_filename))
            )
        except RuntimeError:
            warnings.warn(f"Failed to load {track.track_name}, make sure the data isn't corrupted.")
            return None
        dest = Path(dest_path) / track.track_name
        dest.parent.mkdir(parents=True, exist_ok=True)  # Create destination directory if necessary
        tot = 0
        for frame, (start, end, data) in enumerate(self.transform(waveform)):
            df_cut = df[(df["start_time"] <= end) & (df["end_time"] >= start)].copy()
            df_cut["diff"] = df["end_time"].clip(upper=end) - df["start_time"].clip(lower=start)
            tot += (df_cut.groupby("instrument")["diff"].sum() / (end - start) < 5 / 100).sum()

            instruments = df[(df["start_time"] <= end) & (df["end_time"] >= start)]["instrument"].unique()
            self.serializer(data, f"{dest}_frame{frame}")
            df_to_add = pd.DataFrame(
                data=[[f"{track.track_name}_frame{frame}", ",".join(instruments)]], columns=["track_id_frame", "labels"]
            )
            df_infos = pd.concat([df_infos, df_to_add], axis=0, ignore_index=True)
        return df_infos

    def apply(self, data_path, dest_path, overwrite=False, show_progress=True, n_jobs=-1):
        dest_path = Path(dest_path)
        self._check_overwrite(dest_path, overwrite)
        list_tracks = [file.stem for file in list(Path(data_path).glob("[!._]*"))]
        dataset = list(mdb.MultiTrack(track_name) for track_name in list_tracks)
        new_dataset = [x for x in dataset if x.has_bleed is False]

        df_infos = pd.DataFrame(columns=["track_id_frame", "labels"])
        list_infos = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._apply_file)(new_dataset[j], dest_path, data_path)
            for j in tqdm(range(len(new_dataset)), disable=not show_progress)
        )
        for info in list_infos:
            if info is not None:
                df_infos = pd.concat([df_infos, info], axis=0, ignore_index=True)
        df_infos.to_csv(str(dest_path) + "/infos")
