from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class MedleyDBDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        hierarchy: bool = False,
        class_names: list[str] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.hierarchy = hierarchy
        df_labels = pd.read_csv(self.root / "infos").fillna("")
        if class_names is None:
            self.class_names = sorted({x for label_str in df_labels["labels"].values for x in label_str.split(",")})
        else:
            self.class_names = class_names
        self.aggregated_class_names = []
        self.class_name_to_aggregated_idx = []
        if self.hierarchy:
            self.aggregated_class_names = sorted(set(hornbostel_sachs(class_name) for class_name in self.class_names))
            for class_name in self.class_names:
                agg_idx = self.aggregated_class_names.index(hornbostel_sachs(class_name))
                self.class_name_to_aggregated_idx.append(agg_idx)

        self.files = df_labels.track_id_frame.values.tolist()
        self.labels = [self.vector_labels(label) for label in df_labels.labels.values]

    def vector_labels(self, instrument_label):
        list_instruments = str(instrument_label).split(",")
        vector = torch.zeros(len(self.class_names) + len(self.aggregated_class_names))
        offset = len(self.class_names)
        for i, class_name in enumerate(self.class_names):
            if class_name in list_instruments:
                vector[i] = 1
                if self.class_name_to_aggregated_idx:
                    vector[self.class_name_to_aggregated_idx[i] + offset] = 1
        return vector

    def __getitem__(self, index):
        x = torch.load(self.root / self.files[index]).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[index]

    def __len__(self):
        return len(self.files)


def hornbostel_sachs(instrument_name):
    instrument_classification = {
        "": "",
        "accordion": "41 Free aerophones",
        "acoustic guitar": "32 Composite chordophones",
        "alto saxophone": "42 Non-free aerophones",
        "auxiliary percussion": "11 Struck idiophones",
        "bamboo flute": "42 Non-free aerophones",
        "banjo": "32 Composite chordophones",
        "baritone saxophone": "42 Non-free aerophones",
        "bass clarinet": "42 Non-free aerophones",
        "bass drum": "21 Struck membranophones",
        "bassoon": "42 Non-free aerophones",
        "bongo": "21 Struck membranophones",
        "brass section": "42 Non-free aerophones",
        "cello": "32 Composite chordophones",
        "cello section": "32 Composite chordophones",
        "claps": "11 Struck idiophones",
        "clarinet": "42 Non-free aerophones",
        "clarinet section": "42 Non-free aerophones",
        "clean electric guitar": "32 Composite chordophones",
        "cymbal": "11 Struck idiophones",
        "darbuka": "21 Struck membranophones",
        "distorted electric guitar": "32 Composite chordophones",
        "dizi": "42 Non-free aerophones",
        "double bass": "32 Composite chordophones",
        "doumbek": "21 Struck membranophones",
        "drum machine": "53 Radioelectric instruments",
        "drum set": "21 Struck membranophones",
        "electric bass": "32 Composite chordophones",
        "electric piano": "31 Simple chordophones",
        "erhu": "32 Composite chordophones",
        "female singer": "31 Simple chordophones",
        "flute": "42 Non-free aerophones",
        "flute section": "42 Non-free aerophones",
        "french horn": "42 Non-free aerophones",
        "french horn section": "42 Non-free aerophones",
        "fx/processed sound": "53 Radioelectric instruments",
        "glockenspiel": "11 Struck idiophones",
        "gong": "11 Struck idiophones",
        "gu": "32 Composite chordophones",
        "guzheng": "32 Composite chordophones",
        "harmonica": "42 Non-free aerophones",
        "harp": "32 Composite chordophones",
        "horn section": "42 Non-free aerophones",
        "lap steel guitar": "32 Composite chordophones",
        "liuqin": "32 Composite chordophones",
        "male rapper": "31 Simple chordophones",
        "male singer": "31 Simple chordophones",
        "mandolin": "32 Composite chordophones",
        "oboe": "42 Non-free aerophones",
        "oud": "32 Composite chordophones",
        "piano": "31 Simple chordophones",
        "piccolo": "42 Non-free aerophones",
        "sampler": "53 Radioelectric instruments",
        "scratches": "53 Radioelectric instruments",
        "shaker": "11 Struck idiophones",
        "soprano saxophone": "42 Non-free aerophones",
        "string section": "32 Composite chordophones",
        "synthesizer": "53 Radioelectric instruments",
        "tabla": "21 Struck membranophones",
        "tack piano": "31 Simple chordophones",
        "tambourine": "11 Struck idiophones",
        "tenor saxophone": "42 Non-free aerophones",
        "timpani": "21 Struck membranophones",
        "toms": "21 Struck membranophones",
        "trombone": "42 Non-free aerophones",
        "trombone section": "42 Non-free aerophones",
        "trumpet": "42 Non-free aerophones",
        "trumpet section": "42 Non-free aerophones",
        "tuba": "42 Non-free aerophones",
        "vibraphone": "11 Struck idiophones",
        "viola": "32 Composite chordophones",
        "viola section": "32 Composite chordophones",
        "violin": "32 Composite chordophones",
        "violin section": "32 Composite chordophones",
        "vocalists": "31 Simple chordophones",
        "yangqin": "31 Simple chordophones",
        "zhongruan": "32 Composite chordophones",
    }

    return instrument_classification.get(instrument_name)
