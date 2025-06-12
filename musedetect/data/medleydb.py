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


INSTRUMENT_TO_CATEGORY = {
    "": "",
    "flute": "42 Non-free aerophones",
    "french horn": "42 Non-free aerophones",
    "viola section": "32 Composite chordophones",
    "viola": "32 Composite chordophones",
    "toms": "21 Struck membranophones",
    "synthesizer": "53 Radioelectric instruments",
    "gong": "11 Struck idiophones",
    "bamboo flute": "42 Non-free aerophones",
    "alto saxophone": "42 Non-free aerophones",
    "clarinet": "42 Non-free aerophones",
    "gu": "32 Composite chordophones",
    "zhongruan": "32 Composite chordophones",
    "distorted electric guitar": "32 Composite chordophones",
    "trombone": "42 Non-free aerophones",
    "tack piano": "31 Simple chordophones",
    "violin": "32 Composite chordophones",
    "piccolo": "42 Non-free aerophones",
    "fx/processed sound": "53 Radioelectric instruments",
    "vibraphone": "11 Struck idiophones",
    "double bass": "32 Composite chordophones",
    "trombone section": "42 Non-free aerophones",
    "tenor saxophone": "42 Non-free aerophones",
    "darbuka": "21 Struck membranophones",
    "vocalists": "31 Simple chordophones",
    "harmonica": "42 Non-free aerophones",
    "clarinet section": "42 Non-free aerophones",
    "bass drum": "21 Struck membranophones",
    "baritone saxophone": "42 Non-free aerophones",
    "sampler": "53 Radioelectric instruments",
    "flute section": "42 Non-free aerophones",
    "violin section": "32 Composite chordophones",
    "oboe": "42 Non-free aerophones",
    "french horn section": "42 Non-free aerophones",
    "doumbek": "21 Struck membranophones",
    "horn section": "42 Non-free aerophones",
    "female singer": "31 Simple chordophones",
    "cymbal": "11 Struck idiophones",
    "accordion": "41 Free aerophones",
    "cello section": "32 Composite chordophones",
    "guzheng": "32 Composite chordophones",
    "tuba": "42 Non-free aerophones",
    "liuqin": "32 Composite chordophones",
    "clean electric guitar": "32 Composite chordophones",
    "bassoon": "42 Non-free aerophones",
    "glockenspiel": "11 Struck idiophones",
    "auxiliary percussion": "11 Struck idiophones",
    "lap steel guitar": "32 Composite chordophones",
    "banjo": "32 Composite chordophones",
    "yangqin": "31 Simple chordophones",
    "acoustic guitar": "32 Composite chordophones",
    "piano": "31 Simple chordophones",
    "brass section": "42 Non-free aerophones",
    "timpani": "21 Struck membranophones",
    "trumpet section": "42 Non-free aerophones",
    "scratches": "53 Radioelectric instruments",
    "trumpet": "42 Non-free aerophones",
    "erhu": "32 Composite chordophones",
    "electric piano": "31 Simple chordophones",
    "bass clarinet": "42 Non-free aerophones",
    "dizi": "42 Non-free aerophones",
    "mandolin": "32 Composite chordophones",
    "harp": "32 Composite chordophones",
    "drum machine": "53 Radioelectric instruments",
    "electric bass": "32 Composite chordophones",
    "tabla": "21 Struck membranophones",
    "claps": "11 Struck idiophones",
    "bongo": "21 Struck membranophones",
    "male rapper": "31 Simple chordophones",
    "male singer": "31 Simple chordophones",
    "shaker": "11 Struck idiophones",
    "drum set": "21 Struck membranophones",
    "cello": "32 Composite chordophones",
    "oud": "32 Composite chordophones",
    "soprano saxophone": "42 Non-free aerophones",
    "tambourine": "11 Struck idiophones",
    "string section": "32 Composite chordophones",
}


def get_all_instruments():
    return list(INSTRUMENT_TO_CATEGORY.keys())


def hornbostel_sachs(instrument_name):
    return INSTRUMENT_TO_CATEGORY.get(instrument_name)
