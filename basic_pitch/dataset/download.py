import argparse

from basic_pitch.dataset import commandline
from basic_pitch.dataset.guitarset import main as guitarset_main
from basic_pitch.dataset.ikala import main as ikala_main
from basic_pitch.dataset.maestro import main as maestro_main
from basic_pitch.dataset.medleydb_pitch import main as medleydb_pitch_main
from basic_pitch.dataset.slakh import main as slakh_main

dataset_dict = {
    'guitarset': guitarset_main,
    'ikala': ikala_main,
    'maestro': maestro_main,
    'medleydb_pitch': medleydb_pitch_main,
    'slakh': slakh_main
}

def main():
    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument("dataset", choices=list(dataset_dict.keys()), help="The dataset to download / process.")
    dataset = dataset_parser.parse_args().dataset

    print(f'got the arg: {dataset}')
    cl_parser = argparse.ArgumentParser()
    commandline.add_default(cl_parser, dataset)
    commandline.add_split(cl_parser)
    known_args, pipeline_args = cl_parser.parse_known_args()  # sys.argv)

    dataset_dict[dataset](known_args, pipeline_args)


if __name__ == '__main__':
    main()
