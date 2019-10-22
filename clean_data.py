import csv
import os
import click


def deserialize_data(data_file):
    sentences_raw = [sentence_raw.split("\n") for sentence_raw in data_file.read().split("\n\n")]
    sequences = []
    for sentence in sentences_raw:
        tmp = []
        for token in sentence:
            splited_token = token.split(" ")
            if len(splited_token) == 2:
                tmp.append((splited_token[0], splited_token[1]))
        sequences.append(tmp)
    return sequences


@click.command()
@click.argument("input", type=str)
@click.argument("output", type=str)
@click.option("--split", type=int, help="Split a subset of the input data")
def main(input, output, split):
    """
    Transform space seperate to "#" seperate files
    """
    with open(input, "r") as input_f, open(output, "w") as output_f:
        sentences = deserialize_data(input_f)
        output_data = ["{}\n".format(" ".join(["#".join(token) for token in sentence])) for sentence in sentences]
        output_data = output_data[:split] if split else output_data
        output_f.writelines(output_data)


if __name__ == "__main__":
    main()
