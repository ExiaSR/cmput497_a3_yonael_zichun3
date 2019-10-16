import csv
import os
import click


@click.command()
@click.argument("input", type=str)
@click.argument("output", type=str)
@click.option("--split", type=int, help="Split a subset of the input data")
def main(input, output, split):
    """
    Transform space seperate to "#" seperate files
    """
    with open(input, "r") as input_f, open(output, "w") as output_f:
        output_data = input_f.read().split("\n")  # [line for line in input_f if not line.strip()]
        output_data = [row.split(" ") for row in output_data if row]
        output_data = output_data[:split] if split else output_data

        csv_writer = csv.writer(output_f, delimiter="#")
        csv_writer.writerows(output_data)


if __name__ == "__main__":
    main()
