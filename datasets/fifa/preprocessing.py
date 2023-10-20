from utils import write_output, split_training_test, read_input_data_from_path

if __name__ == "__main__":
    sequences = read_input_data_from_path("FIFA.txt")
    print(len(sequences))
    seq_train, seq_test = split_training_test(sequences, 16360, 1)
    write_output("seq_train.txt", seq_train)
    write_output("seq_test.txt", seq_test)

