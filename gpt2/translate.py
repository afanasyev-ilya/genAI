from datasets import load_dataset

def prepare_translate_dataset():
    print("loading started...")
    # Load the English-Russian translation dataset
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-ru")

    print(f"Dataset loaded successfully!")
    print(f"Train split size: {len(dataset['train'])}")
    print(f"Validation split size: {len(dataset['validation'])}")
    print(f"Test split size: {len(dataset['test'])}\n")

    # Select the train split
    train_data = dataset['train']
    validation_data = dataset['validation']

    postprocessed_train_data = ""

    # Print a few samples from the dataset
    for i in range(len(train_data)):
        postprocessed_train_data += "[st]"
        postprocessed_train_data += f"English: {train_data[i]['translation']['en']}"
        postprocessed_train_data += "[trans]"
        postprocessed_train_data += f"Russian: {train_data[i]['translation']['ru']}"
        postprocessed_train_data += "[end]"
        if i < 5:
            print(f"English: {train_data[i]['translation']['en']}")
            print(f"Russian: {train_data[i]['translation']['ru']}")
            print()
    print(postprocessed_train_data[0:100])

    # Define the file path
    file_path = "en_ru_train_data.txt"

    # Save the data to a text file
    with open(file_path, "w", encoding="utf-8") as file:
        for line in postprocessed_train_data:
            file.write(line)

    print(f"Data saved to {file_path}")


prepare_translate_dataset()